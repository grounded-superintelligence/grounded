"""
A dataset abstraction over GSI hand tracking (v2) outputs.

Hand tracking v2 replaces the per-frame ``left``/``right`` dict format of the
v0.2.x SDK with a flat, MANO-parametric per-frame npz, and it replaces the
rectified left-front reference frame with the **unrectified left-front camera
frame**.

Scope: hand tracking only. SLAM and depth are served by separate APIs and are
intentionally not loaded here.

On-disk layout (per session, per segment)::

    {session}/
      processed-segment{N}/
        hand_v2_outputs.tar                      # as stored on S3
        hand/                                    # tar extracted here
          hand_tracking/
            refinement/params/frame_{i:06d}.npz  # per-frame MANO fits
            save_dataset/
              camera_params.npz                  # intrinsics + extrinsics (all 4 cams)
              continuous_intervals.json          # inclusive [start, end] runs, both hands valid
              yield.json                         # per-hand validity counts
              {left,right}_{front,eye}.mp4       # unrectified (undistorted pinhole) streams
            pose_cleaning/metrics_pose_cleaning.json
"""

import hashlib
import json
import os
import posixpath
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np

if TYPE_CHECKING:
    from grounded.data.processing import AssetDownload, EpisodeDownload

try:  # torch is a declared dependency of the SDK, but hand-only readers can live without it
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object

GROUNDED_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/data/")
LOCKS_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/locks/")

HAND_TAR_NAME_DEFAULT = "hand_v2_outputs.tar"
CLIPPED_HAND_SCHEMA_VERSION = "grounded.episode.hand_clip.v1alpha1"
CLIPPED_POSE_TAR_NAME = "pose_frames.tar"
CLIPPED_POSE_PREFIX = ("hand", "pose_interpolation", "params")
CLIPPED_POSE_FILENAME = re.compile(r"frame_(\d{6,})\.npz")

# Camera naming follows the v2 pipeline (underscores). Order matters: it is the
# view axis of ``HandPose.inlier_mask``.
HAND_CAMS = ["left_front", "right_front", "left_eye", "right_eye"]

SIDES = ("left", "right")


# =============================================================================
# Frame-level dataclasses
# =============================================================================


@dataclass
class HandPose:
    """A single hand's MANO fit for one frame.

    All 3D quantities are metric and expressed in the **unrectified left-front
    camera frame** (x right, y down, z forward). ``keypoints3d[0]`` is the
    wrist and equals ``transl``.
    """

    side: str  # "left" | "right"
    keypoints3d: np.ndarray  # (21, 3) float32, MANO-21 joints
    vertices: np.ndarray  # (778, 3) float32, MANO mesh vertices
    global_orient: np.ndarray  # (3, 3) float32, root rotation
    transl: np.ndarray  # (3,) float32, root translation (== wrist)
    hand_pose: np.ndarray  # (15, 3, 3) float32, articulated joint rotations
    betas: np.ndarray  # (10,) float32, per-frame fitted MANO shape
    source_view: str  # camera the fit was primarily sourced from, one of HAND_CAMS
    inlier_mask: np.ndarray  # (4, 21) bool, per-view keypoint inliers; view axis == HAND_CAMS
    is_detected: bool  # False when pose cleaning rejected this fit
    reason: str  # cleaning rejection reason, "" when valid
    hand_frame_idx: int  # frame the fit was sourced from (== frame_idx unless borrowed)


@dataclass
class HandFrameData:
    """Dataclass holding all synchronized hand tracking data for a single frame.

    RGB fields are ``None`` for cameras not in ``active_cameras``. Hand fields
    are ``None`` when the tracker produced no fit for that hand on this frame
    (the hand is absent from the npz), or - if the episode was constructed with
    ``valid_only=True`` - when pose cleaning rejected the fit.
    """

    frame_idx: int
    left_front_rgb: Optional[np.ndarray]
    right_front_rgb: Optional[np.ndarray]
    left_eye_rgb: Optional[np.ndarray]
    right_eye_rgb: Optional[np.ndarray]
    left_hand: Optional[HandPose]
    right_hand: Optional[HandPose]


# =============================================================================
# Path management
# =============================================================================


class HandPathManager:
    """Utility for resolving hand tracking v2 sub-paths for a session segment."""

    def __init__(self, session_dir: str, segment: int):
        self.session_dir = str(Path(session_dir).expanduser())
        self.segment = segment
        self.segment_dir = os.path.join(self.session_dir, f"processed-segment{segment}")

        self.hand_tracking_dir = os.path.join(self.segment_dir, "hand")

        self.params_dir = os.path.join(self.hand_tracking_dir, "pose_interpolation", "params")
        self.save_dataset_dir = os.path.join(self.hand_tracking_dir, "save_dataset")
        self.camera_params_npz = os.path.join(self.save_dataset_dir, "camera_params.npz")
        self.continuous_intervals_json = os.path.join(self.save_dataset_dir, "continuous_intervals.json")
        self.yield_json = os.path.join(self.save_dataset_dir, "yield.json")
        self.pose_cleaning_json = os.path.join(self.hand_tracking_dir, "pose_cleaning", "metrics_pose_cleaning.json")

    def param_file(self, frame_idx: int) -> str:
        return os.path.join(self.params_dir, f"frame_{frame_idx:06d}.npz")

    def video_file(self, camera: str) -> str:
        return os.path.join(self.save_dataset_dir, f"{camera}.mp4")


class ClippedHandPathManager:
    """Resolve a downloaded episode's flat hand lane without rebuilding a session tree."""

    def __init__(self, lane_dir: Union[str, os.PathLike], params_dir: Union[str, os.PathLike], source_frame_start: int):
        self.lane_dir = str(Path(lane_dir).expanduser().resolve())
        self.segment_dir = self.lane_dir
        self.hand_tracking_dir = self.lane_dir
        self.save_dataset_dir = self.lane_dir
        self.params_dir = str(Path(params_dir).expanduser().resolve())
        self.source_frame_start = source_frame_start

        self.camera_params_npz = os.path.join(self.lane_dir, "camera_params.npz")
        self.continuous_intervals_json = os.path.join(self.lane_dir, "source_continuous_intervals.json")
        self.yield_json = os.path.join(self.lane_dir, "source_yield.json")
        # Pose-cleaning metrics are not part of the clipped-lane contract.
        self.pose_cleaning_json = os.path.join(self.lane_dir, "source_pose_cleaning_metrics.json")

    def param_file(self, frame_idx: int) -> str:
        source_frame_idx = self.source_frame_start + frame_idx
        return os.path.join(self.params_dir, f"frame_{source_frame_idx:06d}.npz")

    def video_file(self, camera: str) -> str:
        return os.path.join(self.lane_dir, f"{camera}.mp4")


def _clipped_manifest_declarations(manifest: dict) -> tuple[List[str], List[str]]:
    """Return supported camera files and flat source sidecars declared by a hand clip."""

    videos = manifest.get("videos")
    if not isinstance(videos, dict):
        raise ValueError("Clipped hand manifest must publish a videos object")
    declared_camera_files = [f"{camera}.mp4" for camera in HAND_CAMS if f"{camera}.mp4" in videos]

    raw_sidecars = manifest.get("source_sidecars")
    if not isinstance(raw_sidecars, list):
        raise ValueError("Clipped hand manifest must publish a source_sidecars list")
    declared_sidecars: List[str] = []
    for value in raw_sidecars:
        if not isinstance(value, str) or not value or value in {".", ".."}:
            raise ValueError("Clipped hand source_sidecars entries must be non-empty filenames")
        if "/" in value or "\\" in value or Path(value).name != value:
            raise ValueError(f"Clipped hand source sidecar must be a flat filename: {value!r}")
        if value in declared_sidecars:
            raise ValueError(f"Clipped hand source sidecar is declared more than once: {value!r}")
        declared_sidecars.append(value)
    if "camera_params.npz" not in declared_sidecars:
        raise ValueError("Clipped hand manifest must declare camera_params.npz as a source sidecar")
    return declared_camera_files, declared_sidecars


def _validate_hand_dir(paths: HandPathManager, active_cameras: List[str]) -> bool:
    """Cheap structural validation of an extracted hand tracking segment."""
    required = [
        paths.params_dir,
        paths.camera_params_npz,
        paths.continuous_intervals_json,
        paths.yield_json,
    ]
    required += [paths.video_file(cam) for cam in active_cameras]
    if not all(os.path.exists(p) for p in required):
        return False

    try:
        with open(paths.yield_json, "r") as f:
            total = int(json.load(f)["total_frames"])
    except Exception:
        return False
    if total <= 0:
        return False
    # spot-check first/last per-frame files rather than all of them
    return os.path.exists(paths.param_file(0)) and os.path.exists(paths.param_file(total - 1))


# =============================================================================
# Download
# =============================================================================


def download_hand_segment(
    session_uri: str,
    segment: int,
    target_dir: str = GROUNDED_DIR_DEFAULT,
    aws_profile: Optional[str] = None,
    tar_name: str = HAND_TAR_NAME_DEFAULT,
    active_cameras: Optional[List[str]] = None,
    keep_tar: bool = False,
    verbose: bool = False,
) -> str:
    """Download + extract one segment's hand tracking tar from S3.

    Thread-/process-safe via a per-(session, segment) file lock, mirroring the
    v0.2.x CacheManager. Returns the **local session dir**, ready to be passed
    to :class:`HandEpisode`. No-ops (fast) when a valid extraction already
    exists.

    Args:
        session_uri: ``s3://bucket/.../{session}`` - the session folder on S3.
            The tar is expected at ``{session_uri}/processed-segment{segment}/{tar_name}``.
        segment: segment number within the session.
        target_dir: local cache root; the S3 layout is mirrored beneath it.
        aws_profile: optional AWS profile name.
        tar_name: name of the tar object.
        active_cameras: cameras whose mp4s must exist for validation
            (defaults to all four).
        keep_tar: keep the downloaded tar next to ``hand/`` after extraction.
        verbose: print progress.
    """
    import boto3  # local import: only needed on the download path
    import filelock

    active_cameras = active_cameras or list(HAND_CAMS)

    parsed = urlparse(session_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"session_uri must be an s3:// URI, got: {session_uri}")
    bucket = parsed.netloc
    session_key = parsed.path.strip("/")

    local_session_dir = str(Path(target_dir).expanduser() / bucket / session_key)
    paths = HandPathManager(local_session_dir, segment)

    locks_dir = Path(LOCKS_DIR_DEFAULT).expanduser()
    os.makedirs(locks_dir, exist_ok=True)
    lock_id = f"{session_key.replace('/', '_')}-segment{segment}-handv2"

    with filelock.FileLock(str(locks_dir / f"{lock_id}.lock")):
        if _validate_hand_dir(paths, active_cameras):
            if verbose:
                print(f"Hand segment already cached: {paths.segment_dir}")
            return local_session_dir

        os.makedirs(paths.segment_dir, exist_ok=True)
        tar_key = posixpath.join(session_key, f"processed-segment{segment}", tar_name)
        local_tar = os.path.join(paths.segment_dir, tar_name)

        if not os.path.exists(local_tar):
            if verbose:
                print(f"Downloading s3://{bucket}/{tar_key} ...")
            session = boto3.Session(profile_name=aws_profile)
            s3_client = session.client("s3")
            s3_client.download_file(bucket, tar_key, local_tar)

        hand_dir = os.path.join(paths.segment_dir, "hand")
        os.makedirs(hand_dir, exist_ok=True)
        if verbose:
            print(f"Extracting {local_tar} -> {hand_dir}")
        with tarfile.open(local_tar) as tf:
            try:
                tf.extractall(hand_dir, filter="data")
            except TypeError:  # `filter` kwarg requires >= 3.10.12 / 3.11.4
                tf.extractall(hand_dir)

        if not keep_tar:
            os.remove(local_tar)

        # re-resolve: `hand/hand_tracking` now exists
        paths = HandPathManager(local_session_dir, segment)
        if not _validate_hand_dir(paths, active_cameras):
            raise ValueError(f"Extracted hand segment {paths.segment_dir} is missing required files.")

    return local_session_dir


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clipped_pose_cache_valid(cache_dir: Path, archive_sha256: str, source_frame_start: int, source_frame_end: int) -> bool:
    marker_path = cache_dir / "extraction.json"
    params_dir = cache_dir.joinpath(*CLIPPED_POSE_PREFIX)
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, ValueError, TypeError):
        return False
    if marker != {
        "archive_sha256": archive_sha256,
        "source_frame_start": source_frame_start,
        "source_frame_end": source_frame_end,
    }:
        return False
    expected_names = {f"frame_{index:06d}.npz" for index in range(source_frame_start, source_frame_end)}
    try:
        actual_names = {path.name for path in params_dir.iterdir() if path.is_file()}
    except OSError:
        return False
    return actual_names == expected_names


def _extract_clipped_pose_frames(
    archive_path: Path,
    *,
    source_frame_start: int,
    source_frame_end: int,
) -> Path:
    """Safely extract source-indexed pose files into an atomic, content-addressed cache."""

    import filelock

    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing clipped hand pose archive: {archive_path}")
    if source_frame_start < 0 or source_frame_end <= source_frame_start:
        raise ValueError(f"Invalid clipped pose range [{source_frame_start}, {source_frame_end})")

    archive_sha256 = _sha256_path(archive_path)
    extraction_root = archive_path.parent / ".grounded"
    extraction_root.mkdir(parents=True, exist_ok=True)
    # Keep components short enough for legacy Windows MAX_PATH environments;
    # the marker still records and validates the full digest.
    cache_dir = extraction_root / f"pose_frames-{archive_sha256[:16]}"
    lock_path = extraction_root / f"pose_frames-{archive_sha256[:16]}.lock"

    with filelock.FileLock(str(lock_path)):
        if cache_dir.exists():
            if _clipped_pose_cache_valid(cache_dir, archive_sha256, source_frame_start, source_frame_end):
                return cache_dir.joinpath(*CLIPPED_POSE_PREFIX)
            raise ValueError(
                f"Clipped pose cache is incomplete or corrupt: {cache_dir}. "
                "Remove only this content-addressed cache directory and reopen the episode."
            )

        temporary_dir = Path(tempfile.mkdtemp(prefix=f".pf-{archive_sha256[:8]}-", dir=extraction_root))
        try:
            params_dir = temporary_dir.joinpath(*CLIPPED_POSE_PREFIX)
            params_dir.mkdir(parents=True, exist_ok=True)
            expected_indices = set(range(source_frame_start, source_frame_end))
            extracted_indices: set[int] = set()

            with tarfile.open(archive_path, mode="r:*") as archive:
                for member in archive.getmembers():
                    member_path = PurePosixPath(member.name)
                    parts = member_path.parts
                    if member_path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
                        raise ValueError(f"Unsafe path in clipped hand pose archive: {member.name!r}")
                    if member.isdir():
                        if tuple(parts) not in {
                            CLIPPED_POSE_PREFIX[:1],
                            CLIPPED_POSE_PREFIX[:2],
                            CLIPPED_POSE_PREFIX,
                        }:
                            raise ValueError(f"Unexpected directory in clipped hand pose archive: {member.name!r}")
                        continue
                    if not member.isfile() or tuple(parts[:-1]) != CLIPPED_POSE_PREFIX:
                        raise ValueError(f"Unexpected member in clipped hand pose archive: {member.name!r}")

                    match = CLIPPED_POSE_FILENAME.fullmatch(parts[-1])
                    if match is None:
                        raise ValueError(f"Unexpected pose filename in clipped hand pose archive: {member.name!r}")
                    source_frame_idx = int(match.group(1))
                    if source_frame_idx not in expected_indices:
                        raise ValueError(
                            f"Pose frame {source_frame_idx} falls outside published range "
                            f"[{source_frame_start}, {source_frame_end})"
                        )
                    if source_frame_idx in extracted_indices:
                        raise ValueError(f"Duplicate pose frame in clipped hand pose archive: {source_frame_idx}")

                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"Could not read clipped hand pose archive member: {member.name!r}")
                    destination = params_dir / parts[-1]
                    with source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    extracted_indices.add(source_frame_idx)

            if extracted_indices != expected_indices:
                missing = sorted(expected_indices - extracted_indices)
                first_missing = missing[0] if missing else "unknown"
                raise ValueError(
                    f"Clipped hand pose archive has {len(extracted_indices)} frame(s), expected "
                    f"{len(expected_indices)}; first missing source frame is {first_missing}"
                )

            marker = {
                "archive_sha256": archive_sha256,
                "source_frame_start": source_frame_start,
                "source_frame_end": source_frame_end,
            }
            (temporary_dir / "extraction.json").write_text(json.dumps(marker, indent=2) + "\n")
            os.replace(temporary_dir, cache_dir)
            return cache_dir.joinpath(*CLIPPED_POSE_PREFIX)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)


def _full_hand_cache_valid(cache_dir: Path, archive_sha256: str, segment: int) -> bool:
    marker_path = cache_dir / "extraction.json"
    session_dir = cache_dir / "session"
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, ValueError, TypeError):
        return False
    if marker != {
        "archive_sha256": archive_sha256,
        "segment": segment,
    }:
        return False
    return _validate_hand_dir(HandPathManager(str(session_dir), segment), [])


def _extract_full_hand_archive(
    archive_path: Path,
    *,
    cache_root: Path,
    segment: int,
    active_cameras: List[str],
) -> Path:
    """Safely extract a full-segment Hand archive into an atomic content cache."""

    import filelock

    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing full hand archive: {archive_path}")
    if segment < 0:
        raise ValueError("segment must be non-negative")

    archive_sha256 = _sha256_path(archive_path)
    extraction_root = cache_root / ".grounded"
    extraction_root.mkdir(parents=True, exist_ok=True)
    cache_dir = extraction_root / f"hand-{archive_sha256[:16]}-segment{segment}"
    lock_path = extraction_root / f"hand-{archive_sha256[:16]}-segment{segment}.lock"

    with filelock.FileLock(str(lock_path)):
        if cache_dir.exists():
            if not _full_hand_cache_valid(cache_dir, archive_sha256, segment):
                raise ValueError(
                    f"Full hand cache is incomplete or corrupt: {cache_dir}. "
                    "Remove only this content-addressed cache directory and reopen the asset."
                )
            session_dir = cache_dir / "session"
            if not _validate_hand_dir(HandPathManager(str(session_dir), segment), active_cameras):
                raise ValueError("Full hand archive is missing one or more requested camera videos")
            return session_dir

        temporary_dir = Path(tempfile.mkdtemp(prefix=f".hand-{archive_sha256[:8]}-", dir=extraction_root))
        try:
            session_dir = temporary_dir / "session"
            segment_dir = session_dir / f"processed-segment{segment}"
            hand_dir = segment_dir / "hand"
            hand_dir.mkdir(parents=True, exist_ok=True)
            extracted_files: set[Path] = set()

            with tarfile.open(archive_path, mode="r:*") as archive:
                normalized_members: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
                for member in archive.getmembers():
                    if "\\" in member.name:
                        raise ValueError(f"Unsafe path in full hand archive: {member.name!r}")
                    member_path = PurePosixPath(member.name)
                    parts = tuple(part for part in member_path.parts if part not in {"", "."})
                    if not parts and member.isdir():
                        continue
                    if member_path.is_absolute() or not parts or any(part == ".." or ":" in part for part in parts):
                        raise ValueError(f"Unsafe path in full hand archive: {member.name!r}")
                    normalized_members.append((member, parts))

                has_hand_prefix = [parts[0] == "hand" for _, parts in normalized_members]
                if any(has_hand_prefix) and not all(has_hand_prefix):
                    raise ValueError("Full hand archive mixes hand-prefixed and rootless paths")
                destination_root = segment_dir if has_hand_prefix and all(has_hand_prefix) else hand_dir
                resolved_destination_root = destination_root.resolve()

                for member, parts in normalized_members:
                    destination = destination_root.joinpath(*parts).resolve()
                    if destination != resolved_destination_root and resolved_destination_root not in destination.parents:
                        raise ValueError(f"Unsafe path in full hand archive: {member.name!r}")
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        raise ValueError(f"Unsupported member in full hand archive: {member.name!r}")
                    if destination in extracted_files:
                        raise ValueError(f"Duplicate file in full hand archive: {member.name!r}")

                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"Could not read full hand archive member: {member.name!r}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    extracted_files.add(destination)

            paths = HandPathManager(str(session_dir), segment)
            if not _validate_hand_dir(paths, active_cameras):
                raise ValueError("Full hand archive is missing required Hand files or requested camera videos")
            marker = {
                "archive_sha256": archive_sha256,
                "segment": segment,
            }
            (temporary_dir / "extraction.json").write_text(json.dumps(marker, indent=2) + "\n")
            os.replace(temporary_dir, cache_dir)
            return cache_dir / "session"
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)


# =============================================================================
# Sequential-friendly video reader
# =============================================================================


class _VideoReader:
    """cv2.VideoCapture wrapper that only seeks when access is non-sequential.

    Iterating an episode front-to-back (the common rendering pattern) therefore
    never seeks; random access falls back to CAP_PROP_POS_FRAMES.
    """

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing video: {path}")
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {path}")
        self.next_idx = 0

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def num_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_rgb(self, frame_idx: int) -> np.ndarray:
        if frame_idx != self.next_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            raise IOError(f"Failed to decode frame {frame_idx} from {self.path}")
        self.next_idx = frame_idx + 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# =============================================================================
# Episode
# =============================================================================


class HandEpisode(Dataset):
    """A pure data-reading representation of one segment's hand tracking outputs.

    Frames index the **entire recording** of the segment (``[0, total_frames)``
    by default); pass ``start_frame``/``end_frame`` to restrict to a
    sub-interval (e.g. one of ``episode.continuous_intervals``).

    Notes:
        - All hand geometry is in the unrectified left-front camera frame.
          Use ``episode.camera_params["K_{cam}"]`` and
          ``episode.camera_params["T_{cam}_from_left_front"]`` to project into
          any of the four (undistorted-pinhole) video streams.
        - Video handles are opened lazily per camera and are not shareable
          across processes; with a multi-worker DataLoader each worker opens
          its own handles on first access.
    """

    def __init__(
        self,
        session_dir: str,
        segment: int = 0,
        active_cameras: Optional[List[str]] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        valid_only: bool = False,
    ):
        self.session_dir = str(Path(session_dir).expanduser())
        self.segment = segment
        self.path_manager = HandPathManager(self.session_dir, segment)
        self.active_cameras = list(active_cameras) if active_cameras is not None else []
        assert set(self.active_cameras).issubset(set(HAND_CAMS)), f"active_cameras must be a subset of {HAND_CAMS}"
        self.valid_only = valid_only

        if not os.path.isdir(self.path_manager.hand_tracking_dir):
            raise FileNotFoundError(
                f"No hand tracking outputs under {self.path_manager.segment_dir}. "
                f"Expected {self.path_manager.hand_tracking_dir}. "
                f"Download/extract the segment first (see download_hand_segment)."
            )

        cam_npz = np.load(self.path_manager.camera_params_npz, allow_pickle=True)
        self.camera_params: Dict[str, np.ndarray] = {key: cam_npz[key] for key in cam_npz.files}

        with open(self.path_manager.yield_json, "r") as f:
            self.yield_stats: dict = json.load(f)
        self.total_frames: int = int(self.yield_stats["total_frames"])

        with open(self.path_manager.continuous_intervals_json, "r") as f:
            intervals_raw = json.load(f)
        # inclusive [start, end] runs where both hands are valid
        self.continuous_intervals: List[dict] = intervals_raw.get("both_intervals", [])

        self.start_frame = start_frame
        self.end_frame = self.total_frames if end_frame is None else end_frame
        if not (0 <= self.start_frame < self.end_frame <= self.total_frames):
            raise ValueError(
                f"Invalid frame range [{self.start_frame}, {self.end_frame}) for a recording with {self.total_frames} frames."
            )

        self._video_readers: Dict[str, _VideoReader] = {}
        self._pose_cleaning_metrics: Optional[dict] = None

    # ---- lazy resources -------------------------------------------------------

    def _reader(self, camera: str) -> _VideoReader:
        if camera not in self._video_readers:
            self._video_readers[camera] = _VideoReader(self.path_manager.video_file(camera))
        return self._video_readers[camera]

    @property
    def fps(self) -> float:
        cam = self.active_cameras[0] if self.active_cameras else HAND_CAMS[0]
        try:
            return self._reader(cam).fps or 30.0
        except (FileNotFoundError, IOError):
            return 30.0

    @property
    def pose_cleaning_metrics(self) -> dict:
        """Per-frame cleaning diagnostics, keyed 'frame_{i:06d}' -> {'left': {...}, 'right': {...}}."""
        if self._pose_cleaning_metrics is None:
            if os.path.exists(self.path_manager.pose_cleaning_json):
                with open(self.path_manager.pose_cleaning_json, "r") as f:
                    self._pose_cleaning_metrics = json.load(f)
            else:
                self._pose_cleaning_metrics = {}
        return self._pose_cleaning_metrics

    def close(self):
        for reader in self._video_readers.values():
            reader.release()
        self._video_readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        """Drop process-local resources so episodes pickle cleanly.

        Open decoder handles cannot cross process boundaries; a pickled copy
        (e.g. in a spawned visualization or DataLoader worker) starts with no
        readers and lazily reopens its own on first frame access, seeking once
        if that access is non-sequential. The cached pose-cleaning metrics are
        dropped too and reload from disk on demand.
        """
        state = self.__dict__.copy()
        state["_video_readers"] = {}
        state["_pose_cleaning_metrics"] = None
        return state

    # ---- frame loading ---------------------------------------------------------

    def _load_hands(self, frame_idx: int) -> Dict[str, Optional[HandPose]]:
        hands: Dict[str, Optional[HandPose]] = {side: None for side in SIDES}
        filepath = self.path_manager.param_file(frame_idx)
        if not os.path.exists(filepath):
            return hands

        with np.load(filepath, allow_pickle=True) as d:
            sides = [str(s) for s in d["sides"]]
            for row, side in enumerate(sides):
                if side not in SIDES:
                    continue
                pose = HandPose(
                    side=side,
                    keypoints3d=np.asarray(d["keypoints3d"][row], dtype=np.float32),
                    vertices=np.asarray(d["vertices"][row], dtype=np.float32),
                    global_orient=np.asarray(d["global_orients"][row], dtype=np.float32),
                    transl=np.asarray(d["transls"][row], dtype=np.float32),
                    hand_pose=np.asarray(d["hand_poses"][row], dtype=np.float32),
                    betas=np.asarray(d["betas"][row], dtype=np.float32),
                    source_view=str(d["source_views"][row]),
                    inlier_mask=np.asarray(d["inlier_masks"][row], dtype=bool),
                    is_detected=bool(d["is_detected"][row]),
                    reason=str(d["reasons"][row]),
                    hand_frame_idx=int(d["hand_frame_idxs"][row]),
                )
                if self.valid_only and not pose.is_detected:
                    continue
                hands[side] = pose
        return hands

    def __len__(self):
        return self.end_frame - self.start_frame

    def __getitem__(self, idx) -> Union[HandFrameData, List[HandFrameData]]:
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for episode of length {len(self)}")

        frame_idx = self.start_frame + idx

        imgs = {cam: None for cam in HAND_CAMS}
        for cam in self.active_cameras:
            imgs[cam] = self._reader(cam).read_rgb(frame_idx)

        hands = self._load_hands(frame_idx)

        return HandFrameData(
            frame_idx=frame_idx,
            left_front_rgb=imgs["left_front"],
            right_front_rgb=imgs["right_front"],
            left_eye_rgb=imgs["left_eye"],
            right_eye_rgb=imgs["right_eye"],
            left_hand=hands["left"],
            right_hand=hands["right"],
        )

    # ---- convenience -----------------------------------------------------------

    @classmethod
    def from_s3(
        cls,
        session_uri: str,
        segment: int = 0,
        target_dir: str = GROUNDED_DIR_DEFAULT,
        aws_profile: Optional[str] = None,
        **episode_kwargs,
    ) -> "HandEpisode":
        """Download (if needed) then open a segment's hand tracking outputs."""
        active_cameras = episode_kwargs.get("active_cameras") or list(HAND_CAMS)
        local_session_dir = download_hand_segment(
            session_uri,
            segment,
            target_dir=target_dir,
            aws_profile=aws_profile,
            active_cameras=active_cameras,
        )
        return cls(local_session_dir, segment=segment, **episode_kwargs)

    @classmethod
    def from_asset_download(
        cls,
        download: "AssetDownload",
        *,
        segment: int,
        active_cameras: Optional[List[str]] = None,
        valid_only: bool = False,
    ) -> "HandEpisode":
        """Open the full Hand segment returned by ``ProcessingClient.download_asset``."""

        files = getattr(download, "files", None)
        if files is None:
            raise TypeError("download must be an AssetDownload returned by ProcessingClient.download_asset")
        hand_files = [
            item
            for item in files
            if str(getattr(item, "lane", "")).lower() == "hand"
            and Path(str(getattr(item, "local_path", ""))).name == HAND_TAR_NAME_DEFAULT
        ]
        if len(hand_files) != 1:
            raise ValueError(
                f"AssetDownload must contain exactly one Hand archive named {HAND_TAR_NAME_DEFAULT}, found {len(hand_files)}"
            )
        archive_path = Path(str(getattr(hand_files[0], "local_path", ""))).expanduser().resolve()
        requested_cameras = list(HAND_CAMS if active_cameras is None else active_cameras)
        if len(set(requested_cameras)) != len(requested_cameras) or not set(requested_cameras).issubset(HAND_CAMS):
            raise ValueError(f"active_cameras must contain unique names from {HAND_CAMS}")

        download_root = Path(str(getattr(download, "root_dir", "") or archive_path.parent)).expanduser().resolve()
        session_dir = _extract_full_hand_archive(
            archive_path,
            cache_root=download_root,
            segment=segment,
            active_cameras=requested_cameras,
        )
        episode = cls(
            str(session_dir),
            segment=segment,
            active_cameras=requested_cameras,
            valid_only=valid_only,
        )
        episode.asset_id = str(getattr(download, "asset_id", "") or "")
        episode.episode_id = ""
        episode.caption = None
        episode.lane_status = "available"
        episode.lane_message = ""
        episode.run_id = str(getattr(hand_files[0], "run_id", "") or "")
        episode.job_id = ""
        episode.download_root = str(download_root)
        return episode

    def project_to_camera(self, points_3d: np.ndarray, camera: str) -> np.ndarray:
        """Project (N, 3) points from the unrectified left-front frame into a camera's pixels.

        Returns (N, 2) float pixel coordinates (u, v) for the requested
        (undistorted pinhole) camera stream.
        """
        assert camera in HAND_CAMS, f"camera must be one of {HAND_CAMS}"
        K = self.camera_params[f"K_{camera}"]
        T = self.camera_params[f"T_{camera}_from_left_front"]
        pts = np.asarray(points_3d, dtype=np.float64)
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)
        pts_cam = (T @ pts_h.T).T[:, :3]
        uvw = (K @ pts_cam.T).T
        z = uvw[:, 2:3].copy()
        z[np.abs(z) < 1e-9] = 1e-9
        return uvw[:, :2] / z


class ClippedHandEpisode(HandEpisode):
    """Open the flat hand lane returned by :meth:`ProcessingClient.download_episode`.

    The clipped videos are indexed from zero, while ``pose_frames.tar`` keeps
    the immutable source-frame filenames. This reader maps episode-local video
    frame ``i`` to source pose frame ``source_frame_start + i`` and otherwise
    implements the same interface as :class:`HandEpisode`, so it can be passed
    directly to the existing hand visualizers.

    Typical usage::

        record = client.get_episode(episode_id)
        download = client.download_episode(episode_id, lane="hand")
        episode = ClippedHandEpisode.from_download(download, caption=record.caption)
        visualize_hand_episode_to_mp4(episode, "hands.mp4", caption=episode.caption)
    """

    def __init__(
        self,
        lane_dir: Union[str, os.PathLike],
        active_cameras: Optional[List[str]] = None,
        valid_only: bool = False,
        caption: Optional[str] = None,
    ):
        self.lane_dir = str(Path(lane_dir).expanduser().resolve())
        lane_path = Path(self.lane_dir)
        if not lane_path.is_dir():
            raise FileNotFoundError(f"Clipped hand lane directory does not exist: {lane_path}")

        self.clip_manifest_path = str(lane_path / "clip_manifest.json")
        try:
            with open(self.clip_manifest_path, "r") as stream:
                self.clip_manifest: dict = json.load(stream)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Missing clipped hand manifest: {self.clip_manifest_path}") from error
        if self.clip_manifest.get("schema_version") != CLIPPED_HAND_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported clipped hand schema {self.clip_manifest.get('schema_version')!r}; "
                f"expected {CLIPPED_HAND_SCHEMA_VERSION!r}"
            )

        interval = self.clip_manifest.get("interval")
        poses = self.clip_manifest.get("poses")
        videos = self.clip_manifest.get("videos")
        if not isinstance(interval, dict) or not isinstance(poses, dict) or not isinstance(videos, dict):
            raise ValueError("Clipped hand manifest must publish interval, poses, and videos objects")
        declared_camera_files, declared_sidecars = _clipped_manifest_declarations(self.clip_manifest)
        self.declared_camera_files = tuple(declared_camera_files)
        self.declared_source_sidecars = tuple(declared_sidecars)
        declared_lane_files = [*declared_camera_files, *declared_sidecars]
        missing_declared_files = sorted(name for name in declared_lane_files if not (lane_path / name).is_file())
        if missing_declared_files:
            raise FileNotFoundError(
                "Clipped hand lane is missing manifest-declared file(s): " + ", ".join(missing_declared_files)
            )

        def manifest_int(section: dict, field: str, *, field_name: Optional[str] = None) -> int:
            value = section.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Clipped hand manifest field must be an integer: {field_name or field}")
            return value

        interval_frame_start = manifest_int(interval, "frame_start")
        interval_frame_end = manifest_int(interval, "frame_end")
        self.source_frame_start = manifest_int(poses, "source_frame_start")
        self.source_frame_end = manifest_int(poses, "source_frame_end")
        local_frame_start = manifest_int(poses, "episode_local_frame_start")
        local_frame_end = manifest_int(poses, "episode_local_frame_end")
        pose_file_count = manifest_int(poses, "pose_file_count")

        if (interval_frame_start, interval_frame_end) != (self.source_frame_start, self.source_frame_end):
            raise ValueError("Clipped hand interval and pose source-frame bounds do not match")
        self.total_frames = self.source_frame_end - self.source_frame_start
        if self.source_frame_start < 0 or self.total_frames <= 0:
            raise ValueError(f"Invalid clipped hand source-frame range [{self.source_frame_start}, {self.source_frame_end})")
        if (local_frame_start, local_frame_end) != (0, self.total_frames):
            raise ValueError(
                "Clipped hand pose mapping must cover episode-local frames "
                f"[0, {self.total_frames}), got [{local_frame_start}, {local_frame_end})"
            )
        if pose_file_count != self.total_frames:
            raise ValueError(f"Clipped hand manifest publishes {pose_file_count} pose files for {self.total_frames} frames")
        if poses.get("pose_file_names_preserve_source_frame_indices") is not True:
            raise ValueError("Clipped hand pose filenames must preserve source-frame indices")

        for video_name, metadata in videos.items():
            if not isinstance(metadata, dict):
                raise ValueError(f"Clipped hand video metadata must be an object: {video_name}")
            output_frame_count = manifest_int(
                metadata,
                "output_frame_count",
                field_name=f"videos.{video_name}.output_frame_count",
            )
            if output_frame_count != self.total_frames:
                raise ValueError(
                    f"Clipped hand video {video_name!r} has {output_frame_count} frames; expected {self.total_frames}"
                )

        pose_archive_path = lane_path / CLIPPED_POSE_TAR_NAME
        params_dir = _extract_clipped_pose_frames(
            pose_archive_path,
            source_frame_start=self.source_frame_start,
            source_frame_end=self.source_frame_end,
        )
        self.pose_archive_path = str(pose_archive_path)
        self.pose_cache_dir = str(params_dir.parent.parent.parent)
        self.path_manager = ClippedHandPathManager(lane_path, params_dir, self.source_frame_start)

        available_cameras = [camera for camera in HAND_CAMS if f"{camera}.mp4" in self.declared_camera_files]
        self.active_cameras = list(available_cameras if active_cameras is None else active_cameras)
        if len(set(self.active_cameras)) != len(self.active_cameras) or not set(self.active_cameras).issubset(HAND_CAMS):
            raise ValueError(f"active_cameras must contain unique names from {HAND_CAMS}")
        missing_cameras = [camera for camera in self.active_cameras if camera not in available_cameras]
        if missing_cameras:
            raise FileNotFoundError(f"Clipped hand lane is missing requested camera video(s): {', '.join(missing_cameras)}")

        if not Path(self.path_manager.camera_params_npz).is_file():
            raise FileNotFoundError(f"Missing clipped hand camera parameters: {self.path_manager.camera_params_npz}")
        with np.load(self.path_manager.camera_params_npz, allow_pickle=True) as camera_params:
            self.camera_params: Dict[str, np.ndarray] = {key: np.asarray(camera_params[key]) for key in camera_params.files}

        self.source_yield_stats = (
            self._load_optional_json(Path(self.path_manager.yield_json))
            if "source_yield.json" in self.declared_source_sidecars
            else {}
        )
        self.source_continuous_intervals = (
            self._load_optional_json(Path(self.path_manager.continuous_intervals_json))
            if "source_continuous_intervals.json" in self.declared_source_sidecars
            else {}
        )
        # These inherited attributes describe the clipped view, not the full source recording.
        self.yield_stats = {"total_frames": self.total_frames}
        self.continuous_intervals = self._localize_source_intervals(self.source_continuous_intervals.get("both_intervals", []))
        self.session_dir = self.lane_dir
        self.segment = 0
        self.start_frame = 0
        self.end_frame = self.total_frames
        self.valid_only = valid_only
        self.episode_id = str(self.clip_manifest.get("episode_id") or "")
        self.asset_id = str(self.clip_manifest.get("asset_id") or "")
        self.start_ns = manifest_int(interval, "start_ns")
        self.end_ns = manifest_int(interval, "end_ns")
        self.caption = caption
        self.lane_status = ""
        self.lane_message = ""
        self.run_id = ""
        self.job_id = ""
        self.download_root = ""
        self._video_readers: Dict[str, _VideoReader] = {}
        self._pose_cleaning_metrics: Optional[dict] = None

    @staticmethod
    def _load_optional_json(path: Path) -> dict:
        if not path.is_file():
            return {}
        with path.open("r") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in clipped hand sidecar: {path}")
        return value

    @property
    def pose_cleaning_metrics(self) -> dict:
        """Load clip metrics only when the exact manifest declared the sidecar."""

        if self._pose_cleaning_metrics is None:
            if "source_pose_cleaning_metrics.json" in self.declared_source_sidecars:
                self._pose_cleaning_metrics = self._load_optional_json(Path(self.path_manager.pose_cleaning_json))
            else:
                self._pose_cleaning_metrics = {}
        return self._pose_cleaning_metrics

    def _localize_source_intervals(self, intervals: object) -> List[dict]:
        """Clip inclusive source intervals to this episode and rebase them to local video frames."""

        if not isinstance(intervals, list):
            raise ValueError("source_continuous_intervals.json interval fields must be lists")
        localized: List[dict] = []
        for interval in intervals:
            if not isinstance(interval, dict):
                raise ValueError("source_continuous_intervals.json interval entries must be objects")
            source_start = interval.get("start")
            source_end = interval.get("end")
            if (
                not isinstance(source_start, int)
                or isinstance(source_start, bool)
                or not isinstance(source_end, int)
                or isinstance(source_end, bool)
            ):
                raise ValueError("source_continuous_intervals.json bounds must be integers")
            clipped_start = max(source_start, self.source_frame_start)
            clipped_end = min(source_end, self.source_frame_end - 1)
            if clipped_end < clipped_start:
                continue
            localized.append(
                {
                    "start": clipped_start - self.source_frame_start,
                    "end": clipped_end - self.source_frame_start,
                    "n_present": clipped_end - clipped_start + 1,
                    "holes": [],
                    "source_start": clipped_start,
                    "source_end": clipped_end,
                }
            )
        return localized

    @classmethod
    def from_download(
        cls,
        download: "EpisodeDownload",
        *,
        active_cameras: Optional[List[str]] = None,
        valid_only: bool = False,
        caption: Optional[str] = None,
    ) -> "ClippedHandEpisode":
        """Open the exact hand files returned by ``ProcessingClient.download_episode``.

        ``available`` and ``partial`` hand lanes are accepted when all files
        needed by the reader are present. Unprocessed or failed lanes produce
        a clear error and never synthesize placeholder data.
        """

        lanes = getattr(download, "lanes", None)
        if lanes is None:
            raise TypeError("download must be an EpisodeDownload returned by ProcessingClient.download_episode")
        hand_lanes = [lane for lane in lanes if str(getattr(lane, "lane", "")).lower() == "hand"]
        if len(hand_lanes) != 1:
            raise ValueError(f"EpisodeDownload must contain exactly one hand lane, found {len(hand_lanes)}")
        hand_lane = hand_lanes[0]
        lane_status = str(getattr(hand_lane, "status", "")).lower()
        lane_message = str(getattr(hand_lane, "message", "") or "")
        if lane_status not in {"available", "partial"}:
            detail = f": {lane_message}" if lane_message else ""
            raise ValueError(f"Episode hand lane is {lane_status or 'unknown'}{detail}")

        paths_by_name: Dict[str, Path] = {}
        for downloaded_file in getattr(hand_lane, "files", ()):
            raw_local_path = str(getattr(downloaded_file, "local_path", "") or "").strip()
            if not raw_local_path:
                raise ValueError("Downloaded hand file has no local path")
            local_path = Path(raw_local_path).expanduser().resolve()
            if local_path.name in paths_by_name:
                raise ValueError(f"Downloaded hand lane has duplicate filename: {local_path.name}")
            paths_by_name[local_path.name] = local_path

        required_names = {"clip_manifest.json", CLIPPED_POSE_TAR_NAME}
        missing_names = sorted(required_names - paths_by_name.keys())
        if missing_names:
            raise FileNotFoundError(f"Downloaded hand lane is missing required file(s): {', '.join(missing_names)}")
        lane_path = paths_by_name["clip_manifest.json"].parent
        non_flat = sorted(name for name, path in paths_by_name.items() if path.parent != lane_path)
        if non_flat:
            raise ValueError(
                "Downloaded clipped hand files must share one flat lane directory; misplaced file(s): " + ", ".join(non_flat)
            )
        missing_local_files = sorted(name for name, path in paths_by_name.items() if not path.is_file())
        if missing_local_files:
            raise FileNotFoundError("EpisodeDownload hand file(s) no longer exist locally: " + ", ".join(missing_local_files))

        with paths_by_name["clip_manifest.json"].open("r") as stream:
            clip_manifest = json.load(stream)
        declared_camera_files, declared_sidecars = _clipped_manifest_declarations(clip_manifest)
        required_names |= set(declared_camera_files) | set(declared_sidecars)
        missing_declared_downloads = sorted(required_names - paths_by_name.keys())
        if missing_declared_downloads:
            raise FileNotFoundError(
                "EpisodeDownload is missing clipped-hand manifest declaration(s): " + ", ".join(missing_declared_downloads)
            )
        expected_episode_id = str(getattr(download, "episode_id", "") or "")
        expected_asset_id = str(getattr(download, "asset_id", "") or "")
        if str(clip_manifest.get("episode_id") or "") != expected_episode_id:
            raise ValueError("Downloaded hand clip manifest episode_id does not match EpisodeDownload")
        if str(clip_manifest.get("asset_id") or "") != expected_asset_id:
            raise ValueError("Downloaded hand clip manifest asset_id does not match EpisodeDownload")
        interval = clip_manifest.get("interval") or {}
        if (interval.get("start_ns"), interval.get("end_ns")) != (
            getattr(download, "start_ns", None),
            getattr(download, "end_ns", None),
        ):
            raise ValueError("Downloaded hand clip manifest bounds do not match EpisodeDownload")

        requested_cameras = list(active_cameras or [])
        if len(set(requested_cameras)) != len(requested_cameras) or not set(requested_cameras).issubset(HAND_CAMS):
            raise ValueError(f"active_cameras must contain unique names from {HAND_CAMS}")
        requested_camera_names = [f"{camera}.mp4" for camera in requested_cameras]
        undeclared_requested = sorted(name for name in requested_camera_names if name not in declared_camera_files)
        if undeclared_requested:
            raise FileNotFoundError(
                "Clipped hand manifest does not declare requested camera video(s): " + ", ".join(undeclared_requested)
            )
        missing_requested = sorted(name for name in requested_camera_names if name not in paths_by_name)
        if missing_requested:
            raise FileNotFoundError(
                f"Downloaded hand lane is missing requested camera video(s): {', '.join(missing_requested)}"
            )

        episode = cls(
            lane_path,
            active_cameras=active_cameras,
            valid_only=valid_only,
            caption=caption,
        )
        consumed_names = required_names | {f"{camera}.mp4" for camera in episode.active_cameras}
        missing_consumed = sorted(consumed_names - paths_by_name.keys())
        if missing_consumed:
            episode.close()
            raise FileNotFoundError(
                "Clipped hand reader would use files not present in this EpisodeDownload: " + ", ".join(missing_consumed)
            )
        episode.lane_status = lane_status
        episode.lane_message = lane_message
        episode.run_id = str(getattr(hand_lane, "run_id", "") or "")
        episode.job_id = str(getattr(hand_lane, "job_id", "") or "")
        episode.download_root = str(getattr(download, "root_dir", "") or "")
        return episode

    def source_frame_index(self, episode_local_frame: int) -> int:
        """Map an episode-local video index to the immutable source pose index."""

        if not isinstance(episode_local_frame, int) or isinstance(episode_local_frame, bool):
            raise TypeError("episode_local_frame must be an integer")
        if not 0 <= episode_local_frame < self.total_frames:
            raise IndexError(f"Episode-local frame {episode_local_frame} is outside [0, {self.total_frames})")
        return self.source_frame_start + episode_local_frame

    def episode_local_frame_index(self, source_frame: int) -> int:
        """Map an immutable source pose index back to the clipped video index."""

        if not isinstance(source_frame, int) or isinstance(source_frame, bool):
            raise TypeError("source_frame must be an integer")
        if not self.source_frame_start <= source_frame < self.source_frame_end:
            raise IndexError(f"Source frame {source_frame} is outside [{self.source_frame_start}, {self.source_frame_end})")
        return source_frame - self.source_frame_start


# =============================================================================
# Episode manifest
# =============================================================================


class HandManifest:
    """An episode-level index over one or more sessions' hand tracking segments.

    Wraps a manifest JSON produced by the manifest-creation post-process
    (candidate hand intervals chopped into discrete captioned episodes by a
    VLM, non-work spans discarded). Entries are plain dicts with ``key``,
    ``session``, ``segment``, ``frame_start``/``frame_end`` (end
    **exclusive**), ``duration_s``, ``source_interval`` and ``activity``;
    captions are loaded into :attr:`captions` from ``captions_path`` (default:
    ``{manifest stem}.captions.jsonl`` next to the manifest).

    Each entry's session folder is resolved under ``sessions_root`` (default:
    the manifest's own directory), falling back to the root itself for a
    manifest that lives inside its single session folder.

    Usage::

        manifest = HandManifest("manifest.json", sessions_root="downloads")
        entry = manifest[3]  # by index, or manifest[entry_key]
        with manifest.open(3, active_cameras=["left_front"]) as episode:
            print(episode.caption)  # attached from the captions JSONL
            frame = episode[0]
    """

    def __init__(
        self,
        manifest_path: Union[str, os.PathLike],
        sessions_root: Optional[str] = None,
        captions_path: Optional[Union[str, os.PathLike]] = None,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with open(self.manifest_path, "r") as f:
            self.meta: dict = json.load(f)
        if not isinstance(self.meta.get("episodes"), list):
            raise ValueError(f"{self.manifest_path} is not an episode manifest (no 'episodes' list)")

        self.episodes: List[dict] = list(self.meta["episodes"])
        self._by_key: Dict[str, int] = {e["key"]: i for i, e in enumerate(self.episodes)}
        self._root = Path(sessions_root).expanduser() if sessions_root else self.manifest_path.parent
        self._session_dirs: Dict[str, str] = {}
        self.captions_path = (
            Path(captions_path).expanduser()
            if captions_path
            else self.manifest_path.with_name(self.manifest_path.stem + ".captions.jsonl")
        )
        self.captions: Dict[str, str] = self._load_captions()

    def session_dir(self, session: str) -> str:
        """Resolves (and caches) the folder holding a session's segments."""
        if session not in self._session_dirs:
            for cand in (self._root / session, self._root):
                if cand.is_dir() and any(cand.glob("processed-segment*")):
                    self._session_dirs[session] = str(cand)
                    break
            else:
                raise FileNotFoundError(
                    f"Could not locate session {session!r} under {self._root}; pass sessions_root= explicitly."
                )
        return self._session_dirs[session]

    def _load_captions(self) -> Dict[str, str]:
        captions: Dict[str, str] = {}
        if not self.captions_path.exists():
            return captions
        with open(self.captions_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    captions.update(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return captions

    # ---- entry access ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    def __getitem__(self, episode: Union[int, str]) -> dict:
        if isinstance(episode, str):
            if episode not in self._by_key:
                raise KeyError(f"No episode {episode!r} in {self.manifest_path}")
            return self.episodes[self._by_key[episode]]
        return self.episodes[episode]

    def caption(self, episode: Union[int, str]) -> Optional[str]:
        return self.captions.get(self[episode]["key"])

    # ---- episode loading ---------------------------------------------------------

    def open(self, episode: Union[int, str], **episode_kwargs) -> HandEpisode:
        """Opens one manifest entry as a :class:`HandEpisode` restricted to the
        entry's ``[frame_start, frame_end)`` range.

        Keyword args are forwarded to ``HandEpisode`` (``active_cameras``,
        ``valid_only``, ...). The entry and its caption ride along as
        ``episode.manifest_entry`` and ``episode.caption``.
        """
        entry = self[episode]
        ep = HandEpisode(
            self.session_dir(entry["session"]),
            segment=int(entry["segment"]),
            start_frame=int(entry["frame_start"]),
            end_frame=int(entry["frame_end"]),
            **episode_kwargs,
        )
        ep.manifest_entry = entry
        ep.caption = self.captions.get(entry["key"])
        return ep
