"""
A dataset abstraction over GSI hand tracking (v2) outputs.

Hand tracking v2 replaces the per-frame ``left``/``right`` dict format of the
v0.2.x SDK with a flat, MANO-parametric per-frame npz, and it replaces the
rectified left-front reference frame with the **unrectified left-front camera
frame**. This module is therefore NOT backwards compatible with
``grounded.data.ego_dataset``.

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

import json
import os
import posixpath
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np

try:  # torch is a declared dependency of the SDK, but hand-only readers can live without it
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object

GROUNDED_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/data/")
LOCKS_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/locks/")

HAND_TAR_NAME_DEFAULT = "hand_v2_outputs.tar"

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
        entry = manifest[3]                # by index, or manifest[entry_key]
        with manifest.open(3, active_cameras=["left_front"]) as episode:
            print(episode.caption)         # attached from the captions JSONL
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
                    f"Could not locate session {session!r} under {self._root}; "
                    f"pass sessions_root= explicitly."
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
