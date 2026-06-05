"""
A simple dataset abstraction over GSI data.
"""

import bisect
import io
import json
import os
import posixpath
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import boto3
import botocore
import cv2
import filelock
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from torch.utils.data import Dataset
from tqdm.auto import tqdm

GROUNDED_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/data/")
LOCKS_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/locks/")

JPG_QUALITY_ON_WRITE = 95


def _parse_wrist_pose_from_hand(hand_dict: Optional[dict]) -> Optional[np.ndarray]:
    """Build a 4x4 wrist pose (rect left-front frame) from a full hand dict.

    The hand dict is what `npz["left"|"right"].item()` returns; it holds the
    fitted root pose as R_world_hand (3x3) + t_world_hand (3,) at the top level.
    Returns None only if those keys are absent (an undetected hand), which the
    interpolation pass treats as a gap. The rotation's shape/validity is a
    dataset invariant (always a proper 3x3), so it is not re-checked here.
    """
    if not isinstance(hand_dict, dict):
        return None
    Rm = hand_dict.get("R_world_hand")
    t = hand_dict.get("t_world_hand")
    if Rm is None or t is None:
        return None
    T = np.eye(4)
    T[:3, :3] = np.asarray(Rm, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _pose_is_missing(T: Optional[np.ndarray]) -> bool:
    return T is None or not np.isfinite(np.asarray(T)).all() or np.all(np.asarray(T)[:3, :3] == 0)


def _mat4x4_to_pos_quat(T: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert a 4x4 pose to 7D [tx, ty, tz, qx, qy, qz, qw] (same layout as c2w).

    Returns None if the pose is missing/degenerate.
    """
    if _pose_is_missing(T):
        return None
    T = np.asarray(T, dtype=np.float64)
    quat = R.from_matrix(T[:3, :3]).as_quat()  # (x, y, z, w), like c2w
    return np.concatenate([T[:3, 3], quat]).astype(np.float64)


def _interp_pose(T_start: np.ndarray, T_end: np.ndarray, w: float) -> np.ndarray:
    """Interpolate two 4x4 poses: LERP translation, SLERP rotation. w in [0, 1]."""
    T = np.eye(4)
    T[:3, 3] = (1.0 - w) * T_start[:3, 3] + w * T_end[:3, 3]
    key_rots = R.from_matrix(np.stack([T_start[:3, :3], T_end[:3, :3]]))
    slerp = Slerp([0.0, 1.0], key_rots)
    T[:3, :3] = slerp([w])[0].as_matrix()
    return T


@dataclass
class FrameData:
    """Dataclass holding all synchronized data for a single frame."""

    timestamp_ns: int
    left_front_rgb: Optional[np.ndarray]
    right_front_rgb: Optional[np.ndarray]
    left_eye_rgb: Optional[np.ndarray]
    right_eye_rgb: Optional[np.ndarray]
    left_hand_kp: Optional[np.ndarray]
    right_hand_kp: Optional[np.ndarray]
    left_front_depth: Optional[np.ndarray]
    left_eye_depth: Optional[np.ndarray]
    c2w: Optional[np.ndarray]  # [tx, ty, tz, qx, qy, qz, qw]
    left_wrist: Optional[np.ndarray]  # 7D [tx,ty,tz,qx,qy,qz,qw] in rect left-front frame, or None
    right_wrist: Optional[np.ndarray]  # 7D [tx,ty,tz,qx,qy,qz,qw] in rect left-front frame, or None


class LocalPathManager:
    """Utility for resolving synchronized sub-paths for an episode."""

    def __init__(self, rectified_dir: str):
        self.rectified_dir = rectified_dir
        hand_dir = posixpath.dirname(rectified_dir)
        processed_dir = posixpath.dirname(hand_dir)

        self.hand_pose_dir = posixpath.join(hand_dir, "hand_tracking", "poses", "refined", "params")
        self.front_pcd_dir = posixpath.join(hand_dir, "compressed_pcds", "left-front")
        self.eye_pcd_dir = posixpath.join(hand_dir, "compressed_pcds", "left-eye")
        self.slam_trajectory_txt = posixpath.join(processed_dir, "slam", "mav0", "pycuvslam_trajectory.txt")
        self.stereo_params_npz = posixpath.join(rectified_dir, "stereo_params.npz")
        self.timestamp_txt = posixpath.join(rectified_dir, "timestamp.txt")


class S3PathManager:
    """Utility for resolving synchronized sub-paths for an episode."""

    def __init__(self, rectified_dir: str):
        self.rectified_dir = rectified_dir
        hand_dir = posixpath.dirname(rectified_dir)
        processed_dir = posixpath.dirname(hand_dir)

        self.hand_pose_dir = posixpath.join(hand_dir, "hand_tracking", "poses", "refined", "params")
        self.front_pcd_dir = posixpath.join(hand_dir, "compressed_pcds", "left-front")
        self.eye_pcd_dir = posixpath.join(hand_dir, "compressed_pcds", "left-eye")
        # NOTE: there was a format change so we need to support both - this is the fallback version
        self.slam_trajectory_txt = [
            posixpath.join(processed_dir, "slam", "mav0", "pycuvslam_trajectory.txt"),
            posixpath.join(processed_dir, "slam", "pycuvslam_trajectory.txt"),
        ]
        self.stereo_params_npz = posixpath.join(rectified_dir, "stereo_params.npz")
        self.timestamp_txt = posixpath.join(rectified_dir, "timestamp.txt")


# =============================================================================
# MP4 segment reader
# =============================================================================


class _PrefetchedS3Reader(io.RawIOBase):
    """
    Seekable file-like wrapper over an S3 object. Reads that fall inside a
    pre-fetched range are served from memory; anything else issues an
    on-demand range GET. PyAV / libavformat use this as the AVIOContext.

    With the moov prefix AND the keyframe-anchored data range both prefetched,
    every read libav issues for normal MP4 parsing + decoding is served from
    cache - so the entire decode is two HTTP GETs.
    """

    def __init__(self, s3, bucket, key, size, prefetched):
        self.s3 = s3
        self.bucket = bucket
        self.key = key
        self.size = size
        self.pos = 0
        # list of (start, end, bytes) sorted by start
        self._chunks = sorted((s, s + len(d), d) for s, d in prefetched)

    def _serve(self, start, end):
        for cs, ce, data in self._chunks:
            if cs <= start and end <= ce:
                return data[start - cs : end - cs]
        return None

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.pos + n
        cached = self._serve(self.pos, end)
        if cached is not None:
            self.pos = end
            return cached
        # Cache miss - serve from S3. Should be rare with proper prefetch.
        resp = self.s3.get_object(
            Bucket=self.bucket,
            Key=self.key,
            Range=f"bytes={self.pos}-{end - 1}",
        )
        data = resp["Body"].read()
        self.pos += len(data)
        return data

    def readall(self):
        return self.read(-1)

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def tell(self):
        return self.pos


class _Mp4SegmentExtractor:
    """
    Pulls a per-segment MP4 + sidecar from S3 and writes the frames for a
    requested global-frame range to disk as frame_XXXXXX.jpg.

    Strategy:
      1. GET the small sidecar JSON.
      2. Map global frame indices -> local (file-order) packet indices.
      3. Find the keyframe at-or-before the start, and the next keyframe
         at-or-after the end (or EOF) - this is the byte range we need.
      4. Two range GETs: [0, init_bytes) for ftyp+moov+mdat header, and
         [data_start, data_end) for the keyframe-anchored data.
      5. Hand the result to PyAV via a seekable in-memory file-like and
         decode just the frames we want.
    """

    def __init__(self, s3_client, bucket: str, segment_prefix: str, camera: str):
        self.s3 = s3_client
        self.bucket = bucket
        self.mp4_key = posixpath.join(segment_prefix, f"{camera}.mp4")
        self.sidecar_key = posixpath.join(segment_prefix, f"{camera}.idx.json")
        self.camera = camera
        self._sidecar = None

    def exists(self) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.mp4_key)
            self.s3.head_object(Bucket=self.bucket, Key=self.sidecar_key)
            return True
        except botocore.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    @property
    def sidecar(self) -> dict:
        if self._sidecar is None:
            body = self.s3.get_object(Bucket=self.bucket, Key=self.sidecar_key)["Body"].read()
            self._sidecar = json.loads(body)
        return self._sidecar

    def extract_range(self, frame_start: int, frame_end: int, out_dir: str):
        """Decode global frames [frame_start, frame_end) and write JPGs into out_dir."""
        import av  # local import: only needed when MP4-backed segments are read

        os.makedirs(out_dir, exist_ok=True)
        side = self.sidecar
        first_gf = side["first_global_frame"]
        contiguous = side.get("contiguous", True)
        offsets = side["offsets"]
        sizes = side["sizes"]
        kfs = side["keyframe_indices"]
        fps = side["fps"]
        file_size = side["file_size"]
        init_bytes = side["init_bytes"]

        # Map global -> local (file/decode-order) packet indices.
        # In a closed-GOP MP4, all packets of GOP K occupy file positions
        # [K*GOP, (K+1)*GOP), independent of internal B-frame reordering, so
        # this packet-index space is the right thing to byte-range against.
        if contiguous:
            local_start = frame_start - first_gf
            local_end = frame_end - first_gf

            def local_to_global(i):
                return first_gf + i
        else:
            gfi = side["global_frame_indices"]
            local_start = gfi.index(frame_start)
            local_end = gfi.index(frame_end - 1) + 1

            def local_to_global(i):
                return gfi[i]

        # Sanity-check the request against the actual MP4 length. The two
        # common ways this trips:
        #   - The index says frame_start=X (a JPG-derived global frame
        #     number), but the sidecar was authored for a legacy MP4
        #     segment where first_global_frame=0. local_start ends up far
        #     past the MP4's actual length.
        #   - The index references frames the segment never had (range
        #     misaligned).
        # Without this check, we'd silently read past the end of `offsets`
        # and produce no output, which surfaces downstream as a less
        # informative FileNotFoundError on the missing JPG cache files.
        n_frames_in_mp4 = len(offsets)
        if local_start < 0 or local_start >= n_frames_in_mp4:
            sidecar_source = side.get("source", "unknown")
            raise IndexError(
                f"Requested global frames [{frame_start}, {frame_end}) "
                f"don't map into this MP4. Sidecar first_global_frame="
                f"{first_gf}, num_frames={n_frames_in_mp4}, source="
                f"{sidecar_source}. Computed local_start={local_start} "
                f"is out of bounds [0, {n_frames_in_mp4}). This usually "
                f"means the index.json's frame numbering doesn't match "
                f"the sidecar's - e.g. an episode whose frame_start is a "
                f"JPG-derived global index but whose segment is a legacy "
                f"MP4 with first_global_frame=0."
            )

        # Clamp to valid range.
        local_start = max(0, local_start)
        local_end = min(len(offsets), local_end)
        if local_end <= local_start:
            return

        # Find byte range covering all packets needed to decode [local_start, local_end):
        #   start = keyframe at-or-before local_start
        #   end   = either the next keyframe after local_end-1, or EOF
        kf_idx = max(0, bisect.bisect_right(kfs, local_start) - 1)
        kf_start = kfs[kf_idx]
        next_kf_idx = bisect.bisect_right(kfs, local_end - 1)
        last_packet = (kfs[next_kf_idx] - 1) if next_kf_idx < len(kfs) else (len(offsets) - 1)

        data_start = offsets[kf_start]
        data_end = offsets[last_packet] + sizes[last_packet]

        # Two range GETs - the bulk of the egress savings happens here.
        init_data = self.s3.get_object(
            Bucket=self.bucket,
            Key=self.mp4_key,
            Range=f"bytes=0-{init_bytes - 1}",
        )["Body"].read()
        data_chunk = self.s3.get_object(
            Bucket=self.bucket,
            Key=self.mp4_key,
            Range=f"bytes={data_start}-{data_end - 1}",
        )["Body"].read()

        reader = _PrefetchedS3Reader(
            self.s3,
            self.bucket,
            self.mp4_key,
            file_size,
            prefetched=[(0, init_data), (data_start, data_chunk)],
        )

        with av.open(reader, mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            # Seek to the keyframe (pts in stream.time_base units). backward=True
            # ensures we land on a keyframe at-or-before the target.
            time_base = float(stream.time_base)
            target_pts = int(round(kf_start / fps / time_base))
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)

            # decode() yields VideoFrames in display order. We filter to the
            # requested range and write each to disk as JPG.
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                local_idx = int(round(float(frame.pts) * time_base * fps))
                if local_idx < local_start:
                    continue
                if local_idx >= local_end:
                    break

                global_idx = local_to_global(local_idx)
                bgr = frame.to_ndarray(format="bgr24")
                out_path = os.path.join(out_dir, f"frame_{global_idx:06d}.jpg")
                cv2.imwrite(out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY_ON_WRITE])


# =============================================================================
# Cache manager
# =============================================================================


class CacheManager:
    """Handles thread-safe downloading, caching, and merging of episode data."""

    def __init__(
        self,
        target_dir: str = GROUNDED_DIR_DEFAULT,
        aws_profile: Optional[str] = None,
        active_cameras: List[str] = None,
        verbose: bool = False,
    ):
        self.target_dir = Path(target_dir).expanduser()
        self.aws_profile = aws_profile
        self.active_cameras = active_cameras or ["left-front", "right-front"]
        self.locks_dir = Path(LOCKS_DIR_DEFAULT).expanduser()
        self.verbose = verbose

        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.locks_dir, exist_ok=True)

    def download_episode(self, episode_info: dict, episode_uri: str, s3_concurrency: int = 64) -> str:
        """
        Thread-safe entry point. Locks the episode ID so multiple PyTorch workers
        don't collide while downloading or interpolating the same episode.
        """
        frame_start = episode_info["frame_start"]
        frame_end = episode_info["frame_end"]
        device_id = episode_info["device_id"]
        session_num = episode_info["session_num"]
        segment_num = episode_info["segment_num"]
        episode_id = f"{device_id}-{session_num}-{segment_num}-{frame_start}-{frame_end}"

        local_rectified_data_dir = self._get_local_path(episode_info, episode_uri)
        local_paths = LocalPathManager(local_rectified_data_dir)

        if episode_uri.startswith("s3://"):
            lock_path = self.locks_dir / f"{episode_id}.lock"

            with filelock.FileLock(str(lock_path)):
                if self._validate_episode_dir(local_paths, frame_start, frame_end):
                    return local_rectified_data_dir

                self._download_and_sync(episode_info, episode_uri, local_paths, s3_concurrency, episode_id)
                self._merge_hand_streams(local_paths, frame_start, frame_end)

                if not self._validate_episode_dir(local_paths, frame_start, frame_end):
                    raise ValueError(
                        f"Downloaded episode {local_rectified_data_dir} is missing required files post-processing."
                    )

        return local_rectified_data_dir

    def _get_local_path(self, episode_info: dict, episode_uri: str) -> str:
        if episode_uri.startswith("s3://"):
            parsed = urlparse(episode_uri)
            s3_rectified_key = posixpath.dirname(parsed.path.lstrip("/"))
            return str(self.target_dir / parsed.netloc / s3_rectified_key)
        else:
            rel_path = os.path.join(
                f"{episode_info['device_id']}_session_{episode_info['session_num']}",
                f"processed-segment{episode_info['segment_num']}",
                "hand",
                "rectified_dataset",
            )
            return os.path.join(self.target_dir, "local_sync", rel_path)

    # ---- the main change: split camera downloads (MP4) from everything else --

    def _download_and_sync(
        self,
        episode_info: dict,
        episode_uri: str,
        local_paths: LocalPathManager,
        s3_concurrency: int,
        episode_id: str,
    ):
        frame_start = episode_info["frame_start"]
        frame_end = episode_info["frame_end"]

        os.makedirs(local_paths.hand_pose_dir, exist_ok=True)
        os.makedirs(local_paths.front_pcd_dir, exist_ok=True)
        os.makedirs(local_paths.eye_pcd_dir, exist_ok=True)
        for cam in self.active_cameras:
            os.makedirs(os.path.join(local_paths.rectified_dir, cam), exist_ok=True)

        parsed = urlparse(episode_uri)
        bucket_name = parsed.netloc
        s3_base_prefix = posixpath.dirname(parsed.path.lstrip("/"))

        config = botocore.config.Config(max_pool_connections=s3_concurrency)
        session = boto3.Session(profile_name=self.aws_profile)
        s3_client = session.client("s3", config=config)

        def _sync_file(src: Union[str, List[str]], dst: str):
            if os.path.exists(dst):
                return
            sources = src if isinstance(src, list) else [src]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for s in sources:
                p = urlparse(s)
                try:
                    s3_client.download_file(p.netloc, p.path.lstrip("/"), dst)
                    return
                except botocore.exceptions.ClientError:
                    continue

        src_paths = S3PathManager(f"s3://{bucket_name}/{s3_base_prefix}")

        # --- Non-camera files: timestamps, params, slam, hand poses, point clouds.
        # All small per-frame npz files still come down individually for now.
        non_camera_tasks = [
            (src_paths.timestamp_txt, local_paths.timestamp_txt),
            (src_paths.stereo_params_npz, local_paths.stereo_params_npz),
            (src_paths.slam_trajectory_txt, local_paths.slam_trajectory_txt),
        ]
        for frame_idx in range(frame_start, frame_end):
            npz_filename = f"frame_{frame_idx:06d}.npz"
            non_camera_tasks.append(
                (
                    posixpath.join(src_paths.hand_pose_dir, npz_filename),
                    os.path.join(local_paths.hand_pose_dir, npz_filename),
                )
            )
            non_camera_tasks.append(
                (
                    posixpath.join(src_paths.front_pcd_dir, npz_filename),
                    os.path.join(local_paths.front_pcd_dir, npz_filename),
                )
            )
            non_camera_tasks.append(
                (
                    posixpath.join(src_paths.eye_pcd_dir, npz_filename),
                    os.path.join(local_paths.eye_pcd_dir, npz_filename),
                )
            )

        with ThreadPoolExecutor(max_workers=s3_concurrency) as executor:
            futures = [executor.submit(_sync_file, src, dst) for src, dst in non_camera_tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Downloading {episode_id} (metadata)",
                leave=False,
                disable=not self.verbose,
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed to sync a file: {e}")

        # Camera frames: every segment must have a converted MP4 + sidecar.
        # If anything's missing or extraction fails, this is a hard error -
        # there's no per-JPG fallback after the conversion was finalized.
        for cam in self.active_cameras:
            cam_dir = os.path.join(local_paths.rectified_dir, cam)
            if self._cam_dir_already_populated(cam_dir, frame_start, frame_end):
                continue

            extractor = _Mp4SegmentExtractor(s3_client, bucket_name, s3_base_prefix, cam)
            if not extractor.exists():
                raise RuntimeError(
                    f"No converted MP4+sidecar found for {cam} on "
                    f"{episode_id}. Expected "
                    f"s3://{bucket_name}/{s3_base_prefix}/{cam}.mp4 and "
                    f"s3://{bucket_name}/{s3_base_prefix}/{cam}.idx.json. "
                    f"If this segment was supposed to be converted, "
                    f"investigate the conversion pipeline; if it's not in "
                    f"the converted set, filter it out of index.json."
                )
            try:
                extractor.extract_range(frame_start, frame_end, cam_dir)
            except Exception as e:
                raise RuntimeError(f"MP4 extraction failed for {cam} on {episode_id}: {e!r}")

        if self.verbose:
            print(f"Finished downloading {episode_id}.")

    @staticmethod
    def _cam_dir_already_populated(cam_dir: str, frame_start: int, frame_end: int) -> bool:
        if not os.path.isdir(cam_dir):
            return False
        for i in range(frame_start, frame_end):
            if not os.path.exists(os.path.join(cam_dir, f"frame_{i:06d}.jpg")):
                return False
        return True

    # ---- everything below is unchanged from the original SDK ------------------

    def _merge_hand_streams(self, path_manager: LocalPathManager, frame_start: int, frame_end: int):
        """
        Phase 1: Project missing front detections from the eye cameras and save to disk.
        Phase 2: Group remaining complete gaps and fill them using Linear Interpolation (LERP).
        """
        stereo_npz = np.load(path_manager.stereo_params_npz, allow_pickle=True)
        T_f2e_unrect = stereo_npz["T_front_to_eye"]

        R_eye_4x4 = np.eye(4)
        R_eye_4x4[:3, :3] = stereo_npz["eye_R1"]
        R_eye_inv = np.linalg.inv(R_eye_4x4)

        T_e2f_unrect = np.linalg.inv(T_f2e_unrect)

        R_front_4x4 = np.eye(4)
        R_front_4x4[:3, :3] = stereo_npz["front_R1"]

        T_recteye_to_rectfront = R_front_4x4 @ T_e2f_unrect @ R_eye_inv

        def is_missing(kp):
            return kp is None or np.size(kp) == 0 or np.all(kp == 0)

        def project_eye_to_front(kp_eye):
            if is_missing(kp_eye) or T_recteye_to_rectfront is None:
                return None
            ones = np.ones((kp_eye.shape[0], 1), dtype=kp_eye.dtype)
            kp_eye_h = np.concatenate([kp_eye, ones], axis=-1)
            return (T_recteye_to_rectfront @ kp_eye_h.T).T[:, :3]

        missing_left = []
        missing_right = []

        for i in range(frame_start, frame_end):
            filepath = os.path.join(path_manager.hand_pose_dir, f"frame_{i:06d}.npz")
            if not os.path.exists(filepath):
                missing_left.append(i)
                missing_right.append(i)
                continue

            try:
                with np.load(filepath, allow_pickle=True) as d:
                    left_data = d["left"].item()
                    right_data = d["right"].item()
            except Exception:
                missing_left.append(i)
                missing_right.append(i)
                continue

            l_front = (left_data.get("front") or {}).get("keypoints_3d_rectcam")
            r_front = (right_data.get("front") or {}).get("keypoints_3d_rectcam")

            needs_save = False

            if is_missing(l_front):
                l_eye = (left_data.get("eye") or {}).get("keypoints_3d_rectcam")
                l_front = project_eye_to_front(l_eye)
                if not is_missing(l_front):
                    needs_save = True

            if is_missing(r_front):
                r_eye = (right_data.get("eye") or {}).get("keypoints_3d_rectcam")
                r_front = project_eye_to_front(r_eye)
                if not is_missing(r_front):
                    needs_save = True

            if needs_save:
                out_left = left_data.copy()
                out_right = right_data.copy()

                if not is_missing(l_front):
                    out_left.setdefault("front", {})["keypoints_3d_rectcam"] = l_front
                if not is_missing(r_front):
                    out_right.setdefault("front", {})["keypoints_3d_rectcam"] = r_front

                np.savez(filepath, left=np.array(out_left), right=np.array(out_right))

            if is_missing(l_front):
                missing_left.append(i)
            if is_missing(r_front):
                missing_right.append(i)

        if not missing_left and not missing_right:
            return

        def _load_front_only(frame_idx: int):
            filepath = os.path.join(path_manager.hand_pose_dir, f"frame_{frame_idx:06d}.npz")
            if not os.path.exists(filepath):
                return None, None
            try:
                with np.load(filepath, allow_pickle=True) as d:
                    l = (d["left"].item().get("front") or {}).get("keypoints_3d_rectcam")
                    r = (d["right"].item().get("front") or {}).get("keypoints_3d_rectcam")
                    return l if not is_missing(l) else None, r if not is_missing(r) else None
            except:
                return None, None

        def _group_gaps(indices):
            if not indices:
                return []
            gaps, current = [], [indices[0]]
            for i in range(1, len(indices)):
                if indices[i] == indices[i - 1] + 1:
                    current.append(indices[i])
                else:
                    gaps.append(current)
                    current = [indices[i]]
            gaps.append(current)
            return gaps

        def _process_hand_gaps(gaps, is_left: bool):
            for gap in gaps:
                start_valid = gap[0] - 1
                start_kp = None
                while start_valid >= 0:
                    l, r = _load_front_only(start_valid)
                    start_kp = l if is_left else r
                    if start_kp is not None:
                        break
                    start_valid -= 1

                end_valid = gap[-1] + 1
                end_kp = None
                while end_valid < frame_end + 10000:
                    l, r = _load_front_only(end_valid)
                    end_kp = l if is_left else r
                    if end_kp is not None:
                        break
                    end_valid += 1

                if start_kp is None or end_kp is None:
                    continue

                for i in gap:
                    w = (i - start_valid) / (end_valid - start_valid)
                    interp_kp = start_kp + w * (end_kp - start_kp)

                    filepath = os.path.join(path_manager.hand_pose_dir, f"frame_{i:06d}.npz")

                    out_left, out_right = {}, {}
                    if os.path.exists(filepath):
                        with np.load(filepath, allow_pickle=True) as d:
                            out_left = d["left"].item()
                            out_right = d["right"].item()

                    if is_left:
                        out_left.setdefault("front", {})["keypoints_3d_rectcam"] = interp_kp
                    else:
                        out_right.setdefault("front", {})["keypoints_3d_rectcam"] = interp_kp

                    np.savez(filepath, left=np.array(out_left), right=np.array(out_right))

        _process_hand_gaps(_group_gaps(missing_left), is_left=True)
        _process_hand_gaps(_group_gaps(missing_right), is_left=False)

        # Independent of the keypoint stream above: build and gap-fill the wrist-pose
        # stream (translation LERP + rotation Slerp), gated by MAX_POSE_GAP_FRAMES.
        self._merge_wrist_poses(path_manager, frame_start, frame_end)

    # Largest gap (in frames) we will Slerp-fill for wrist poses. Longer gaps are
    # left missing rather than risk a large-rotation interpolation error.
    MAX_POSE_GAP_FRAMES = 15

    def _merge_wrist_poses(self, path_manager: "LocalPathManager", frame_start: int, frame_end: int):
        """Fill missing wrist poses independently of the 21-keypoint interpolation.

        For each hand, reads the stored fitted root pose per frame, finds runs of
        missing frames, and fills runs no longer than MAX_POSE_GAP_FRAMES by
        interpolating the bracketing valid poses (LERP translation, SLERP rotation).
        Filled poses are written under the 'wrist_pose_rectcam' key, never touching
        'keypoints_3d_rectcam'.
        """

        def _load_pose(frame_idx: int, hand: str) -> Optional[np.ndarray]:
            fp = os.path.join(path_manager.hand_pose_dir, f"frame_{frame_idx:06d}.npz")
            if not os.path.exists(fp):
                return None
            try:
                with np.load(fp, allow_pickle=True) as d:
                    hand_dict = d[hand].item()
            except Exception:
                return None
            # Prefer an already-materialized wrist pose (cached in the front view by a
            # previous pass), else parse the fitted root pose from the top-level dict.
            cached = (hand_dict.get("front") or {}).get("wrist_pose_rectcam")
            if cached is not None:
                T = np.asarray(cached, dtype=np.float64)
                return None if _pose_is_missing(T) else T
            T = _parse_wrist_pose_from_hand(hand_dict)
            return None if _pose_is_missing(T) else T

        def _write_pose(frame_idx: int, hand: str, T: np.ndarray):
            fp = os.path.join(path_manager.hand_pose_dir, f"frame_{frame_idx:06d}.npz")
            out_left, out_right = {}, {}
            if os.path.exists(fp):
                with np.load(fp, allow_pickle=True) as d:
                    out_left = d["left"].item()
                    out_right = d["right"].item()
            target = out_left if hand == "left" else out_right
            target.setdefault("front", {})["wrist_pose_rectcam"] = T.astype(np.float32)
            np.savez(fp, left=np.array(out_left), right=np.array(out_right))

        def _group_gaps(indices):
            if not indices:
                return []
            gaps, current = [], [indices[0]]
            for k in range(1, len(indices)):
                if indices[k] == indices[k - 1] + 1:
                    current.append(indices[k])
                else:
                    gaps.append(current)
                    current = [indices[k]]
            gaps.append(current)
            return gaps

        for hand in ("left", "right"):
            # First materialize any natively-available pose so reads are stable,
            # and record which frames are missing a pose.
            missing = []
            for i in range(frame_start, frame_end):
                T = _load_pose(i, hand)
                if T is None:
                    missing.append(i)
                else:
                    # persist parsed-from-params pose under the canonical key (idempotent)
                    _write_pose(i, hand, T)

            for gap in _group_gaps(missing):
                if len(gap) > self.MAX_POSE_GAP_FRAMES:
                    continue  # too long; leave missing

                start_valid = gap[0] - 1
                T_start = None
                while start_valid >= frame_start:
                    T_start = _load_pose(start_valid, hand)
                    if T_start is not None:
                        break
                    start_valid -= 1

                end_valid = gap[-1] + 1
                T_end = None
                while end_valid < frame_end:
                    T_end = _load_pose(end_valid, hand)
                    if T_end is not None:
                        break
                    end_valid += 1

                # Need both brackets to interpolate (no extrapolation at episode edges).
                if T_start is None or T_end is None:
                    continue

                span = end_valid - start_valid
                for i in gap:
                    w = (i - start_valid) / span
                    _write_pose(i, hand, _interp_pose(T_start, T_end, w))

    def _validate_episode_dir(self, path_manager: LocalPathManager, frame_start: int, frame_end: int) -> bool:
        required = [
            path_manager.timestamp_txt,
            path_manager.stereo_params_npz,
            path_manager.slam_trajectory_txt,
            path_manager.hand_pose_dir,
            path_manager.front_pcd_dir,
            path_manager.eye_pcd_dir,
        ]
        for cam in self.active_cameras:
            required.append(os.path.join(path_manager.rectified_dir, cam))

        for p in required:
            if not os.path.exists(p):
                return False

        if frame_end > frame_start:
            for i in (frame_start, frame_end - 1):
                if not os.path.exists(os.path.join(path_manager.hand_pose_dir, f"frame_{i:06d}.npz")):
                    return False
                if not os.path.exists(os.path.join(path_manager.front_pcd_dir, f"frame_{i:06d}.npz")):
                    return False
                if not os.path.exists(os.path.join(path_manager.eye_pcd_dir, f"frame_{i:06d}.npz")):
                    return False
                for cam in self.active_cameras:
                    if not os.path.exists(os.path.join(path_manager.rectified_dir, cam, f"frame_{i:06d}.jpg")):
                        return False

        return True

    # ---- local-cache cleanup --------------------------------------------------

    def delete_episode(self, episode_info: dict, episode_uri: str, purge_segment: bool = False) -> int:
        """Delete an episode's cached files to free disk. The cache is
        per-segment, so by default only this episode's frame_*.{jpg,npz} in
        [frame_start, frame_end) are removed; purge_segment=True drops the
        whole segment dir (also wiping sibling episodes). Returns file count
        removed. Locks the episode so it won't race its own download."""
        fs, fe = episode_info["frame_start"], episode_info["frame_end"]
        rect_dir = self._get_local_path(episode_info, episode_uri)
        if not os.path.exists(rect_dir):
            return 0
        lp = LocalPathManager(rect_dir)

        def _delete() -> int:
            if purge_segment:
                dirs = [lp.rectified_dir, lp.hand_pose_dir, lp.front_pcd_dir, lp.eye_pcd_dir]
                n = sum(len(files) for d in dirs if os.path.isdir(d) for _, _, files in os.walk(d))
                for d in dirs:
                    shutil.rmtree(d, ignore_errors=True)
                return n
            cam_dirs = (
                [
                    os.path.join(lp.rectified_dir, e)
                    for e in os.listdir(lp.rectified_dir)
                    if os.path.isdir(os.path.join(lp.rectified_dir, e))
                ]
                if os.path.isdir(lp.rectified_dir)
                else []
            )
            removed = 0
            for i in range(fs, fe):
                npz, jpg = f"frame_{i:06d}.npz", f"frame_{i:06d}.jpg"
                paths = [
                    os.path.join(lp.hand_pose_dir, npz),
                    os.path.join(lp.front_pcd_dir, npz),
                    os.path.join(lp.eye_pcd_dir, npz),
                    *(os.path.join(c, jpg) for c in cam_dirs),
                ]
                for fp in paths:
                    if os.path.exists(fp):
                        os.remove(fp)
                        removed += 1
            return removed

        if episode_uri.startswith("s3://"):
            eid = f"{episode_info['device_id']}-{episode_info['session_num']}-{episode_info['segment_num']}-{fs}-{fe}"
            with filelock.FileLock(str(self.locks_dir / f"{eid}.lock")):
                return _delete()
        return _delete()


class EgoEpisode(Dataset):
    """A pure data-reading representation of a single episode interval."""

    LEFT_FRONT_WH = (1920, 1080)
    RIGHT_FRONT_WH = (1920, 1080)
    LEFT_EYE_WH = (1920, 1080)
    RIGHT_EYE_WH = (1920, 1080)

    def __init__(
        self,
        rectified_data_dir: str,
        start_frame: int,
        end_frame: int,
        active_cameras: List[str],
        caption: Optional[str] = None,
    ):
        self.rectified_data_dir = rectified_data_dir
        self.path_manager = LocalPathManager(rectified_data_dir)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.active_cameras = active_cameras
        self.caption = caption

        stereo_npz = np.load(self.path_manager.stereo_params_npz, allow_pickle=True)
        self.stereo_params = {key: stereo_npz[key] for key in stereo_npz.files}

        with open(self.path_manager.timestamp_txt, "r") as f:
            self.timestamps = f.read().strip().split("\n")

        traj_data = np.loadtxt(self.path_manager.slam_trajectory_txt, comments="#")
        if traj_data.ndim == 1:
            traj_data = traj_data[None, :]
        self.c2w_timestamps = (traj_data[:, 0] * 1e9).astype(np.int64)
        unrect_c2w_poses = traj_data[:, 1:]

        R1 = np.asarray(self.stereo_params["front_R1"])
        R_unrect_from_rect = R.from_matrix(R1.T)
        rot_unrect = R.from_quat(unrect_c2w_poses[:, 3:])
        rot_rect = rot_unrect * R_unrect_from_rect
        t_rect = unrect_c2w_poses[:, :3]
        q_rect = rot_rect.as_quat()
        self.c2w_poses = np.concatenate([t_rect, q_rect], axis=1)

    def _load_hand_streams(self, global_frame: int):
        filepath = os.path.join(self.path_manager.hand_pose_dir, f"frame_{global_frame:06d}.npz")
        if not os.path.exists(filepath):
            return None, None, None, None

        with np.load(filepath, allow_pickle=True) as hand_pose_data:
            left_data = hand_pose_data["left"].item()
            right_data = hand_pose_data["right"].item()
            l_view = left_data.get("front", {}) or {}
            r_view = right_data.get("front", {}) or {}
            l_front = l_view.get("keypoints_3d_rectcam")
            r_front = r_view.get("keypoints_3d_rectcam")
            # Wrist poses: prefer the materialized/interpolated key (cached in the
            # front view by _merge_wrist_poses); else parse the fitted root pose
            # (R_world_hand / t_world_hand) from the top-level hand dict.
            l_cached = l_view.get("wrist_pose_rectcam")
            r_cached = r_view.get("wrist_pose_rectcam")
            l_wrist = (
                np.asarray(l_cached, dtype=np.float64) if l_cached is not None else _parse_wrist_pose_from_hand(left_data)
            )
            r_wrist = (
                np.asarray(r_cached, dtype=np.float64) if r_cached is not None else _parse_wrist_pose_from_hand(right_data)
            )

        def is_missing(kp):
            return kp is None or np.size(kp) == 0 or np.all(kp == 0)

        # Wrist poses are stored/interpolated as 4x4 internally; expose them as 7D
        # [tx,ty,tz,qx,qy,qz,qw] to match the c2w layout (returns None if missing).
        return (
            None if is_missing(l_front) else l_front,
            None if is_missing(r_front) else r_front,
            _mat4x4_to_pos_quat(l_wrist),
            _mat4x4_to_pos_quat(r_wrist),
        )

    def _load_depth_stream(self, global_frame: int, cam_name: str) -> Optional[np.ndarray]:
        pcd_dir = self.path_manager.front_pcd_dir if cam_name == "left-front" else self.path_manager.eye_pcd_dir
        pcd_path = os.path.join(pcd_dir, f"frame_{global_frame:06d}.npz")
        if os.path.exists(pcd_path):
            with np.load(pcd_path) as pcd_data:
                return pcd_data["z"]
        return None

    def __len__(self):
        return self.end_frame - self.start_frame

    def __getitem__(self, idx) -> Union[FrameData, List[FrameData]]:
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]

        global_frame = self.start_frame + idx
        timestamp_ns = int(self.timestamps[global_frame])

        imgs = {cam: None for cam in self.active_cameras}
        for cam in self.active_cameras:
            img_path = os.path.join(self.rectified_data_dir, cam, f"frame_{global_frame:06d}.jpg")
            frame_bgr = cv2.imread(img_path)
            if frame_bgr is None:
                raise FileNotFoundError(f"Missing image frame: {img_path}")
            imgs[cam] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        l_front, r_front, l_wrist, r_wrist = self._load_hand_streams(global_frame)
        left_front_depth = self._load_depth_stream(global_frame, "left-front")
        left_eye_depth = self._load_depth_stream(global_frame, "left-eye")

        closest_idx = np.abs(self.c2w_timestamps - timestamp_ns).argmin()
        c2w = self.c2w_poses[closest_idx]

        return FrameData(
            timestamp_ns=timestamp_ns,
            left_front_rgb=imgs.get("left-front"),
            right_front_rgb=imgs.get("right-front"),
            left_eye_rgb=imgs.get("left-eye"),
            right_eye_rgb=imgs.get("right-eye"),
            left_hand_kp=l_front,
            right_hand_kp=r_front,
            left_front_depth=left_front_depth,
            left_eye_depth=left_eye_depth,
            c2w=c2w,
            left_wrist=l_wrist,
            right_wrist=r_wrist,
        )


class EgoDataset(Dataset):
    """
    Main Dataset entry point.
    Maps indices to cached episodes using the CacheManager.
    """

    CAMS = ["left-front", "right-front", "left-eye", "right-eye"]

    def __init__(
        self,
        index_path: str,
        captions_path: Optional[str] = None,
        active_cameras: Optional[List[str]] = None,
        aws_profile: Optional[str] = None,
        target_dir: str = GROUNDED_DIR_DEFAULT,
        min_duration_sec: float = 1.0,
        fps: float = 30.0,
    ):
        self.index_path = Path(index_path).expanduser()
        self.active_cameras = active_cameras
        assert set(active_cameras).issubset(set(self.CAMS)) and len(active_cameras) > 0, (
            f"active_cameras must be one of {self.CAMS}"
        )

        self.cache_manager = CacheManager(
            target_dir=target_dir,
            aws_profile=aws_profile,
            active_cameras=self.active_cameras,
        )

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        with open(self.index_path, "r") as f:
            raw_data = json.load(f)

        self.metadata = raw_data.get("metadata", {})
        raw_index = list(raw_data.get("index", {}).values())
        dataset_fps = self.metadata.get("fps", fps)

        self.captions_map = {}
        if captions_path and Path(captions_path).expanduser().exists():
            with open(Path(captions_path).expanduser(), "r") as f:
                for line in f:
                    if line.strip():
                        self.captions_map.update(json.loads(line))

        self.index = []
        for episode in raw_index:
            duration_sec = (episode["frame_end"] - episode["frame_start"]) / dataset_fps
            if duration_sec >= min_duration_sec:
                self.index.append(episode)

        self.unique_uris = [episode["perception_uri"] for episode in self.index]
        print(f"Loaded dataset index with {len(self.index)} episodes.")

    def download(self, max_workers: int = 4):
        """Optional helper tool for downloading the dataset upfront."""
        print(f"Starting parallel cache population with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.cache_manager.download_episode, self.index[idx], self.unique_uris[idx]): idx
                for idx in range(len(self.index))
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Populating Cache"):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Download exception: {exc}")

    def get_caption(self, idx: int) -> Optional[str]:
        if not self.captions_map:
            return

        ep = self.index[idx]
        key = (
            f"{ep['device_id']}_session_{ep['session_num']}"
            f"_segment_{ep['segment_num']}_interval_"
            f"{ep['frame_start']}_{ep['frame_end']}"
        )
        return self.captions_map.get(key)

    def __len__(self):
        return len(self.index)

    def delete_episode(self, idx: int, purge_segment: bool = False) -> int:
        """Delete episode `idx`'s cached files (see CacheManager.delete_episode).
        purge_segment=True frees more but wipes sibling episodes from the same
        segment; safe only in one-pass jobs. Returns file count removed."""
        return self.cache_manager.delete_episode(self.index[idx], self.unique_uris[idx], purge_segment=purge_segment)

    def __getitem__(self, idx) -> Union[EgoEpisode, List[EgoEpisode]]:
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self.__getitem__(int(i)) for i in idx]

        episode_info = self.index[idx]
        episode_uri = self.unique_uris[idx]
        local_rectified_dir = self.cache_manager.download_episode(episode_info, episode_uri)

        return EgoEpisode(
            rectified_data_dir=local_rectified_dir,
            start_frame=episode_info["frame_start"],
            end_frame=episode_info["frame_end"],
            active_cameras=self.active_cameras,
            caption=self.get_caption(idx),
        )
