import json
import os
import posixpath
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
from tqdm.auto import tqdm

GROUNDED_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/data/")


@dataclass
class FrameData:
    """Dataclass holding all synchronized data for a single frame."""

    timestamp_ns: int
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    stereo_params: Dict[str, np.ndarray]
    left_hand_kp: np.ndarray
    right_hand_kp: np.ndarray
    left_depth: np.ndarray
    c2w: Optional[np.ndarray]  # [tx, ty, tz, qx, qy, qz, qw]


class PathManager:
    def __init__(self, rectified_dir: str):
        # .../processed-segmentXX/hand/
        hand_dir = posixpath.dirname(rectified_dir)
        # .../processed-segmentXX/
        processed_dir = posixpath.dirname(hand_dir)

        # Define synchronized sub-paths
        self.hand_pose_dir = posixpath.join(hand_dir, "hand_tracking", "poses", "refined", "params")
        self.pcd_dir = posixpath.join(hand_dir, "compressed_pcds", "left-front")
        self.slam_trajectory_txt = posixpath.join(processed_dir, "slam", "mav0", "pycuvslam_trajectory.txt")
        self.left_front_mp4 = posixpath.join(rectified_dir, "left-front.mp4")
        self.right_front_mp4 = posixpath.join(rectified_dir, "right-front.mp4")
        self.stereo_params_npz = posixpath.join(rectified_dir, "stereo_params.npz")
        self.timestamp_txt = posixpath.join(rectified_dir, "timestamp.txt")


class EgoEpisode:
    """A lazy-loaded, sliceable representation of a single episode interval."""

    def __init__(
        self,
        rectified_data_dir: str,
        start_frame: int,
        end_frame: int,
        active_cameras: list[str],
    ):
        self.rectified_data_dir = rectified_data_dir
        self.path_manager = PathManager(rectified_data_dir)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.active_cameras = active_cameras

        # TODO: add support for caption loading in next version
        self.caption = None

        stereo_npz = np.load(self.path_manager.stereo_params_npz, allow_pickle=True)
        self.stereo_params = {key: stereo_npz[key] for key in stereo_npz.files}

        with open(self.path_manager.timestamp_txt, "r") as f:
            self.timestamps = f.read().strip().split("\n")

        # State for VideoCapture objects
        self._caps = {}
        self._last_read_idx = {cam: -1 for cam in self.active_cameras}

        # Load Trajectory Data
        self.c2w_timestamps = np.array([])
        self.c2w_poses = np.array([])

        traj_data = np.loadtxt(self.path_manager.slam_trajectory_txt, comments="#")
        if traj_data.ndim == 1:
            traj_data = traj_data[None, :]

        # Note: based on your data, trajectory is in seconds.
        # Convert trajectory timestamps to nanoseconds.
        self.c2w_timestamps = (traj_data[:, 0] * 1e9).astype(np.int64)
        self.c2w_poses = traj_data[:, 1:]
        self._compute_and_print_max_delta()

    def _compute_and_print_max_delta(self):
        max_delta_ns = 0
        for i in range(self.start_frame, self.end_frame):
            if i < len(self.timestamps):
                frame_ts_ns = int(self.timestamps[i])
                closest_idx = np.abs(self.c2w_timestamps - frame_ts_ns).argmin()
                delta = abs(self.c2w_timestamps[closest_idx] - frame_ts_ns)
                if delta > max_delta_ns:
                    max_delta_ns = delta

        # Print the max delta in ms for easier reading
        print(f"Max timestamp delta (rgb/slam) for frames [{self.start_frame}-{self.end_frame}]: {max_delta_ns / 1e6:.3f} ms")

    def _get_cap(self, cam_name: str):
        if cam_name not in self._caps:
            vid_path = os.path.join(self.rectified_data_dir, f"{cam_name}.mp4")
            if not os.path.exists(vid_path):
                raise FileNotFoundError(f"Missing video: {vid_path}")
            self._caps[cam_name] = cv2.VideoCapture(vid_path)
        return self._caps[cam_name]

    def _load_and_merge_hand_streams(self, global_frame: int):
        with np.load(
            os.path.join(self.path_manager.hand_pose_dir, f"frame_{global_frame:06d}.npz"),
            allow_pickle=True,
        ) as hand_pose_data:
            l_front = hand_pose_data["left"].item()["front"]["keypoints_3d_rectcam"]
            r_front = hand_pose_data["right"].item()["front"]["keypoints_3d_rectcam"]

        return l_front, r_front

    def _load_depth_stream(self, global_frame: int) -> np.ndarray | None:
        pcd_path = os.path.join(self.path_manager.pcd_dir, f"frame_{global_frame:06d}.npz")
        with np.load(pcd_path) as pcd_data:
            return pcd_data["z"]

    def __len__(self):
        return self.end_frame - self.start_frame

    def __getitem__(self, idx) -> FrameData:
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]

        global_frame = self.start_frame + idx
        timestamp_ns = int(self.timestamps[global_frame])

        imgs = {cam: None for cam in ["left-front", "right-front"]}

        for cam in self.active_cameras:
            cap = self._get_cap(cam)
            if self._last_read_idx[cam] != global_frame - 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame)

            ret, frame = cap.read()
            if ret:
                imgs[cam] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._last_read_idx[cam] = global_frame

        l_front, r_front = self._load_and_merge_hand_streams(global_frame)
        left_depth = self._load_depth_stream(global_frame)

        # Match c2w pose
        frame_ts_ns = int(timestamp_ns)
        closest_idx = np.abs(self.c2w_timestamps - frame_ts_ns).argmin()
        c2w = self.c2w_poses[closest_idx]

        return FrameData(
            timestamp_ns=timestamp_ns,
            left_rgb=imgs["left-front"],
            right_rgb=imgs["right-front"],
            stereo_params=self.stereo_params,
            left_hand_kp=l_front,
            right_hand_kp=r_front,
            left_depth=left_depth,
            c2w=c2w,
        )

    def __del__(self):
        for cap in getattr(self, "_caps", {}).values():
            cap.release()


class EgoDataset:
    def __init__(
        self,
        index_path: str,
        active_cameras: list[str] = None,
        aws_profile: str = None,
        target_dir: str = GROUNDED_DIR_DEFAULT,
        min_duration_sec: float = 1.0,
        fps: float = 30.0,
    ):
        self.index_path = Path(index_path).expanduser()
        self.aws_profile = aws_profile
        self.target_dir = Path(target_dir).expanduser()
        self.active_cameras = active_cameras or ["left-front", "right-front"]

        os.makedirs(self.target_dir, exist_ok=True)

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        with open(self.index_path, "r") as f:
            raw_data = json.load(f)

        self.metadata = raw_data.get("metadata", {})
        raw_index = list(raw_data.get("index", {}).values())
        dataset_fps = self.metadata.get("fps", fps)

        self.index = []
        total_frames = 0

        # Filter episodes by minimum duration
        for episode in raw_index:
            frame_count = episode["frame_end"] - episode["frame_start"]
            duration_sec = frame_count / dataset_fps

            if duration_sec >= min_duration_sec:
                self.index.append(episode)
                total_frames += frame_count

        total_duration_sec = total_frames / dataset_fps
        self.unique_uris = [episode["perception_uri"] for episode in self.index]

        print(f"Loaded dataset index with {len(self.index)} episodes (min duration threshold: {min_duration_sec}s).")
        print(f"Total dataset duration: {total_duration_sec:.2f} seconds ({total_frames} frames at {dataset_fps} FPS).")

    def _interpolate_missing_frames(self, path_manager: PathManager, frame_start: int, frame_end: int):
        """Scans for missing hand pose .npz files inside the valid interval and always LERPs them."""
        missing = []
        for i in range(frame_start, frame_end):
            hand_file = os.path.join(path_manager.hand_pose_dir, f"frame_{i:06d}.npz")
            if not os.path.exists(hand_file):
                missing.append(i)

        if not missing:
            return

        # Group missing frames into contiguous sub-gaps
        gaps = []
        current_gap = [missing[0]]
        for i in range(1, len(missing)):
            if missing[i] == missing[i - 1] + 1:
                current_gap.append(missing[i])
            else:
                gaps.append(current_gap)
                current_gap = [missing[i]]
        gaps.append(current_gap)

        for gap in gaps:
            # Dynamically seek the nearest valid frame BEFORE the gap
            start_valid = gap[0] - 1
            while not os.path.exists(os.path.join(path_manager.hand_pose_dir, f"frame_{start_valid:06d}.npz")):
                start_valid -= 1

            # Dynamically seek the nearest valid frame AFTER the gap
            end_valid = gap[-1] + 1
            while not os.path.exists(os.path.join(path_manager.hand_pose_dir, f"frame_{end_valid:06d}.npz")):
                end_valid += 1

            valid_start_path = os.path.join(path_manager.hand_pose_dir, f"frame_{start_valid:06d}.npz")
            valid_end_path = os.path.join(path_manager.hand_pose_dir, f"frame_{end_valid:06d}.npz")

            # Load the bounding valid frames
            with np.load(valid_start_path, allow_pickle=True) as d1, np.load(valid_end_path, allow_pickle=True) as d2:
                l1 = d1["left"].item()["front"]["keypoints_3d_rectcam"]
                r1 = d1["right"].item()["front"]["keypoints_3d_rectcam"]

                l2 = d2["left"].item()["front"]["keypoints_3d_rectcam"]
                r2 = d2["right"].item()["front"]["keypoints_3d_rectcam"]

                for i in gap:
                    # Calculate weight: 0.0 at start_valid, 1.0 at end_valid
                    w = (i - start_valid) / (end_valid - start_valid)

                    # Apply LERP
                    l_interp = l1 + w * (l2 - l1) if (l1 is not None and l2 is not None) else None
                    r_interp = r1 + w * (r2 - r1) if (r1 is not None and r2 is not None) else None

                    # Save the synthesized frame to disk
                    out_path = os.path.join(path_manager.hand_pose_dir, f"frame_{i:06d}.npz")
                    out_left = {"front": {"keypoints_3d_rectcam": l_interp}}
                    out_right = {"front": {"keypoints_3d_rectcam": r_interp}}

                    np.savez(out_path, left=np.array(out_left), right=np.array(out_right))

    def _validate_episode_dir(self, path_manager: PathManager, frame_start: int, frame_end: int) -> bool:
        required = [
            path_manager.timestamp_txt,
            path_manager.stereo_params_npz,
            path_manager.slam_trajectory_txt,
            path_manager.hand_pose_dir,
            path_manager.pcd_dir,
        ]
        if "left-front" in self.active_cameras:
            required.append(path_manager.left_front_mp4)
        if "right-front" in self.active_cameras:
            required.append(path_manager.right_front_mp4)

        # 1. Check if base required files and directories exist
        for p in required:
            if not os.path.exists(p):
                print(f"\n[VALIDATION ERROR] Missing core file or directory: {p}")
                return False

        # 2. Validate that the specific requested boundary frames actually downloaded
        if frame_end > frame_start:
            for i in (frame_start, frame_end - 1):
                hand_file = os.path.join(path_manager.hand_pose_dir, f"frame_{i:06d}.npz")
                pcd_file = os.path.join(path_manager.pcd_dir, f"frame_{i:06d}.npz")

                if not os.path.exists(hand_file):
                    print(f"\n[VALIDATION ERROR] Missing boundary hand pose: {hand_file}")
                    return False
                if not os.path.exists(pcd_file):
                    print(f"\n[VALIDATION ERROR] Missing boundary point cloud: {pcd_file}")
                    return False

        return True

    def download_episode(self, episode_idx: int, s3_concurrency: int = 10) -> str:
        episode_info = self.index[episode_idx]
        frame_start = episode_info["frame_start"]
        frame_end = episode_info["frame_end"]

        episode_uri = self.unique_uris[episode_idx]

        # --- S3 DOWNLOAD LOGIC ---
        if episode_uri.startswith("s3://"):
            parsed = urlparse(episode_uri)
            bucket = parsed.netloc

            session = boto3.Session(profile_name=self.aws_profile)
            s3_client = session.client("s3")

            s3_rectified_key = posixpath.dirname(parsed.path.lstrip("/"))
            local_rectified_data_dir = str(self.target_dir / bucket / s3_rectified_key)

            src_paths = PathManager(f"s3://{bucket}/{s3_rectified_key}")
            local_paths = PathManager(local_rectified_data_dir)

            def _sync_file(src: str, dst: str):
                if not os.path.exists(dst):
                    print(f"\rpulling {src[-80:]} ...", end="", flush=True)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    p = urlparse(src)
                    s3_client.download_file(p.netloc, p.path.lstrip("/"), dst)

        # --- LOCAL SYNC LOGIC ---
        else:
            src_rectified_data_dir = os.path.abspath(os.path.dirname(episode_uri))
            src_paths = PathManager(src_rectified_data_dir)

            # Construct a safe target directory structure to prevent collisions
            rel_path = os.path.join(
                f"{episode_info['device_id']}_session_{episode_info['session_num']}",
                f"processed-segment{episode_info['segment_num']}",
                "hand",
                "rectified_dataset",
            )
            local_rectified_data_dir = os.path.join(self.target_dir, "local_sync", rel_path)
            local_paths = PathManager(local_rectified_data_dir)

            def _sync_file(src: str, dst: str):
                if not os.path.exists(dst) and os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)

        # --- COMMON SYNC & INTERPOLATION EXECUTION ---
        os.makedirs(local_paths.hand_pose_dir, exist_ok=True)
        os.makedirs(local_paths.pcd_dir, exist_ok=True)

        # 1. Sync Core Files
        files_to_sync = [
            (src_paths.timestamp_txt, local_paths.timestamp_txt),
            (src_paths.stereo_params_npz, local_paths.stereo_params_npz),
            (src_paths.slam_trajectory_txt, local_paths.slam_trajectory_txt),
        ]
        if "left-front" in self.active_cameras:
            files_to_sync.append((src_paths.left_front_mp4, local_paths.left_front_mp4))
        if "right-front" in self.active_cameras:
            files_to_sync.append((src_paths.right_front_mp4, local_paths.right_front_mp4))

        for src_path, dst_path in files_to_sync:
            _sync_file(src_path, dst_path)

        # 2. Sync Frame-level .npz Files Concurrently
        def sync_single_frame(frame_idx: int):
            filename = f"frame_{frame_idx:06d}.npz"
            frame_targets = [
                (posixpath.join(src_paths.hand_pose_dir, filename), os.path.join(local_paths.hand_pose_dir, filename)),
                (posixpath.join(src_paths.pcd_dir, filename), os.path.join(local_paths.pcd_dir, filename)),
            ]
            try:
                for src_path, dst_path in frame_targets:
                    _sync_file(src_path, dst_path)
            except Exception:
                # Silently ignore missing files; the interpolator will catch and synthesize them.
                pass

        with ThreadPoolExecutor(max_workers=s3_concurrency) as executor:
            futures = [executor.submit(sync_single_frame, i) for i in range(frame_start, frame_end)]
            for future in as_completed(futures):
                future.result()

        self._interpolate_missing_frames(local_paths, frame_start, frame_end)

        if not self._validate_episode_dir(local_paths, frame_start, frame_end):
            raise ValueError(f"Downloaded episode {local_rectified_data_dir} is missing required files.")

        return local_rectified_data_dir

    def download_dataset(self, max_workers: int = 4):
        print(f"Starting parallel download with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_uri = {executor.submit(self.download_episode, idx): idx for idx in range(len(self.unique_uris))}

            for future in tqdm(as_completed(future_to_uri), total=len(self.unique_uris), desc="Downloading episode Segments"):
                uri = future_to_uri[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\nepisode {uri} generated an exception: {exc}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> EgoEpisode:
        episode_info = self.index[idx]

        # Download files and get the local path
        local_rectified_dir = self.download_episode(idx)

        # Pass the local directory to EgoEpisode
        return EgoEpisode(
            rectified_data_dir=local_rectified_dir,
            start_frame=episode_info["frame_start"],
            end_frame=episode_info["frame_end"],
            active_cameras=self.active_cameras,
        )
