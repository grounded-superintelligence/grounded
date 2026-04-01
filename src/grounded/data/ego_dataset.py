import json
import os
import posixpath
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from tqdm.auto import tqdm

GROUNDED_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/data/")


@dataclass
class FrameData:
    """Dataclass holding all synchronized data for a single frame."""

    timestamp_ns: str
    left_front_img: np.ndarray | None
    right_front_img: np.ndarray | None
    stereo_params: dict
    left_hand_3d: np.ndarray
    right_hand_3d: np.ndarray
    c2w_pose: Optional[np.ndarray]  # [tx, ty, tz, qx, qy, qz, qw]
    caption: str = None  # TODO: add patch for this


class PathManager:
    def __init__(self, rectified_dir: str):
        # .../processed-segmentXX/hand/
        hand_dir = posixpath.dirname(rectified_dir)
        # .../processed-segmentXX/
        processed_dir = posixpath.dirname(hand_dir)

        # Define synchronized sub-paths
        self.hand_pose_dir = posixpath.join(hand_dir, "hand_tracking", "poses", "refined", "params")
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
            traj_data = traj_data[np.newaxis, :]

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

    def __len__(self):
        return self.end_frame - self.start_frame

    def __getitem__(self, idx) -> FrameData:
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]

        global_frame = self.start_frame + idx
        timestamp_ns = self.timestamps[global_frame] if global_frame < len(self.timestamps) else "0"

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

        # Match c2w pose
        frame_ts_ns = int(timestamp_ns)
        closest_idx = np.abs(self.c2w_timestamps - frame_ts_ns).argmin()
        c2w_pose = self.c2w_poses[closest_idx]

        return FrameData(
            timestamp_ns=timestamp_ns,
            left_front_img=imgs["left-front"],
            right_front_img=imgs["right-front"],
            stereo_params=self.stereo_params,
            left_hand_3d=l_front,
            right_hand_3d=r_front,
            c2w_pose=c2w_pose,
            caption=None,  # TODO: add patch for this
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

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        with open(self.index_path, "r") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        raw_index = list(data.get("index", {}).values())

        # Attempt to get fps from metadata, fallback to the provided init parameter
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

    def _validate_episode_dir(self, path_manager: PathManager) -> bool:
        required = [
            path_manager.timestamp_txt,
            path_manager.stereo_params_npz,
            path_manager.slam_trajectory_txt,
            path_manager.hand_pose_dir,
        ]
        if "left-front" in self.active_cameras:
            required.append(path_manager.left_front_mp4)
        if "right-front" in self.active_cameras:
            required.append(path_manager.right_front_mp4)
        return all(os.path.exists(p) for p in required)

    def download_episode(self, episode_idx: int, s3_concurrency: int = 10) -> str:
        episode_uri = self.unique_uris[episode_idx]  # s3://.../hand/rectified_dataset/timestamp.txt

        if episode_uri.startswith("s3://"):
            parsed = urlparse(episode_uri)
            bucket = parsed.netloc

            # 1. Identify the 'rectified_dataset' directory as the root for PathManager
            s3_rectified_key = posixpath.dirname(parsed.path.lstrip("/"))
            s3_rectified_data_dir = f"s3://{bucket}/{s3_rectified_key}"

            # 2. Local root mirror
            local_rectified_data_dir = str(self.target_dir / bucket / s3_rectified_key)

            # 3. Apply PathManager to both (Recycling logic)
            s3_paths = PathManager(s3_rectified_data_dir)
            local_paths = PathManager(local_rectified_data_dir)

            # Ensure local directories exist
            os.makedirs(local_rectified_data_dir, exist_ok=True)
            os.makedirs(local_paths.hand_pose_dir, exist_ok=True)

            env = os.environ.copy()
            env["AWS_MAX_CONCURRENT_REQUESTS"] = str(s3_concurrency)

            # 1. Download specific files inside rectified_data_dir (cp)
            files_to_download = [
                (s3_paths.timestamp_txt, local_paths.timestamp_txt),
                (s3_paths.stereo_params_npz, local_paths.stereo_params_npz),
                (s3_paths.slam_trajectory_txt, local_paths.slam_trajectory_txt),
            ]
            if "left-front" in self.active_cameras:
                files_to_download.append((s3_paths.left_front_mp4, local_paths.left_front_mp4))
            if "right-front" in self.active_cameras:
                files_to_download.append((s3_paths.right_front_mp4, local_paths.right_front_mp4))

            for s3_src, local_dst in files_to_download:
                if not os.path.exists(local_dst):
                    cmd = ["aws", "s3", "cp", s3_src, local_dst]
                    if self.aws_profile:
                        cmd.extend(["--profile", self.aws_profile])
                    subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

            # 2. Download the hand pose directory (sync)
            # Using sync because this directory contains multiple frame_XXXXXX.npz files
            if not os.listdir(local_paths.hand_pose_dir):
                cmd = ["aws", "s3", "sync", s3_paths.hand_pose_dir, local_paths.hand_pose_dir]
                if self.aws_profile:
                    cmd.extend(["--profile", self.aws_profile])
                subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

        else:
            local_rectified_data_dir = os.path.abspath(os.path.dirname(episode_uri))
            local_paths = PathManager(local_rectified_data_dir)

        if not self._validate_episode_dir(local_paths):
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
