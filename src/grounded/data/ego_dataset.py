import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
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
    def __init__(self, rectified_data_dir: str):
        self.rectified_data_dir = rectified_data_dir
        self.left_front_mp4 = os.path.join(rectified_data_dir, "left-front.mp4")
        self.right_front_mp4 = os.path.join(rectified_data_dir, "right-front.mp4")
        self.stereo_params_npz = os.path.join(rectified_data_dir, "stereo_params.npz")
        self.timestamps_txt = os.path.join(rectified_data_dir, "timestamp.txt")
        self.processed_dir = os.path.join(rectified_data_dir, "..")
        self.hand_pose_dir = os.path.join(self.processed_dir, "hand_tracking", "poses", "refined", "params")


class EgoEpisode:
    """A lazy-loaded, sliceable representation of a single demonstration interval."""

    def __init__(
        self,
        rectified_data_dir: str,
        stereo_params: Dict[str, np.ndarray],
        start_frame: int,
        end_frame: int,
        active_cameras: list[str],
        trajectory_uri: str = None,
    ):
        self.rectified_data_dir = rectified_data_dir
        self.path_manager = PathManager(rectified_data_dir)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.active_cameras = active_cameras
        self.stereo_params = stereo_params

        with open(self.path_manager.timestamps_txt, "r") as f:
            self.timestamps = f.read().strip().split("\n")

        # State for VideoCapture objects
        self._caps = {}
        self._last_read_idx = {cam: -1 for cam in self.active_cameras}

        # Load Trajectory Data
        self.c2w_timestamps = np.array([])
        self.c2w_poses = np.array([])

        if trajectory_uri and os.path.exists(trajectory_uri):
            traj_data = np.loadtxt(trajectory_uri, comments="#")
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
        self.index = list(data.get("index", {}).values())
        print(f"Loaded dataset index with {len(self.index)} demonstrations.")

    def _validate_demo_dir(self, rectified_data_dir: Path) -> bool:
        required = [rectified_data_dir / "timestamp.txt", rectified_data_dir / "stereo_params.npz"]
        for cam in self.active_cameras:
            required.append(rectified_data_dir / f"{cam}.mp4")
        return all(p.exists() for p in required)

    def download_demo(self, demo_uri: str, s3_concurrency: int = 10) -> str:
        if not demo_uri.startswith("s3://"):
            local_path = Path(demo_uri).parent
            if not self._validate_demo_dir(local_path):
                raise ValueError(f"Local demo {local_path} is missing required files.")
            return str(local_path)

        parsed = urlparse(demo_uri)
        bucket = parsed.netloc
        s3_dir_key = str(Path(parsed.path.lstrip("/")).parent).replace("\\", "/")
        s3_target_dir = f"s3://{bucket}/{s3_dir_key}"
        local_cache_dir = self.target_dir / bucket / s3_dir_key

        if local_cache_dir.exists() and self._validate_demo_dir(local_cache_dir):
            return str(local_cache_dir)

        local_cache_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["aws", "s3", "sync", s3_target_dir, str(local_cache_dir)]
        if self.aws_profile:
            cmd.extend(["--profile", self.aws_profile])

        env = os.environ.copy()
        env["AWS_MAX_CONCURRENT_REQUESTS"] = str(s3_concurrency)

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download {s3_target_dir}. Error: {e.stderr}") from e

        if not self._validate_demo_dir(local_cache_dir):
            raise ValueError(f"Downloaded demo {local_cache_dir} is missing required files.")

        return str(local_cache_dir)

    def download_dataset(self, max_workers: int = 4):
        print(f"Starting parallel download with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            unique_uris = {demo["perception_uri"] for demo in self.index}
            future_to_uri = {executor.submit(self.download_demo, uri): uri for uri in unique_uris}

            for future in tqdm(as_completed(future_to_uri), total=len(unique_uris), desc="Downloading Demo Segments"):
                uri = future_to_uri[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\nDemo {uri} generated an exception: {exc}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> EgoEpisode:
        demo_info = self.index[idx]
        perception_uri = demo_info["perception_uri"]
        trajectory_uri = demo_info.get("trajectory_uri")
        local_dir = self.download_demo(perception_uri)

        stereo_npz = np.load(
            os.path.join(os.path.dirname(perception_uri), "stereo_params.npz"),
            allow_pickle=True,
        )
        stereo_params = {key: stereo_npz[key] for key in stereo_npz.files}

        return EgoEpisode(
            rectified_data_dir=local_dir,
            stereo_params=stereo_params,
            start_frame=demo_info["frame_start"],
            end_frame=demo_info["frame_end"],
            active_cameras=self.active_cameras,
            trajectory_uri=trajectory_uri,
        )
