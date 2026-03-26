"""
Dataset interface for GSI labeled data
"""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from tqdm.auto import tqdm

GROUNDED_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/data/")


@dataclass
class FrameData:
    """Dataclass holding all synchronized data for a single frame."""

    timestamp: float
    front_img: np.ndarray
    eye_img: np.ndarray
    left_kps_front: np.ndarray
    right_kps_front: np.ndarray
    left_kps_eye: np.ndarray
    right_kps_eye: np.ndarray
    stereo_params: dict


class EgoEpisode:
    """A lazy-loaded, sliceable representation of a single demonstration."""

    def __init__(self, demo_dir: str):
        self.demo_dir = Path(demo_dir)
        self.front_dir = self.demo_dir / "left-front"
        self.eye_dir = self.demo_dir / "left-eye"

        # Load Stereo Params (Strictly required per constraints)
        stereo_path = self.demo_dir / "stereo_params.npz"
        if not stereo_path.exists():
            raise FileNotFoundError(f"Missing mandatory stereo_params.npz in {self.demo_dir}")

        # Load all keys from stereo rectify into a dict
        stereo_npz = np.load(stereo_path, allow_pickle=True)
        self.stereo_params = {key: stereo_npz[key] for key in stereo_npz.files}

        # Load data.npz
        data_path = self.demo_dir / "data.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing mandatory data.npz in {self.demo_dir}")

        data_npz = np.load(data_path, allow_pickle=True)
        self.data_dict = data_npz["data"].item()
        self.timestamps = self.data_dict.get("timestamps", [])

        # Discover and align image frames
        front_imgs = sorted([f for f in os.listdir(self.front_dir) if f.endswith((".jpg", ".png"))])
        eye_imgs = sorted([f for f in os.listdir(self.eye_dir) if f.endswith((".jpg", ".png"))])
        self.common_frames = sorted(list(set(front_imgs).intersection(set(eye_imgs))))

        if not self.common_frames:
            raise ValueError(f"No matching frames found between cameras in {self.demo_dir}")

    def __len__(self):
        return len(self.common_frames)

    def __getitem__(self, idx) -> FrameData:
        # Handle standard slicing
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]

        frame_name = self.common_frames[idx]

        # Lazy load images
        front_img = cv2.imread(str(self.front_dir / frame_name))
        eye_img = cv2.imread(str(self.eye_dir / frame_name))

        # Safely extract keypoints (fallback to empty arrays if tracking dropped)
        l_front = self.data_dict.get("left_hands_21kp_front", [])
        r_front = self.data_dict.get("right_hands_21kp_front", [])
        l_eye = self.data_dict.get("left_hands_21kp_eye", [])
        r_eye = self.data_dict.get("right_hands_21kp_eye", [])

        # Helper to get kps safely
        def get_kps(arr):
            return arr[idx] if len(arr) > idx else np.zeros((21, 3))

        return FrameData(
            timestamp=self.timestamps[idx] if idx < len(self.timestamps) else 0.0,
            front_img=front_img,
            eye_img=eye_img,
            left_kps_front=get_kps(l_front),
            right_kps_front=get_kps(r_front),
            left_kps_eye=get_kps(l_eye),
            right_kps_eye=get_kps(r_eye),
            stereo_params=self.stereo_params,
        )


class EgoDataset:
    def __init__(self, index_path: str, aws_profile: str = None, target_dir: str = GROUNDED_DIR_DEFAULT):
        self.index_path = Path(index_path).expanduser()
        self.aws_profile = aws_profile
        self.target_dir = Path(target_dir).expanduser()

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        with open(self.index_path, "r") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.index = data.get("index", [])

        print(f"Loaded dataset index with {len(self.index)} demos.")

    def _validate_demo_dir(self, demo_dir: Path) -> bool:
        """Checks if a local directory has all necessary valid components."""
        required = [demo_dir / "data.npz", demo_dir / "stereo_params.npz", demo_dir / "left-front", demo_dir / "left-eye"]
        return all(p.exists() for p in required)

    def download_demo(self, demo_uri: str, s3_concurrency: int = 10) -> str:
        """Ensures a specific demo is available locally and returns its path."""
        if not demo_uri.startswith("s3://"):
            local_path = Path(demo_uri).parent
            if not self._validate_demo_dir(local_path):
                raise ValueError(f"Local demo {local_path} is missing required files.")
            return str(local_path)

        parsed = urlparse(demo_uri)
        bucket = parsed.netloc

        s3_parent_key = str(Path(parsed.path.lstrip("/")).parent)
        s3_target_dir = f"s3://{bucket}/{s3_parent_key}"

        local_cache_dir = self.target_dir / bucket / s3_parent_key

        if local_cache_dir.exists() and self._validate_demo_dir(local_cache_dir):
            return str(local_cache_dir)

        local_cache_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["aws", "s3", "sync", s3_target_dir, str(local_cache_dir)]
        if self.aws_profile:
            cmd.extend(["--profile", self.aws_profile])

        # Speed up the individual AWS S3 sync process by injecting environment variables
        env = os.environ.copy()
        env["AWS_MAX_CONCURRENT_REQUESTS"] = str(s3_concurrency)

        try:
            # We capture output rather than sending to DEVNULL so we can print it if it fails
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download {s3_target_dir}. Error: {e.stderr}") from e

        if not self._validate_demo_dir(local_cache_dir):
            raise ValueError(f"Downloaded demo {local_cache_dir} is missing required files.")

        return str(local_cache_dir)

    def download_dataset(self, max_workers: int = 4):
        """Pre-fetches the entire dataset using multiple threads."""
        print(f"Starting parallel download with {max_workers} workers...")

        # Use ThreadPoolExecutor for I/O bound tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_uri = {executor.submit(self.download_demo, uri): uri for uri in self.index}

            # Use tqdm to show a progress bar as tasks complete
            for future in tqdm(as_completed(future_to_uri), total=len(self.index), desc="Downloading Demos"):
                uri = future_to_uri[future]
                try:
                    future.result()  # This will raise any exceptions caught during download
                except Exception as exc:
                    print(f"\nDemo {uri} generated an exception: {exc}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> EgoEpisode:
        demo_uri = self.index[idx]
        local_dir = self.download_demo(demo_uri)
        return EgoEpisode(local_dir)
