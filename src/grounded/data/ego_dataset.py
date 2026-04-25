import json
import os
import posixpath
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import boto3
import botocore
import cv2
import filelock
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

GROUNDED_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/data/")
LOCKS_DIR_DEFAULT = os.path.expanduser("~/.cache/grounded/locks/")


@dataclass
class FrameData:
    """Dataclass holding all synchronized data for a single frame."""

    timestamp_ns: int
    left_front_rgb: Optional[np.ndarray]
    right_front_rgb: Optional[np.ndarray]
    left_eye_rgb: Optional[np.ndarray]
    right_eye_rgb: Optional[np.ndarray]
    stereo_params: Dict[str, np.ndarray]
    left_hand_kp: Optional[np.ndarray]
    right_hand_kp: Optional[np.ndarray]
    left_front_depth: Optional[np.ndarray]
    left_eye_depth: Optional[np.ndarray]
    c2w: Optional[np.ndarray]  # [tx, ty, tz, qx, qy, qz, qw]


class PathManager:
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


class CacheManager:
    """Handles thread-safe downloading, caching, and merging of episode data."""

    def __init__(
        self, target_dir: str = GROUNDED_DIR_DEFAULT, aws_profile: Optional[str] = None, active_cameras: List[str] = None
    ):
        self.target_dir = Path(target_dir).expanduser()
        self.aws_profile = aws_profile
        self.active_cameras = active_cameras or ["left-front", "right-front"]
        self.locks_dir = Path(LOCKS_DIR_DEFAULT).expanduser()

        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.locks_dir, exist_ok=True)

    def download_episode(self, episode_info: dict, episode_uri: str, s3_concurrency: int = 100) -> str:
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
        local_paths = PathManager(local_rectified_data_dir)

        # a unique lock file for this specific episode
        lock_path = self.locks_dir / f"{episode_id}.lock"

        with filelock.FileLock(str(lock_path)):
            # if valid cache exists, return immediately (like hf datasets)
            if self._validate_episode_dir(local_paths, frame_start, frame_end):
                return local_rectified_data_dir

            self._download_and_sync(episode_info, episode_uri, local_paths, s3_concurrency, episode_id)
            self._merge_hand_streams(local_paths, frame_start, frame_end)

            if not self._validate_episode_dir(local_paths, frame_start, frame_end):
                raise ValueError(f"Downloaded episode {local_rectified_data_dir} is missing required files post-processing.")

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

    def _download_and_sync(
        self,
        episode_info: dict,
        episode_uri: str,
        local_paths: PathManager,
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

        if episode_uri.startswith("s3://"):
            parsed = urlparse(episode_uri)
            bucket_name = parsed.netloc
            s3_base_prefix = posixpath.dirname(parsed.path.lstrip("/"))

            # boto3 config
            config = botocore.config.Config(max_pool_connections=s3_concurrency)
            session = boto3.Session(profile_name=self.aws_profile)
            s3_client = session.client("s3", config=config)

            def _sync_file(src: str, dst: str):
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    p = urlparse(src)
                    s3_client.download_file(p.netloc, p.path.lstrip("/"), dst)

            src_paths = PathManager(f"s3://{bucket_name}/{s3_base_prefix}")
        else:
            # local sync
            src_rectified_data_dir = os.path.abspath(os.path.dirname(episode_uri))
            src_paths = PathManager(src_rectified_data_dir)

            def _sync_file(src: str, dst: str):
                if not os.path.exists(dst) and os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)

        all_download_tasks = [
            (src_paths.timestamp_txt, local_paths.timestamp_txt),
            (src_paths.stereo_params_npz, local_paths.stereo_params_npz),
            (src_paths.slam_trajectory_txt, local_paths.slam_trajectory_txt),
        ]
        for frame_idx in range(frame_start, frame_end):
            npz_filename = f"frame_{frame_idx:06d}.npz"
            jpg_filename = f"frame_{frame_idx:06d}.jpg"

            all_download_tasks.append(
                (posixpath.join(src_paths.hand_pose_dir, npz_filename), os.path.join(local_paths.hand_pose_dir, npz_filename))
            )
            all_download_tasks.append(
                (posixpath.join(src_paths.front_pcd_dir, npz_filename), os.path.join(local_paths.front_pcd_dir, npz_filename))
            )
            all_download_tasks.append(
                (posixpath.join(src_paths.eye_pcd_dir, npz_filename), os.path.join(local_paths.eye_pcd_dir, npz_filename))
            )

            for cam in self.active_cameras:
                all_download_tasks.append(
                    (
                        posixpath.join(src_paths.rectified_dir, cam, jpg_filename),
                        os.path.join(local_paths.rectified_dir, cam, jpg_filename),
                    )
                )

        # pull from s3 with concurrency
        with ThreadPoolExecutor(max_workers=s3_concurrency) as executor:
            futures = [executor.submit(_sync_file, src, dst) for src, dst in all_download_tasks]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Downloading {episode_id} (Files)", leave=False
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed to sync a file: {e}")

        print(f"Finished downloading {episode_id}.")

    def _merge_hand_streams(self, path_manager: PathManager, frame_start: int, frame_end: int):
        """
        Phase 1: Project missing front detections from the eye cameras and save to disk.
        Phase 2: Group remaining complete gaps and fill them using Linear Interpolation (LERP).
        """
        stereo_npz = np.load(path_manager.stereo_params_npz, allow_pickle=True)
        T_f2e_unrect = stereo_npz["T_front_to_eye"]

        # 1. unrectify eye
        R_eye_4x4 = np.eye(4)
        R_eye_4x4[:3, :3] = stereo_npz["eye_R1"]
        R_eye_inv = np.linalg.inv(R_eye_4x4)

        # 2. eye-to-front projection
        T_e2f_unrect = np.linalg.inv(T_f2e_unrect)

        # 3. rectify front
        R_front_4x4 = np.eye(4)
        R_front_4x4[:3, :3] = stereo_npz["front_R1"]

        # combine: unrectify eye -> eye-to-front -> rectify front
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

            # if left front is missing, check left eye
            if is_missing(l_front):
                l_eye = (left_data.get("eye") or {}).get("keypoints_3d_rectcam")
                l_front = project_eye_to_front(l_eye)
                if not is_missing(l_front):
                    needs_save = True

            # if right front is missing, check right eye
            if is_missing(r_front):
                r_eye = (right_data.get("eye") or {}).get("keypoints_3d_rectcam")
                r_front = project_eye_to_front(r_eye)
                if not is_missing(r_front):
                    needs_save = True

            # if we successfully recovered data from the eyes, SAVE it immediately
            if needs_save:
                out_left = left_data.copy()
                out_right = right_data.copy()

                if not is_missing(l_front):
                    out_left.setdefault("front", {})["keypoints_3d_rectcam"] = l_front
                if not is_missing(r_front):
                    out_right.setdefault("front", {})["keypoints_3d_rectcam"] = r_front

                np.savez(filepath, left=np.array(out_left), right=np.array(out_right))

            # Record if it is STILL missing after attempting the eye fallback
            if is_missing(l_front):
                missing_left.append(i)
            if is_missing(r_front):
                missing_right.append(i)

        # check for lerp
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
                # 1. seek backwards for nearest valid frame
                start_valid = gap[0] - 1
                start_kp = None
                while start_valid >= 0:
                    l, r = _load_front_only(start_valid)
                    start_kp = l if is_left else r
                    if start_kp is not None:
                        break
                    start_valid -= 1

                # 2. seek forwards for nearest valid frame
                end_valid = gap[-1] + 1
                end_kp = None
                while end_valid < end_valid + 10000:
                    l, r = _load_front_only(end_valid)
                    end_kp = l if is_left else r
                    if end_kp is not None:
                        break
                    end_valid += 1

                if start_kp is None or end_kp is None:
                    continue

                # 3. lerp the gap
                for i in gap:
                    w = (i - start_valid) / (end_valid - start_valid)
                    interp_kp = start_kp + w * (end_kp - start_kp)

                    filepath = os.path.join(path_manager.hand_pose_dir, f"frame_{i:06d}.npz")

                    out_left, out_right = {}, {}
                    if os.path.exists(filepath):
                        with np.load(filepath, allow_pickle=True) as d:
                            out_left = d["left"].item()
                            out_right = d["right"].item()

                    # Merge the interpolated data safely
                    if is_left:
                        out_left.setdefault("front", {})["keypoints_3d_rectcam"] = interp_kp
                    else:
                        out_right.setdefault("front", {})["keypoints_3d_rectcam"] = interp_kp

                    np.savez(filepath, left=np.array(out_left), right=np.array(out_right))

        _process_hand_gaps(_group_gaps(missing_left), is_left=True)
        _process_hand_gaps(_group_gaps(missing_right), is_left=False)

    def _validate_episode_dir(self, path_manager: PathManager, frame_start: int, frame_end: int) -> bool:
        required = [
            path_manager.timestamp_txt,
            path_manager.stereo_params_npz,
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


class EgoEpisode(Dataset):
    """A pure data-reading representation of a single episode interval."""

    CAMS = ["left-front", "right-front", "left-eye", "right-eye"]

    def __init__(
        self,
        rectified_data_dir: str,
        start_frame: int,
        end_frame: int,
        active_cameras: List[str],
        caption: Optional[str] = None,
    ):
        self.rectified_data_dir = rectified_data_dir
        self.path_manager = PathManager(rectified_data_dir)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.active_cameras = active_cameras
        self.caption = caption

        assert set(active_cameras).issubset(set(self.CAMS))

        stereo_npz = np.load(self.path_manager.stereo_params_npz, allow_pickle=True)
        self.stereo_params = {key: stereo_npz[key] for key in stereo_npz.files}

        with open(self.path_manager.timestamp_txt, "r") as f:
            self.timestamps = f.read().strip().split("\n")

        traj_data = np.loadtxt(self.path_manager.slam_trajectory_txt, comments="#")
        if traj_data.ndim == 1:
            traj_data = traj_data[None, :]
        self.c2w_timestamps = (traj_data[:, 0] * 1e9).astype(np.int64)
        self.c2w_poses = traj_data[:, 1:]

    def _load_hand_streams(self, global_frame: int):
        filepath = os.path.join(self.path_manager.hand_pose_dir, f"frame_{global_frame:06d}.npz")
        if not os.path.exists(filepath):
            return None, None

        with np.load(filepath, allow_pickle=True) as hand_pose_data:
            left_data = hand_pose_data["left"].item()
            right_data = hand_pose_data["right"].item()
            l_front = left_data.get("front", {}).get("keypoints_3d_rectcam")
            r_front = right_data.get("front", {}).get("keypoints_3d_rectcam")

        def is_missing(kp):
            return kp is None or np.size(kp) == 0 or np.all(kp == 0)

        return (None if is_missing(l_front) else l_front, None if is_missing(r_front) else r_front)

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

        l_front, r_front = self._load_hand_streams(global_frame)
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
            stereo_params=self.stereo_params,
            left_hand_kp=l_front,
            right_hand_kp=r_front,
            left_front_depth=left_front_depth,
            left_eye_depth=left_eye_depth,
            c2w=c2w,
        )


class EgoDataset(Dataset):
    """
    Main Dataset entry point.
    Maps indices to cached episodes using the CacheManager.
    """

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
        self.active_cameras = active_cameras or ["left-front", "right-front"]
        self.cache_manager = CacheManager(target_dir=target_dir, aws_profile=aws_profile, active_cameras=self.active_cameras)

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
                        try:
                            self.captions_map.update(json.loads(line))
                        except json.JSONDecodeError:
                            pass

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
        pass  # not implemented yet

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> Union[EgoEpisode, List[EgoEpisode]]:
        # Support dataset slicing
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
