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
    left_front_rgb: Optional[np.ndarray]
    right_front_rgb: Optional[np.ndarray]
    left_eye_rgb: Optional[np.ndarray]
    right_eye_rgb: Optional[np.ndarray]
    stereo_params: Dict[str, np.ndarray]
    left_hand_kp: np.ndarray
    right_hand_kp: np.ndarray
    left_front_depth: np.ndarray
    left_eye_depth: np.ndarray
    c2w: Optional[np.ndarray]  # [tx, ty, tz, qx, qy, qz, qw]


class PathManager:
    def __init__(self, rectified_dir: str):
        self.rectified_dir = rectified_dir  # Save base dir to easily construct dynamic camera paths

        # .../processed-segmentXX/hand/
        hand_dir = posixpath.dirname(rectified_dir)
        # .../processed-segmentXX/
        processed_dir = posixpath.dirname(hand_dir)

        # Define synchronized sub-paths
        self.hand_pose_dir = posixpath.join(hand_dir, "hand_tracking", "poses", "refined", "params")
        self.front_pcd_dir = posixpath.join(hand_dir, "compressed_pcds", "left-front")
        self.eye_pcd_dir = posixpath.join(hand_dir, "compressed_pcds", "left-eye")
        self.slam_trajectory_txt = posixpath.join(processed_dir, "slam", "mav0", "pycuvslam_trajectory.txt")
        self.stereo_params_npz = posixpath.join(rectified_dir, "stereo_params.npz")
        self.timestamp_txt = posixpath.join(rectified_dir, "timestamp.txt")


class EgoEpisode:
    """A lazy-loaded, sliceable representation of a single episode interval."""

    CAMS = [
        "left-front",
        "right-front",
        "left-eye",
        "right-eye",
    ]

    def __init__(
        self,
        rectified_data_dir: str,
        start_frame: int,
        end_frame: int,
        active_cameras: list[str],
        caption: Optional[str] = None,
    ):
        self.rectified_data_dir = rectified_data_dir
        self.path_manager = PathManager(rectified_data_dir)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.active_cameras = active_cameras
        self.caption = caption

        assert set(active_cameras).issubset(set(EgoEpisode.CAMS))

        stereo_npz = np.load(self.path_manager.stereo_params_npz, allow_pickle=True)
        self.stereo_params = {key: stereo_npz[key] for key in stereo_npz.files}

        with open(self.path_manager.timestamp_txt, "r") as f:
            self.timestamps = f.read().strip().split("\n")

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
        max_delta_ns = self._compute_and_print_max_delta()
        print(f"Max timestamp delta (rgb/slam) for frames [{self.start_frame}-{self.end_frame}]: {max_delta_ns / 1e6:.3f} ms")

    def _compute_and_print_max_delta(self):
        max_delta_ns = 0
        for i in range(self.start_frame, self.end_frame):
            if i < len(self.timestamps):
                frame_ts_ns = int(self.timestamps[i])
                closest_idx = np.abs(self.c2w_timestamps - frame_ts_ns).argmin()
                delta = abs(self.c2w_timestamps[closest_idx] - frame_ts_ns)
                if delta > max_delta_ns:
                    max_delta_ns = delta
        return max_delta_ns

    def _load_and_merge_hand_streams(self, global_frame: int):
        """Now purely reads from disk since all recovery/interpolation happens during download."""
        with np.load(
            os.path.join(self.path_manager.hand_pose_dir, f"frame_{global_frame:06d}.npz"),
            allow_pickle=True,
        ) as hand_pose_data:
            left_data = hand_pose_data["left"].item()
            right_data = hand_pose_data["right"].item()

            l_front = left_data.get("front", {}).get("keypoints_3d_rectcam")
            r_front = right_data.get("front", {}).get("keypoints_3d_rectcam")

        def is_missing(kp):
            return kp is None or np.size(kp) == 0 or np.all(kp == 0)

        if is_missing(l_front):
            l_front = None
        if is_missing(r_front):
            r_front = None

        return l_front, r_front

    def _load_depth_stream(self, global_frame: int, cam_name: str) -> np.ndarray | None:
        # only left-front and left-eye pcds are saved
        if cam_name == "left-front":
            pcd_path = os.path.join(self.path_manager.front_pcd_dir, f"frame_{global_frame:06d}.npz")
        else:
            pcd_path = os.path.join(self.path_manager.eye_pcd_dir, f"frame_{global_frame:06d}.npz")
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

        # Load jpg frames directly via cv2.imread
        for cam in self.active_cameras:
            img_path = os.path.join(self.rectified_data_dir, cam, f"frame_{global_frame:06d}.jpg")
            frame_bgr = cv2.imread(img_path)

            if frame_bgr is None:
                raise FileNotFoundError(f"Missing image frame: {img_path}")

            imgs[cam] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        l_front, r_front = self._load_and_merge_hand_streams(global_frame)
        left_front_depth = self._load_depth_stream(global_frame, "left-front")
        left_eye_depth = self._load_depth_stream(global_frame, "left-eye")

        # Match c2w pose
        frame_ts_ns = int(timestamp_ns)
        closest_idx = np.abs(self.c2w_timestamps - frame_ts_ns).argmin()
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


class EgoDataset:
    def __init__(
        self,
        index_path: str,
        captions_path: str = None,
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

        # Load Captions
        self.captions_map = {}
        if captions_path:
            cap_path = Path(captions_path).expanduser()
            if cap_path.exists():
                with open(cap_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            # Handles standard format JSONL loading
                            entry = json.loads(line)
                            self.captions_map.update(entry)
                        except json.JSONDecodeError:
                            pass

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
        if self.captions_map:
            print(f"Loaded {len(self.captions_map)} captions from {captions_path}.")

    def _interpolate_missing_frames(self, path_manager: PathManager, frame_start: int, frame_end: int):
        """
        Phase 1: Project missing front detections from the eye cameras and save to disk.
        Phase 2: Group remaining complete gaps and fill them using Linear Interpolation (LERP).
        """
        stereo_npz = np.load(path_manager.stereo_params_npz, allow_pickle=True)
        T_f2e_unrect = stereo_npz["T_front_to_eye"]

        # 1. Inverse of Eye Rectification (Un-rectify Eye)
        R_eye_4x4 = np.eye(4)
        R_eye_4x4[:3, :3] = stereo_npz["eye_R1"]
        R_eye_inv = np.linalg.inv(R_eye_4x4)

        # 2. Eye to Front Extrinsics (Unrectified)
        T_e2f_unrect = np.linalg.inv(T_f2e_unrect)

        # 3. Front Rectification (Rectify Front)
        R_front_4x4 = np.eye(4)
        R_front_4x4[:3, :3] = stereo_npz["front_R1"]

        # Combine: Un-rectify Eye -> Extrinsics to Front -> Rectify Front
        T_recteye_to_rectfront = R_front_4x4 @ T_e2f_unrect @ R_eye_inv

        def is_missing(kp):
            return kp is None or np.size(kp) == 0 or np.all(kp == 0)

        def project_eye_to_front(kp_eye):
            if is_missing(kp_eye) or T_recteye_to_rectfront is None:
                return None
            ones = np.ones((kp_eye.shape[0], 1), dtype=kp_eye.dtype)
            kp_eye_h = np.concatenate([kp_eye, ones], axis=-1)
            return (T_recteye_to_rectfront @ kp_eye_h.T).T[:, :3]

        # ==========================================
        # PHASE 1: EYE-TO-FRONT PROJECTION FALLBACK
        # ==========================================

        # We will keep track of what still failed AFTER projection for Phase 2
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

            # If left front is missing, check left eye
            if is_missing(l_front):
                l_eye = (left_data.get("eye") or {}).get("keypoints_3d_rectcam")
                l_front = project_eye_to_front(l_eye)
                if not is_missing(l_front):
                    needs_save = True

            # If right front is missing, check right eye
            if is_missing(r_front):
                r_eye = (right_data.get("eye") or {}).get("keypoints_3d_rectcam")
                r_front = project_eye_to_front(r_eye)
                if not is_missing(r_front):
                    needs_save = True

            # If we successfully recovered data from the eyes, SAVE it immediately
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

        # If everything is tracked, we can skip Phase 2 entirely!
        if not missing_left and not missing_right:
            return

        # ==========================================
        # PHASE 2: LINEAR INTERPOLATION (LERP)
        # ==========================================

        # Because Phase 1 saved the projected data to disk, our LERP helper
        # just blindly reads the "front" key, knowing it contains the best possible data.
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
                # 1. Seek backwards for nearest valid frame
                start_valid = gap[0] - 1
                start_kp = None
                while start_valid >= 0:
                    l, r = _load_front_only(start_valid)
                    start_kp = l if is_left else r
                    if start_kp is not None:
                        break
                    start_valid -= 1

                # 2. Seek forwards for nearest valid frame
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

                # 3. LERP the gap
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
            path_manager.slam_trajectory_txt,
            path_manager.hand_pose_dir,
            path_manager.front_pcd_dir,
            path_manager.eye_pcd_dir,
        ]

        # Require camera directories instead of mp4s
        for cam in self.active_cameras:
            required.append(os.path.join(path_manager.rectified_dir, cam))

        # 1. Check if base required files and directories exist
        for p in required:
            if not os.path.exists(p):
                print(f"\n[VALIDATION ERROR] Missing core file or directory: {p}")
                return False

        # 2. Validate that the specific requested boundary frames actually downloaded
        if frame_end > frame_start:
            for i in (frame_start, frame_end - 1):
                hand_file = os.path.join(path_manager.hand_pose_dir, f"frame_{i:06d}.npz")
                front_pcd_file = os.path.join(path_manager.front_pcd_dir, f"frame_{i:06d}.npz")
                eye_pcd_file = os.path.join(path_manager.eye_pcd_dir, f"frame_{i:06d}.npz")

                if not os.path.exists(hand_file):
                    print(f"\n[VALIDATION ERROR] Missing boundary hand pose: {hand_file}")
                    return False
                if not os.path.exists(front_pcd_file):
                    print(f"\n[VALIDATION ERROR] Missing boundary point cloud: {front_pcd_file}")
                    return False
                if not os.path.exists(eye_pcd_file):
                    print(f"\n[VALIDATION ERROR] Missing boundary point cloud: {eye_pcd_file}")
                    return False

                # Validate boundary jpg images
                for cam in self.active_cameras:
                    img_file = os.path.join(path_manager.rectified_dir, cam, f"frame_{i:06d}.jpg")
                    if not os.path.exists(img_file):
                        print(f"\n[VALIDATION ERROR] Missing boundary image frame: {img_file}")
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
        os.makedirs(local_paths.front_pcd_dir, exist_ok=True)
        os.makedirs(local_paths.eye_pcd_dir, exist_ok=True)

        # Ensure image directories exist
        for cam in self.active_cameras:
            os.makedirs(os.path.join(local_paths.rectified_dir, cam), exist_ok=True)

        # 1. Sync Core Files (Videos removed)
        files_to_sync = [
            (src_paths.timestamp_txt, local_paths.timestamp_txt),
            (src_paths.stereo_params_npz, local_paths.stereo_params_npz),
            (src_paths.slam_trajectory_txt, local_paths.slam_trajectory_txt),
        ]

        for src_path, dst_path in files_to_sync:
            _sync_file(src_path, dst_path)

        # 2. Sync Frame-level Files Concurrently (Now includes .npz AND .jpg)
        def sync_single_frame(frame_idx: int):
            npz_filename = f"frame_{frame_idx:06d}.npz"
            jpg_filename = f"frame_{frame_idx:06d}.jpg"

            frame_targets = [
                (posixpath.join(src_paths.hand_pose_dir, npz_filename), os.path.join(local_paths.hand_pose_dir, npz_filename)),
                (posixpath.join(src_paths.front_pcd_dir, npz_filename), os.path.join(local_paths.front_pcd_dir, npz_filename)),
                (posixpath.join(src_paths.eye_pcd_dir, npz_filename), os.path.join(local_paths.eye_pcd_dir, npz_filename)),
            ]

            # Inject active camera image paths into the concurrent sync targets
            for cam in self.active_cameras:
                src_img_path = posixpath.join(src_paths.rectified_dir, cam, jpg_filename)
                dst_img_path = os.path.join(local_paths.rectified_dir, cam, jpg_filename)
                frame_targets.append((src_img_path, dst_img_path))

            try:
                for src_path, dst_path in frame_targets:
                    _sync_file(src_path, dst_path)
            except Exception:
                # Silently ignore missing files; the interpolator will catch and synthesize hand gaps.
                # A missing image will correctly be flagged by the boundary check at the end.
                pass

        with ThreadPoolExecutor(max_workers=s3_concurrency) as executor:
            futures = [executor.submit(sync_single_frame, i) for i in range(frame_start, frame_end)]
            for future in as_completed(futures):
                future.result()

        # RUN DATA CLEANING & RECOVERY BEFORE VALIDATION
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

    def get_caption(self, idx: int) -> str:
        """Returns the caption for a given episode index, if available."""
        episode_info = self.index[idx]

        device_id = episode_info["device_id"]
        session_num = episode_info["session_num"]
        segment_num = episode_info["segment_num"]
        f_start = episode_info["frame_start"]
        f_end = episode_info["frame_end"]

        key_full = f"{device_id}_session_{session_num}_segment_{segment_num}_interval_{f_start}_{f_end}"
        key_short = f"{device_id}_interval_{f_start}_{f_end}"

        return self.captions_map.get(key_full) or self.captions_map.get(key_short)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> EgoEpisode:
        # Support slicing operations
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]

        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self.__getitem__(int(i)) for i in idx]

        episode_info = self.index[idx]

        # Download files and get the local path
        local_rectified_dir = self.download_episode(idx)

        # Lookup caption mapping dynamically
        episode_caption = self.get_caption(idx)

        # Pass the local directory and identified caption to EgoEpisode
        return EgoEpisode(
            rectified_data_dir=local_rectified_dir,
            start_frame=episode_info["frame_start"],
            end_frame=episode_info["frame_end"],
            active_cameras=self.active_cameras,
            caption=episode_caption,
        )
