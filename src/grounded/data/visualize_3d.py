import os
from typing import Tuple

import cv2
import numpy as np
import rerun as rr
from tqdm import tqdm

from grounded.data.ego_dataset import EgoEpisode
from grounded.data.visualize import HAND_EDGES, LEFT_HAND_COLOR, RIGHT_HAND_COLOR


def extract_intrinsics(P: np.ndarray) -> Tuple[float, float, float, float]:
    """Extracts focal lengths and principal points from a 3x4 projection matrix."""
    fx, fy = P[0, 0], P[1, 1]
    cx, cy = P[0, 2], P[1, 2]
    return fx, fy, cx, cy


def unproject_depth(
    depth: np.ndarray,
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    step: int = 1,
    max_depth: int = 2.5,  # meters
) -> Tuple[np.ndarray, np.ndarray]:
    """Unprojects a depth map into 3D points in the local camera coordinate frame."""
    # subsample uniformly across height and width BEFORE flattening
    depth_sub = depth[::step, ::step]
    rgb_sub = rgb[::step, ::step]
    u_sub = u_grid[::step, ::step]
    v_sub = v_grid[::step, ::step]

    # filter valid depths
    valid = (depth_sub > 0) & (depth_sub < max_depth) & np.isfinite(depth_sub)
    z = depth_sub[valid]
    u = u_sub[valid]
    v = v_sub[valid]

    # unproject
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_local = np.stack([x, y, z], axis=-1)
    colors = rgb_sub[valid]

    return points_local, colors


def log_hand_to_rerun(entity_path: str, keypoints: np.ndarray, color_bgr: tuple):
    """Plots hand joints and bone segments, or clears them if keypoints are missing."""
    if keypoints is None or len(keypoints) == 0:
        rr.log(entity_path, rr.Clear(recursive=True))
        return

    # visualize.py colors are bgr, rerun expects rgb
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    # joints
    rr.log(f"{entity_path}/joints", rr.Points3D(keypoints, colors=[color_rgb] * len(keypoints), radii=0.005))

    # bones (line strips)
    strips = []
    for i, j in HAND_EDGES:
        if i < len(keypoints) and j < len(keypoints):
            strips.append([keypoints[i], keypoints[j]])

    if strips:
        rr.log(f"{entity_path}/bones", rr.LineStrips3D(strips, colors=[color_rgb] * len(strips)))


def visualize_episode_to_rerun(
    episode: EgoEpisode,
    output_path: str,
    pcd_downsample: int = 5,
    fps_downsample: int = 3,
):
    """
    Unprojects left-front depth maps into point clouds, aligns hands,
    and logs them to a Rerun file in the local camera frame.
    """
    assert "left-front" in episode.active_cameras, "Rerun visualizer needs left-front in active_cameras"

    if len(episode) == 0:
        print("Error: The provided episode is empty.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rr.init("EgoDataset 3D Vis", spawn=False)
    rr.save(output_path)

    fx, fy, cx, cy = extract_intrinsics(episode.stereo_params["front_P1"])
    W, H = episode.LEFT_FRONT_WH
    v_grid, u_grid = np.indices((H, W))

    # coordinate system is right/down/forward
    rr.log("camera", rr.ViewCoordinates.RDF, static=True)

    for i in tqdm(range(len(episode)), desc="visualizing 3d", leave=False):
        if i % fps_downsample > 0:
            continue

        frame = episode[i]

        rr.set_time("frame_idx", sequence=i)
        rr.set_time("timestamp", timestamp=np.datetime64(frame.timestamp_ns, "ns"))

        # unproject depth into local point clouds
        depth = cv2.resize(frame.left_front_depth, (W, H), interpolation=cv2.INTER_NEAREST)
        points_local, colors = unproject_depth(
            depth,
            frame.left_front_rgb,
            fx,
            fy,
            cx,
            cy,
            u_grid,
            v_grid,
            pcd_downsample,
        )

        rr.log("camera/point_cloud", rr.Points3D(points_local, colors=colors))

        # left hand (logged directly in local frame)
        log_hand_to_rerun("camera/left_hand", frame.left_hand_kp, LEFT_HAND_COLOR)

        # right hand (logged directly in local frame)
        log_hand_to_rerun("camera/right_hand", frame.right_hand_kp, RIGHT_HAND_COLOR)

    print(f"Saved Rerun visualizer file to: {output_path}")
