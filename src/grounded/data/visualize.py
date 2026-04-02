#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from grounded.data.ego_dataset import GROUNDED_DIR_DEFAULT, EgoDataset, EgoEpisode

# --- Drawing Constants & Helpers ---
LEFT_HAND_COLOR = (165, 255, 100)
RIGHT_HAND_COLOR = (255, 100, 200)
JOINTS_COLOR = (255, 255, 255)


def project_points(points_3d: np.ndarray, P: np.ndarray):
    """Projects 3D points into 2D."""
    if points_3d is None or len(points_3d) == 0 or np.linalg.norm(points_3d) < 1e-2:
        return np.zeros((0, 2), dtype=int)

    ones = np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)
    points_3d_h = np.concatenate([points_3d, ones], axis=-1)

    points_2d_h = (P @ points_3d_h.T).T
    z = points_2d_h[:, 2]
    z[np.abs(z) < 1e-5] = 1e-5
    u, v = points_2d_h[:, 0] / z, points_2d_h[:, 1] / z
    return np.stack([u, v], axis=-1)


def draw_uv_skeleton(image: np.ndarray, uvs: np.ndarray, is_right: bool):
    if len(uvs) == 0:
        return image
    img = image.copy()
    h, w = img.shape[:2]

    edges = [
        (0, 1),
        (1, 5),
        (5, 9),
        (9, 13),
        (13, 17),
        (17, 0),  # Palm
        (1, 2),
        (2, 3),
        (3, 4),  # Thumb
        (5, 6),
        (6, 7),
        (7, 8),  # Index
        (9, 10),
        (10, 11),
        (11, 12),  # Middle
        (13, 14),
        (14, 15),
        (15, 16),  # Ring
        (17, 18),
        (18, 19),
        (19, 20),  # Pinky
    ]
    for i, j in edges:
        if i >= len(uvs) or j >= len(uvs):
            continue
        u1, v1 = uvs[i]
        u2, v2 = uvs[j]
        if not np.all(np.isfinite([u1, v1, u2, v2])):
            continue
        p1, p2 = (int(round(u1)), int(round(v1))), (int(round(u2)), int(round(v2)))
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(img, p1, p2, JOINTS_COLOR, 3, cv2.LINE_AA)
    return img


def draw_uv_points(image: np.ndarray, uvs: np.ndarray, is_right: bool):
    if len(uvs) == 0:
        return image
    img = image.copy()
    h, w = img.shape[:2]
    color = RIGHT_HAND_COLOR if is_right else LEFT_HAND_COLOR
    for u, v in uvs:
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        x, y = int(round(u)), int(round(v))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
    return img


# --- Depth Processing Helpers ---


def get_global_depth_bounds(episode: EgoEpisode):
    """Scans the episode to find the absolute min and max valid depth values."""
    global_min = float("inf")
    global_max = float("-inf")

    for i in tqdm(range(len(episode)), desc="Finding depth bounds", leave=False):
        d = episode[i].left_depth
        if d is not None:
            valid = d[d > 0]
            if len(valid) > 0:
                global_min = min(global_min, valid.min())
                global_max = max(global_max, valid.max())

    if global_min == float("inf"):
        print("Warning: No valid depth data found in the episode. Using dummy bounds.")
        return 0.0, 1.0

    return global_min, global_max


def warp_left_depth_to_right(left_depth: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Forward-warps a rectified left depth map to the right camera view."""
    H, W = left_depth.shape
    right_depth = np.full((H, W), np.inf)

    y, x = np.indices((H, W))

    valid_mask = left_depth > 0
    z_valid = left_depth[valid_mask]
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Calculate disparity
    Tx_fx = P2[0, 3]
    disparity = -Tx_fx / z_valid

    x_right = np.round(x_valid - disparity).astype(int)

    in_bounds = (x_right >= 0) & (x_right < W)
    x_right = x_right[in_bounds]
    y_right = y_valid[in_bounds]
    z_val = z_valid[in_bounds]

    # Z-buffer sort (render far to near so near overwrites)
    sort_idx = np.argsort(z_val)[::-1]
    x_right = x_right[sort_idx]
    y_right = y_right[sort_idx]
    z_val = z_val[sort_idx]

    right_depth[y_right, x_right] = z_val
    right_depth[right_depth == np.inf] = 0

    # Morphological close to fill rounding cracks
    kernel = np.ones((3, 3), np.uint8)
    right_depth = cv2.morphologyEx(right_depth, cv2.MORPH_CLOSE, kernel)

    return right_depth


def colorize_normalized_depth(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalizes a depth map against global bounds and applies a colormap."""
    depth = np.clip(depth, a_min=vmin, a_max=vmax)
    norm = (depth - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    norm = 1.0 - norm  # Invert so closer is hotter

    norm_8u = (norm * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(norm_8u, cv2.COLORMAP_INFERNO)
    colorized[depth == 0] = 0

    return colorized


# --- Visualization Routines ---


def visualize_episode_to_mp4(episode: EgoEpisode, output_path: str, downsample: int = 2, fps: float = 30.0):
    """
    Renders an EgoEpisode object with canonical 2D keypoint projections and tiled depth maps.
    """
    if len(episode) == 0:
        print("Error: The provided episode is empty.")
        return

    out_path = output_path if output_path.endswith(".mp4") else f"{output_path}.mp4"
    writer = None

    # Pass 1: Global Bounds for consistent depth coloring across the video
    vmin, vmax = get_global_depth_bounds(episode)
    vmax = min(vmax, 20)

    # Pass 2: Render Video
    for i in tqdm(range(len(episode)), desc="Rendering Video", leave=False):
        frame = episode[i]
        params = frame.stereo_params

        # Prepare depths for active cameras
        left_depth = frame.left_depth
        rgb_h, rgb_w = frame.left_rgb.shape[:2]
        if left_depth.shape[:2] != (rgb_h, rgb_w):
            left_depth = cv2.resize(left_depth, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

        right_depth = None
        if "right-front" in episode.active_cameras:
            right_depth = warp_left_depth_to_right(
                left_depth,
                params["front_P1"],
                params["front_P2"],
            )

        cam_configs = []
        if "left-front" in episode.active_cameras:
            cam_configs.append(
                {
                    "img": frame.left_rgb,
                    "P": params["front_P1"],
                    "depth": left_depth,
                    "transform": lambda pts: pts,
                }
            )
        if "right-front" in episode.active_cameras:
            cam_configs.append(
                {
                    "img": frame.right_rgb,
                    "P": params["front_P2"],
                    "depth": right_depth,
                    "transform": lambda pts: pts,
                }
            )

        processed_rows = []

        for config in cam_configs:
            img_rgb = config["img"]
            P = config["P"]
            depth_raw = config["depth"]

            if img_rgb is None:
                continue

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Project Left Hand
            if frame.left_hand_kp is not None and len(frame.left_hand_kp) > 0:
                l_pts = config["transform"](frame.left_hand_kp)
                l_kps2d = project_points(l_pts, P)
                img_bgr = draw_uv_skeleton(draw_uv_points(img_bgr, l_kps2d, False), l_kps2d, False)

            # Project Right Hand
            if frame.right_hand_kp is not None and len(frame.right_hand_kp) > 0:
                r_pts = config["transform"](frame.right_hand_kp)
                r_kps2d = project_points(r_pts, P)
                img_bgr = draw_uv_skeleton(draw_uv_points(img_bgr, r_kps2d, True), r_kps2d, True)

            # Colorize Depth
            depth_color = colorize_normalized_depth(depth_raw, vmin, vmax)

            # Ensure depth shape strictly matches image shape before hstack
            if depth_color.shape[:2] != img_bgr.shape[:2]:
                depth_color = cv2.resize(depth_color, (img_bgr.shape[1], img_bgr.shape[0]))

            # Tile: [RGB with Overlays | Depth Map]
            row = np.hstack([img_bgr, depth_color])
            processed_rows.append(row)

        if not processed_rows:
            continue

        # Vertically stack all active cameras
        stacked = np.vstack(processed_rows)

        # Downsample the entire combined image block
        if downsample > 1:
            h, w = stacked.shape[:2]
            new_w = w // downsample
            new_h = int(h * (new_w / w))
            stacked = cv2.resize(stacked, (new_w, new_h))

        # Initialize VideoWriter dynamically based on final stacked frame size
        if writer is None:
            target_h, target_w = stacked.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

        writer.write(stacked)

    if writer is not None:
        writer.release()
        print(f"Saved to {out_path}")
    else:
        print("Error: No frames were written to the video.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=GROUNDED_DIR_DEFAULT)
    parser.add_argument("--profile", type=str, default=None, help="AWS profile to use for downloading")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    print(f"Loading dataset from {args.index_json}...")

    dataset = EgoDataset(
        index_path=args.index_json,
        active_cameras=["left-front", "right-front"],
        aws_profile=args.profile,
        target_dir=args.dataset_dir,
        min_duration_sec=2,
    )

    os.makedirs("outputs/", exist_ok=True)
    for episode_idx in range(10):
        if episode_idx >= len(dataset):
            break

        episode = dataset[episode_idx]

        visualize_episode_to_mp4(
            episode=episode,
            output_path=f"outputs/sdkvis{episode_idx}.mp4",
            downsample=args.downsample,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
