#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from grounded.data.ego_dataset import EgoEpisode

# --- Drawing Constants & Helpers ---
LEFT_HAND_COLOR = (165, 255, 100)
RIGHT_HAND_COLOR = (255, 100, 200)
JOINTS_COLOR = (255, 255, 255)


def transform_points(points_3d: np.ndarray, T: np.ndarray):
    """Applies a 4x4 homogenous transformation matrix to 3D points."""
    if points_3d is None or len(points_3d) == 0:
        return points_3d
    ones = np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)
    points_3d_h = np.concatenate([points_3d, ones], axis=-1)
    transformed_h = (T @ points_3d_h.T).T
    return transformed_h[:, :3]


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

    return right_depth


def colorize_normalized_depth(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalizes a depth map against global bounds and applies a colormap."""
    if vmax <= vmin:
        vmax = vmin + 1e-5

    missing_mask = depth == 0

    depth_clipped = np.clip(depth, a_min=vmin, a_max=vmax)
    norm = (depth_clipped - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    norm = 1.0 - norm  # Invert so closer is hotter

    norm = np.nan_to_num(norm, nan=0.0)

    norm_8u = (norm * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(norm_8u, cv2.COLORMAP_INFERNO)
    colorized[missing_mask] = 0

    return colorized


def ensure_depth_size(depth: np.ndarray, img_shape: tuple) -> np.ndarray:
    """Resizes depth map to match RGB dimensions if needed."""
    if depth is None:
        return None
    h, w = img_shape[:2]
    if depth.shape[:2] != (h, w):
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth


# --- Visualization Routines ---


def visualize_episode_to_mp4(
    episode: EgoEpisode,
    output_path: str,
    downsample: int = 2,
    fps: float = 30.0,
    max_workers: int = 4,
    max_depth: float = 20.0,
):
    if len(episode) == 0:
        print("Error: The provided episode is empty.")
        return

    out_path = output_path if output_path.endswith(".mp4") else f"{output_path}.mp4"
    writer = None

    # Use fixed bounds instead of precomputing
    vmin, vmax = 0.0, max_depth

    # Frame processing function for mapping over executor
    def _process_frame(i):
        frame = episode[i]
        params = frame.stereo_params

        cam_configs = []

        # --- Process Front Cameras ---
        left_front_depth = (
            ensure_depth_size(frame.left_front_depth, frame.left_front_rgb.shape) if frame.left_front_rgb is not None else None
        )

        if "left-front" in episode.active_cameras and frame.left_front_rgb is not None:
            cam_configs.append(
                {
                    "img": frame.left_front_rgb,
                    "P": params["front_P1"],
                    "depth": left_front_depth,
                    "transform": lambda pts: pts,
                }
            )

        if "right-front" in episode.active_cameras and frame.right_front_rgb is not None:
            right_front_depth = None
            if left_front_depth is not None:
                right_front_depth = warp_left_depth_to_right(left_front_depth, params["front_P1"], params["front_P2"])
            cam_configs.append(
                {
                    "img": frame.right_front_rgb,
                    "P": params["front_P2"],
                    "depth": right_front_depth,
                    "transform": lambda pts: pts,
                }
            )

        # --- Process Eye Cameras ---
        left_eye_depth = (
            ensure_depth_size(frame.left_eye_depth, frame.left_eye_rgb.shape) if frame.left_eye_rgb is not None else None
        )

        T_f2e_unrect = params.get("T_front_to_eye")
        if T_f2e_unrect is not None:
            # 1. Inverse of Front Rectification
            R_front_4x4 = np.eye(4)
            R_front_4x4[:3, :3] = params["front_R1"]
            R_front_inv = np.linalg.inv(R_front_4x4)

            # 2. Eye Rectification
            R_eye_4x4 = np.eye(4)
            R_eye_4x4[:3, :3] = params["eye_R1"]

            # Combine: Un-rectify Front -> Extrinsics to Eye -> Rectify Eye
            T_rectfront_to_recteye = R_eye_4x4 @ T_f2e_unrect @ R_front_inv

            def eye_transform(pts):
                return transform_points(pts, T_rectfront_to_recteye)
        else:

            def eye_transform(pts):
                return pts

        if "left-eye" in episode.active_cameras and frame.left_eye_rgb is not None:
            cam_configs.append(
                {
                    "img": frame.left_eye_rgb,
                    "P": params["eye_P1"],
                    "depth": left_eye_depth,
                    "transform": eye_transform,
                }
            )

        if "right-eye" in episode.active_cameras and frame.right_eye_rgb is not None:
            right_eye_depth = None
            if left_eye_depth is not None:
                right_eye_depth = warp_left_depth_to_right(left_eye_depth, params["eye_P1"], params["eye_P2"])
            cam_configs.append(
                {
                    "img": frame.right_eye_rgb,
                    "P": params["eye_P2"],
                    "depth": right_eye_depth,
                    "transform": eye_transform,
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

            # Handle Depth
            if depth_raw is not None:
                depth_color = colorize_normalized_depth(depth_raw, vmin, vmax)
                if depth_color.shape[:2] != img_bgr.shape[:2]:
                    depth_color = cv2.resize(depth_color, (img_bgr.shape[1], img_bgr.shape[0]))
            else:
                depth_color = np.zeros_like(img_bgr)

            # Tile: [RGB with Overlays | Depth Map]
            row = np.hstack([img_bgr, depth_color])
            processed_rows.append(row)

        if not processed_rows:
            return None

        stacked = np.vstack(processed_rows)

        if downsample > 1:
            h, w = stacked.shape[:2]
            new_w = w // downsample
            new_h = int(h * (new_w / w))
            stacked = cv2.resize(stacked, (new_w, new_h))

        if hasattr(episode, "caption") and episode.caption:
            caption_text = f"Caption: {episode.caption}"

            # Setup font configurations
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Retrieve text bounds for the background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(caption_text, font, font_scale, thickness)

            # Position the text at the bottom left
            margin = 15
            x = margin
            y = stacked.shape[0] - margin - baseline

            # Draw semi-transparent background via overlay (alpha blending)
            overlay = stacked.copy()
            cv2.rectangle(overlay, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, stacked, 0.4, 0, stacked)

            # Draw the actual text on top
            cv2.putText(stacked, caption_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return stacked

    # Render Video
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map guarantees results are returned in the original sequential order
        rendered_frames = executor.map(_process_frame, range(len(episode)))

        for stacked in tqdm(rendered_frames, total=len(episode), desc="Rendering Video", leave=False):
            if stacked is None:
                continue

            if writer is None:
                target_h, target_w = stacked.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

            writer.write(stacked)

    if writer is not None:
        writer.release()
        print(f"Saved to {out_path}")
    else:
        print("Error: No frames were written to the video.")
