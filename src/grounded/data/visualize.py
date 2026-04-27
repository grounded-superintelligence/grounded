"""
Visualization script for viewing EgoDataset episodes
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from grounded.data.ego_dataset import EgoEpisode

LEFT_HAND_COLOR = (165, 255, 100)
RIGHT_HAND_COLOR = (255, 100, 200)
JOINTS_COLOR = (255, 255, 255)

HAND_EDGES = [
    (0, 1),
    (1, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (17, 0),  # palm
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
]


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

    for i, j in HAND_EDGES:
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


def warp_left_depth_to_right(left_depth: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Forward-warps a rectified left depth map to the right camera view."""
    H, W = left_depth.shape
    right_depth = np.full((H, W), np.inf)

    y, x = np.indices((H, W))

    valid_mask = left_depth > 0
    z_valid = left_depth[valid_mask]
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # disparity
    Tx_fx = P2[0, 3]
    disparity = -Tx_fx / z_valid

    x_right = np.round(x_valid - disparity).astype(int)

    in_bounds = (x_right >= 0) & (x_right < W)
    x_right = x_right[in_bounds]
    y_right = y_valid[in_bounds]
    z_val = z_valid[in_bounds]

    # z-buffer sort (render far to near so near overwrites)
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
    norm = 1.0 - norm  # invert so closer is hotter

    norm = np.nan_to_num(norm, nan=0.0)

    norm_8u = (norm * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(norm_8u, cv2.COLORMAP_INFERNO)
    colorized[missing_mask] = 0

    return colorized


def as_4x4(x: np.ndarray):
    T = np.eye(4)
    if x.shape == (3, 3):
        T[:3, :3] = x
    elif x.shape == (3):
        T[:3, 3] = x
    else:
        raise ValueError(f"x.shape={x.shape} not supported")
    return T


def visualize_episode_to_mp4(
    episode: EgoEpisode,
    output_path: str,
    downsample: int = 2,
    fps: float = 30.0,
    max_workers: int = 8,
    max_depth: float = 20.0,
):
    if len(episode) == 0:
        print("Error: The provided episode is empty.")
        return

    writer = None
    vmin, vmax = 0.0, max_depth  # for visualizing depth

    params = episode.stereo_params

    # 1. compute front-to-eye transformation
    #
    # the device intrinsics/extrinsics are calibrated in unrectified frame,
    # but the images/points/depth are computed in rectified frame.
    # therefore, to go from front-eye, we need to apply the transformations
    #
    # rectified left-front -> unrectified left-front -> unrectified left-eye -> rectified left-eye
    #                      ||                        ||                      ||
    #                 inv(R_front)      ->      T_f2e_unrect       ->      R_eye
    T_f2e_unrect = params["T_front_to_eye"]
    R_front = as_4x4(params["front_R1"])
    R_eye = as_4x4(params["eye_R1"])
    T_f2e = R_eye @ T_f2e_unrect @ np.linalg.inv(R_front)

    def eye_transform(pts):
        return transform_points(pts, T_f2e)

    def _process_frame(i):
        frame = episode[i]
        cam_configs = []

        # 1. depth maps (always loaded)
        left_front_depth = cv2.resize(frame.left_front_depth, episode.LEFT_FRONT_WH, interpolation=cv2.INTER_NEAREST)
        right_front_depth = warp_left_depth_to_right(left_front_depth, params["front_P1"], params["front_P2"])
        left_eye_depth = cv2.resize(frame.left_eye_depth, episode.LEFT_FRONT_WH, interpolation=cv2.INTER_NEAREST)
        right_eye_depth = warp_left_depth_to_right(left_eye_depth, params["eye_P1"], params["eye_P2"])

        # 2. hand keypoints projections per camera (if in active_cameras)
        if "left-front" in episode.active_cameras:
            cam_configs.append(
                {
                    "img": frame.left_front_rgb,
                    "P": params["front_P1"],
                    "depth": left_front_depth,
                    "transform": lambda pts: pts,
                }
            )
        if "right-front" in episode.active_cameras:
            cam_configs.append(
                {
                    "img": frame.right_front_rgb,
                    "P": params["front_P2"],
                    "depth": right_front_depth,
                    "transform": lambda pts: pts,
                }
            )
        if "left-eye" in episode.active_cameras:
            cam_configs.append(
                {
                    "img": frame.left_eye_rgb,
                    "P": params["eye_P1"],
                    "depth": left_eye_depth,
                    "transform": eye_transform,
                }
            )
        if "right-eye" in episode.active_cameras:
            cam_configs.append(
                {
                    "img": frame.right_eye_rgb,
                    "P": params["eye_P2"],
                    "depth": right_eye_depth,
                    "transform": eye_transform,
                }
            )

        # 3. loop through each camera and visualize
        processed_rows = []
        for config in cam_configs:
            img_rgb = config["img"]
            P = config["P"]
            depth_raw = config["depth"]

            if img_rgb is not None:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # project left hand keypoints
                if frame.left_hand_kp is not None and len(frame.left_hand_kp) > 0:
                    l_pts = config["transform"](frame.left_hand_kp)
                    l_kps2d = project_points(l_pts, P)
                    img_bgr = draw_uv_skeleton(draw_uv_points(img_bgr, l_kps2d, False), l_kps2d, False)

                # project right hand keypoints
                if frame.right_hand_kp is not None and len(frame.right_hand_kp) > 0:
                    r_pts = config["transform"](frame.right_hand_kp)
                    r_kps2d = project_points(r_pts, P)
                    img_bgr = draw_uv_skeleton(draw_uv_points(img_bgr, r_kps2d, True), r_kps2d, True)

                # normalize depth
                depth_color = colorize_normalized_depth(depth_raw, vmin, vmax)

                # tile: [rgb + hand kp | depth map]
                row = np.hstack([img_bgr, depth_color])
                processed_rows.append(row)

        stacked = np.vstack(processed_rows)

        # downsample to reduce file size
        if downsample > 1:
            h, w = stacked.shape[:2]
            new_w = w // downsample
            new_h = int(h * (new_w / w))
            stacked = cv2.resize(stacked, (new_w, new_h))

        # draw caption
        if hasattr(episode, "caption") and episode.caption:
            caption_text = f"Caption: {episode.caption}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(caption_text, font, font_scale, thickness)
            margin = 15
            x = margin
            y = stacked.shape[0] - margin - baseline

            overlay = stacked.copy()
            cv2.rectangle(overlay, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, stacked, 0.4, 0, stacked)
            cv2.putText(stacked, caption_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return stacked

    # threaded video rendering
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rendered_frames = executor.map(_process_frame, range(len(episode)))
        for stacked in tqdm(rendered_frames, total=len(episode), desc="visualizing 2d", leave=False):
            if writer is None:
                target_h, target_w = stacked.shape[:2]
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))
            writer.write(stacked)

    if writer is not None:
        writer.release()
        print(f"Saved to {output_path}")
    else:
        print("Error: No frames were written to the video.")
