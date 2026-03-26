#!/usr/bin/env python3
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from grounded.data.ego_dataset import GROUNDED_DIR_DEFAULT, EgoDataset

# --- Drawing Constants & Helpers ---
LEFT_HAND_COLOR = (25, 50, 255)
RIGHT_HAND_COLOR = (25, 50, 255)


def project_points(points_3d: np.ndarray, P: np.ndarray):
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
    color = RIGHT_HAND_COLOR if is_right else LEFT_HAND_COLOR
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    for i, j in edges:
        u1, v1 = uvs[i]
        u2, v2 = uvs[j]
        if not np.all(np.isfinite([u1, v1, u2, v2])):
            continue
        p1, p2 = (int(round(u1)), int(round(v1))), (int(round(u2)), int(round(v2)))
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
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
            cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)
    return img


def visualize_episode_to_mp4(episode, output_path: str, downsample: int = 2, fps: float = 30.0, overlay_text: bool = False):
    """
    Renders an EgoEpisode object with 2D keypoint projections and saves it as an MP4 video.
    """
    if len(episode) == 0:
        print("Error: The provided episode is empty.")
        return

    # 1. Setup Dimensions
    first_frame = episode[0]
    h_f, w_f = first_frame.front_img.shape[:2]
    h_e, w_e = first_frame.eye_img.shape[:2]

    target_w = max(w_f // downsample, w_e // downsample)
    new_h_front = int((target_w / (w_f // downsample)) * (h_f // downsample))
    new_h_eye = int((target_w / (w_e // downsample)) * (h_e // downsample))
    target_h = new_h_front + new_h_eye

    # 2. Setup VideoWriter
    out_path = output_path if output_path.endswith(".mp4") else f"{output_path}.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

    # 3. Render Loop
    for i in tqdm(range(len(episode)), desc="Rendering Video"):
        # The SDK lazy-loads the images and keypoints here
        frame = episode[i]

        img_front = frame.front_img
        img_eye = frame.eye_img

        # Extract projection matrices from the stereo dict
        front_P1 = frame.stereo_params.get("front_P1", None)
        eye_P1 = frame.stereo_params.get("eye_P1", None)
        baseline = frame.stereo_params.get("front_baseline", None)

        # Draw Front
        if front_P1 is not None:
            l_kps2d = project_points(frame.left_kps_front, front_P1)
            img_front = draw_uv_skeleton(draw_uv_points(img_front, l_kps2d, False), l_kps2d, False)

            r_kps2d = project_points(frame.right_kps_front, front_P1)
            img_front = draw_uv_skeleton(draw_uv_points(img_front, r_kps2d, True), r_kps2d, True)

        # Draw Eye
        if eye_P1 is not None:
            l_kps2d = project_points(frame.left_kps_eye, eye_P1)
            img_eye = draw_uv_skeleton(draw_uv_points(img_eye, l_kps2d, False), l_kps2d, False)

            r_kps2d = project_points(frame.right_kps_eye, eye_P1)
            img_eye = draw_uv_skeleton(draw_uv_points(img_eye, r_kps2d, True), r_kps2d, True)

        # Resize and Stack
        img_front = cv2.resize(img_front, (target_w, new_h_front))
        img_eye = cv2.resize(img_eye, (target_w, new_h_eye))
        stacked = np.vstack((img_front, img_eye))

        # Text Overlay
        if overlay_text:
            text_y = 30
            cv2.putText(stacked, f"Time: {frame.timestamp:.3f}s", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if baseline is not None:
                cv2.putText(
                    stacked, f"Baseline: {baseline:.4f}", (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        writer.write(stacked)

    writer.release()
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=GROUNDED_DIR_DEFAULT)
    parser.add_argument("--episode_idx", type=int, default=0, help="Index of the demonstration to visualize")
    parser.add_argument("--profile", type=str, default="grounded", help="AWS profile to use for downloading")
    parser.add_argument("--output", type=str, default="sdk_visualization.mp4")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overlay_text", action="store_true")
    args = parser.parse_args()

    # Initialize SDK
    print(f"Loading dataset from {args.index_json}...")
    dataset = EgoDataset(index_path=args.index_json, aws_profile=args.profile, target_dir=args.dataset_dir)

    # Grab the specified episode
    episode = dataset[args.episode_idx]
    print(f"Loaded Episode {args.episode_idx} with {len(episode)} frames.")

    # Call the new visualization function
    visualize_episode_to_mp4(
        episode=episode, output_path=args.output, downsample=args.downsample, fps=args.fps, overlay_text=args.overlay_text
    )


if __name__ == "__main__":
    main()
