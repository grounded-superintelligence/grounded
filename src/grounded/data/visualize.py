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
    """
    Projects 3D points into 2D.
    """
    if points_3d is None or len(points_3d) == 0 or np.linalg.norm(points_3d) < 1e-2:
        return np.zeros((0, 2), dtype=int)

    ones = np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)
    points_3d_h = np.concatenate([points_3d, ones], axis=-1)

    # Project using camera intrinsics
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


def visualize_episode_to_mp4(episode: EgoEpisode, output_path: str, downsample: int = 2, fps: float = 30.0):
    """
    Renders an EgoEpisode object with canonical 2D keypoint projections and saves it.
    """
    if len(episode) == 0:
        print("Error: The provided episode is empty.")
        return

    out_path = output_path if output_path.endswith(".mp4") else f"{output_path}.mp4"
    writer = None
    target_w = None

    for i in tqdm(range(len(episode)), desc="Rendering Video"):
        frame = episode[i]
        params = episode.stereo_params

        # Define configurations for all possible 4 cameras
        # Note: P2 intrinsically handles the stereo baseline translation
        # relative to the left camera's canonical frame.
        cam_configs = [
            {
                "img": frame.left_front_img,
                "P": params["front_P1"],
                "transform": lambda pts: pts,  # Identity
            },
            {
                "img": frame.right_front_img,
                "P": params["front_P2"],
                "transform": lambda pts: pts,  # Identity
            },
        ]

        processed_imgs = []

        for config in cam_configs:
            img_rgb = config["img"]
            P = config["P"]

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Project Left Hand
            if frame.left_hand_3d is not None and len(frame.left_hand_3d) > 0:
                l_pts = config["transform"](frame.left_hand_3d)
                l_kps2d = project_points(l_pts, P)
                img_bgr = draw_uv_skeleton(draw_uv_points(img_bgr, l_kps2d, False), l_kps2d, False)

            # Project Right Hand
            if frame.right_hand_3d is not None and len(frame.right_hand_3d) > 0:
                r_pts = config["transform"](frame.right_hand_3d)
                r_kps2d = project_points(r_pts, P)
                img_bgr = draw_uv_skeleton(draw_uv_points(img_bgr, r_kps2d, True), r_kps2d, True)

            processed_imgs.append(img_bgr)

        if not processed_imgs:
            continue

        # Establish target width based on the first active image
        if target_w is None:
            max_w = max(img.shape[1] for img in processed_imgs)
            target_w = max_w // downsample

        # Resize all valid images to match width, preserving aspect ratio
        resized_imgs = []
        for img in processed_imgs:
            h, w = img.shape[:2]
            new_h = int(h * (target_w / w))
            resized = cv2.resize(img, (target_w, new_h))
            resized_imgs.append(resized)

        # Vertically stack all active cameras
        stacked = np.vstack(resized_imgs)

        # Initialize VideoWriter dynamically based on stacked frame size
        if writer is None:
            target_h = stacked.shape[0]
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

    # Updated to grab all 4 cameras if they exist
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
