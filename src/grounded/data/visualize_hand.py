"""
Visualization script for rendering HandEpisode recordings to mp4.

Projects the unrectified left-front-frame keypoints into every active camera
(via each camera's pinhole intrinsics + extrinsics) and draws MANO-21
skeletons over the video streams, tiled into a grid.
"""

from typing import Optional

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from grounded.data.hand_dataset import HAND_CAMS, HandEpisode, HandPose

LEFT_HAND_COLOR = (165, 255, 100)  # BGR
RIGHT_HAND_COLOR = (255, 100, 200)  # BGR
INVALID_HAND_COLOR = (128, 128, 128)  # BGR, used for cleaning-rejected fits
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


def draw_uv_skeleton(image: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    if len(uvs) == 0:
        return image
    img = image.copy()

    CLAMP = 1 << 15

    for i, j in HAND_EDGES:
        if i >= len(uvs) or j >= len(uvs):
            continue
        u1, v1 = uvs[i]
        u2, v2 = uvs[j]
        if not np.all(np.isfinite([u1, v1, u2, v2])):
            continue
        u1, v1, u2, v2 = np.clip([u1, v1, u2, v2], -CLAMP, CLAMP)
        p1, p2 = (int(round(u1)), int(round(v1))), (int(round(u2)), int(round(v2)))
        cv2.line(img, p1, p2, JOINTS_COLOR, 3, cv2.LINE_AA)
    return img


def draw_uv_points(image: np.ndarray, uvs: np.ndarray, color: tuple) -> np.ndarray:
    if len(uvs) == 0:
        return image
    img = image.copy()
    h, w = img.shape[:2]
    for u, v in uvs:
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        x, y = int(round(u)), int(round(v))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
    return img


def _draw_hand(img_bgr: np.ndarray, episode: HandEpisode, hand: Optional[HandPose], camera: str, base_color: tuple):
    if hand is None:
        return img_bgr
    # points behind the camera plane project to garbage; skip the hand entirely
    T = episode.camera_params[f"T_{camera}_from_left_front"]
    pts_h = np.concatenate([hand.keypoints3d, np.ones((len(hand.keypoints3d), 1))], axis=-1)
    z_cam = (T @ pts_h.T).T[:, 2]
    if np.all(z_cam <= 1e-6):
        return img_bgr
    uvs = episode.project_to_camera(hand.keypoints3d, camera)
    color = base_color if hand.is_detected else INVALID_HAND_COLOR
    return draw_uv_skeleton(draw_uv_points(img_bgr, uvs, color), uvs)


def visualize_hand_episode_to_mp4(
    episode: HandEpisode,
    output_path: str,
    downsample: int = 2,
    fps: Optional[float] = None,
    grid_cols: int = 2,
    show_invalid: bool = True,
):
    """Renders the hand tracking of an entire episode over its video streams.

    Args:
        episode: episode to render; frames are decoded sequentially from the
            per-camera mp4s, so rendering the full recording never seeks.
        output_path: output .mp4 path.
        downsample: spatial downsample factor applied to the tiled grid.
        fps: output framerate; defaults to the recording's framerate.
        grid_cols: number of cameras per grid row.
        show_invalid: draw cleaning-rejected fits in gray instead of skipping them.
    """
    if len(episode) == 0:
        print("Error: The provided episode is empty.")
        return
    if not episode.active_cameras:
        print("Error: episode has no active_cameras; nothing to render over.")
        return

    fps = float(fps) if fps is not None else episode.fps
    cams = [c for c in HAND_CAMS if c in episode.active_cameras]

    writer = None
    for i in tqdm(range(len(episode)), desc="visualizing hands", leave=False):
        frame = episode[i]
        rgb_by_cam = {
            "left_front": frame.left_front_rgb,
            "right_front": frame.right_front_rgb,
            "left_eye": frame.left_eye_rgb,
            "right_eye": frame.right_eye_rgb,
        }

        panels = []
        for cam in cams:
            img_bgr = cv2.cvtColor(rgb_by_cam[cam], cv2.COLOR_RGB2BGR)
            for hand, color in ((frame.left_hand, LEFT_HAND_COLOR), (frame.right_hand, RIGHT_HAND_COLOR)):
                if hand is not None and not hand.is_detected and not show_invalid:
                    continue
                img_bgr = _draw_hand(img_bgr, episode, hand, cam, color)
            panels.append(img_bgr)

        # tile: rows of `grid_cols` cameras, black-padded to a full rectangle
        rows = []
        for r in range(0, len(panels), grid_cols):
            row_panels = panels[r : r + grid_cols]
            while len(row_panels) < min(grid_cols, len(panels)):
                row_panels.append(np.zeros_like(panels[0]))
            rows.append(np.hstack(row_panels))
        stacked = np.vstack(rows)

        if downsample > 1:
            h, w = stacked.shape[:2]
            new_w = w // downsample
            new_h = int(h * (new_w / w))
            stacked = cv2.resize(stacked, (new_w, new_h))

        if writer is None:
            writer = imageio.get_writer(output_path, codec="libx264", fps=fps)
        writer.append_data(cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))

    if writer is not None:
        writer.close()
        print(f"Saved to {output_path}")
    else:
        print("Error: No frames were written to the video.")
