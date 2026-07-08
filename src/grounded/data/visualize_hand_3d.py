"""
Rerun (3D) visualization for HandEpisode recordings.

Logs MANO-21 skeletons and (optionally) the full 778-vertex MANO meshes as
point clouds in the unrectified left-front camera frame, alongside the
left-front video stream behind a pinhole camera.
"""

import os

import numpy as np
import rerun as rr
from tqdm import tqdm

from grounded.data.hand_dataset import HandEpisode, HandPose
from grounded.data.visualize_hand import HAND_EDGES, LEFT_HAND_COLOR, RIGHT_HAND_COLOR

INVALID_ALPHA_SCALE = 0.35  # dim cleaning-rejected fits


def log_hand_to_rerun(entity_path: str, hand: HandPose, color_bgr: tuple, log_vertices: bool = True):
    """Plots hand joints, bone segments and mesh vertices, or clears them if the hand is missing."""
    if hand is None or hand.keypoints3d is None or len(hand.keypoints3d) == 0:
        rr.log(entity_path, rr.Clear(recursive=True))
        return

    # visualize_hand.py colors are bgr, rerun expects rgb
    color_rgb = [color_bgr[2], color_bgr[1], color_bgr[0], 255]
    if not hand.is_detected:
        color_rgb = [int(c * INVALID_ALPHA_SCALE) for c in color_rgb[:3]] + [255]

    keypoints = hand.keypoints3d

    # joints
    rr.log(f"{entity_path}/joints", rr.Points3D(keypoints, colors=[color_rgb] * len(keypoints), radii=0.005))

    # bones (line strips)
    strips = []
    for i, j in HAND_EDGES:
        if i < len(keypoints) and j < len(keypoints):
            strips.append([keypoints[i], keypoints[j]])
    if strips:
        rr.log(f"{entity_path}/bones", rr.LineStrips3D(strips, colors=[color_rgb] * len(strips)))

    # MANO mesh vertices
    if log_vertices and hand.vertices is not None:
        rr.log(
            f"{entity_path}/vertices",
            rr.Points3D(hand.vertices, colors=[color_rgb] * len(hand.vertices), radii=0.0015),
        )
    else:
        rr.log(f"{entity_path}/vertices", rr.Clear(recursive=False))


def visualize_hand_episode_to_rerun(
    episode: HandEpisode,
    output_path: str,
    fps_downsample: int = 3,
    log_vertices: bool = True,
    log_image: bool = True,
    image_plane_distance: float = 0.05,
    image_opacity: float = 1.0,
):
    """
    Logs the hand tracking of an episode to a Rerun file in the unrectified
    left-front camera frame.

    Args:
        image_plane_distance: distance (meters) from the camera origin at which
            the viewer draws the video image plane. Purely cosmetic - it does
            not affect the projection - but it must be set explicitly: when
            absent, the viewer falls back to `0.02 x scene-bbox diagonal`,
            re-evaluated as the (smoothed) scene bounds change, so the image
            grows and shrinks during playback. The lens is very wide
            (~127 deg horizontal), so the drawn quad is ~4x the distance in
            width; the 0.1 m default keeps it a compact viewfinder between
            the camera origin and the hand content (valid fits stay beyond
            ~0.10 m in practice), with the skeletons floating past it.
        image_opacity: opacity of the drawn image in [0, 1]. Values < 1 let
            geometry behind the image plane show through, useful when viewing
            the scene from behind the camera with a near image plane.
    """
    if len(episode) == 0:
        print("Error: The provided episode is empty.")
        return
    log_image = log_image and "left_front" in episode.active_cameras

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rr.init("HandEpisode 3D Vis", spawn=False)
    rr.save(output_path)

    # coordinate system is right/down/forward (camera frame)
    rr.log("camera", rr.ViewCoordinates.RDF, static=True)
    if log_image:
        K = episode.camera_params["K_left_front"]
        W, H = (int(v) for v in episode.camera_params["res_left_front"])
        rr.log(
            "camera/image",
            rr.Pinhole(
                image_from_camera=K.astype(np.float32),
                width=W,
                height=H,
                image_plane_distance=image_plane_distance,
            ),
            static=True,
        )
        if image_opacity < 1.0:
            # partial update: opacity applies to every per-frame image below
            rr.log("camera/image", rr.EncodedImage.from_fields(opacity=image_opacity), static=True)

    fps = episode.fps
    for i in tqdm(range(len(episode)), desc="visualizing hands 3d", leave=False):
        if i % fps_downsample > 0:
            continue

        frame = episode[i]

        rr.set_time("frame_idx", sequence=frame.frame_idx)
        rr.set_time("time", duration=frame.frame_idx / fps)

        if log_image and frame.left_front_rgb is not None:
            rr.log("camera/image", rr.Image(frame.left_front_rgb).compress(jpeg_quality=85))

        log_hand_to_rerun("camera/left_hand", frame.left_hand, LEFT_HAND_COLOR, log_vertices)
        log_hand_to_rerun("camera/right_hand", frame.right_hand, RIGHT_HAND_COLOR, log_vertices)

    print(f"Saved Rerun visualizer file to: {output_path}")
