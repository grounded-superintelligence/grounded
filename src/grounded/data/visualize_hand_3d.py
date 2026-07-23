"""
Rerun (3D) visualization for HandEpisode recordings.

Logs MANO-21 skeletons and (optionally) the full 778-vertex MANO meshes as
point clouds, alongside the left-front video stream behind a pinhole camera.

When a SLAM trajectory is provided, the whole scene is expressed in the world
frame: the camera entity carries a per-frame pose from the trajectory (so the
video plane travels through the room) and the hands are placed at their world
positions. The trajectory poses map the unrectified left-front camera frame -
the frame all hand geometry lives in - into a gravity-aligned world frame
with +z up. Without a trajectory, everything stays in the camera frame as
before.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from grounded.data.ego_dataset import HandEpisode, HandPose
from grounded.data.visualize_hand import HAND_EDGES, LEFT_HAND_COLOR, RIGHT_HAND_COLOR

INVALID_ALPHA_SCALE = 0.35  # dim cleaning-rejected fits
TRAJECTORY_COLOR = [160, 160, 160, 255]
POSE_MATCH_WARN_GAP_S = 0.05  # frame-to-pose timestamp gaps beyond this suggest missing SLAM coverage


def load_tum_trajectory(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads a TUM-format trajectory (``# timestamp tx ty tz qx qy qz qw``).

    Returns ``(times_ns, positions, rotations)`` sorted by time, with
    ``rotations`` as (N, 3, 3) matrices. For the SLAM episode lane these are
    camera-to-world poses of the unrectified left-front camera.
    """
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != 8:
        raise ValueError(f"Expected 8 TUM columns (t tx ty tz qx qy qz qw) in {path}, got {data.shape[1]}")
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    times_ns = np.round(data[:, 0] * 1e9).astype(np.int64)
    positions = data[:, 1:4]
    rotations = Rotation.from_quat(data[:, 4:8]).as_matrix()
    return times_ns, positions, rotations


def _frame_times_ns(episode: HandEpisode) -> Optional[dict]:
    """Maps source frame index -> sensor timestamp (ns) from the hand lane's timebase.json.

    Only clipped episode lanes ship a timebase excerpt; full-segment assets
    return None (their visualization stays in the camera frame).
    """
    lane_dir = getattr(episode.path_manager, "lane_dir", None)
    if not lane_dir:
        return None
    timebase_path = Path(lane_dir) / "timebase.json"
    if not timebase_path.is_file():
        return None
    try:
        captures = json.loads(timebase_path.read_text()).get("captures")
        times = {int(capture["source_frame"]): int(capture["t"]) for capture in captures}
    except (ValueError, TypeError, KeyError, OSError):
        return None
    return times or None


def _nearest_pose_index(times_ns: np.ndarray, t_ns: int) -> int:
    j = int(np.searchsorted(times_ns, t_ns))
    if j <= 0:
        return 0
    if j >= len(times_ns):
        return len(times_ns) - 1
    return j if times_ns[j] - t_ns < t_ns - times_ns[j - 1] else j - 1


def log_hand_to_rerun(
    entity_path: str,
    hand: HandPose,
    color_bgr: tuple,
    log_vertices: bool = True,
    world_from_camera: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """Plots hand joints, bone segments and mesh vertices, or clears them if the hand is missing.

    ``world_from_camera`` is an optional ``(R, t)`` applied to the camera-frame
    geometry before logging, placing the hand in the world frame.
    """
    if hand is None or hand.keypoints3d is None or len(hand.keypoints3d) == 0:
        rr.log(entity_path, rr.Clear(recursive=True))
        return

    # visualize_hand.py colors are bgr, rerun expects rgb
    color_rgb = [color_bgr[2], color_bgr[1], color_bgr[0], 255]
    if not hand.is_detected:
        color_rgb = [int(c * INVALID_ALPHA_SCALE) for c in color_rgb[:3]] + [255]

    keypoints = hand.keypoints3d
    vertices = hand.vertices
    if world_from_camera is not None:
        rotation, translation = world_from_camera
        keypoints = keypoints @ rotation.T + translation
        if vertices is not None:
            vertices = vertices @ rotation.T + translation

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
    if log_vertices and vertices is not None:
        rr.log(
            f"{entity_path}/vertices",
            rr.Points3D(vertices, colors=[color_rgb] * len(vertices), radii=0.0015),
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
    trajectory_path: Optional[str] = None,
):
    """
    Logs the hand tracking of an episode to a Rerun file.

    With ``trajectory_path`` the scene is world-frame: the camera pinhole and
    video plane follow the SLAM poses, the hands are logged at their world
    positions, and the full camera path is drawn as a static line. Frames are
    matched to poses by nearest sensor timestamp via the hand lane's
    timebase.json; if that sidecar is unavailable the visualization falls back
    to the unrectified left-front camera frame (the frame all hand geometry is
    stored in), which is also the behavior when no trajectory is given.

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
        trajectory_path: optional TUM trajectory (``timestamp tx ty tz qx qy
            qz qw``) of camera-to-world poses for the unrectified left-front
            camera, e.g. the SLAM episode lane's ``trajectory.txt``.
    """
    if len(episode) == 0:
        print("Error: The provided episode is empty.")
        return
    log_image = log_image and "left_front" in episode.active_cameras

    pose_times_ns = positions = rotations = None
    frame_times: Optional[dict] = None
    if trajectory_path is not None:
        pose_times_ns, positions, rotations = load_tum_trajectory(trajectory_path)
        frame_times = _frame_times_ns(episode)
        if frame_times is None:
            print("Warning: no timebase.json next to the hand lane; rendering in camera frame instead of world.")
    world_mode = frame_times is not None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rr.init("HandEpisode 3D Vis", spawn=False)
    rr.save(output_path)

    root = "world" if world_mode else "camera"
    camera_entity = "world/camera" if world_mode else "camera"

    # the SLAM world is gravity-aligned with +z up (the camera's mean "down"
    # axis over a trajectory points along -z), so declare a z-up convention
    # there; the camera-frame fallback keeps the camera's right/down/forward
    rr.log(root, rr.ViewCoordinates.FLU if world_mode else rr.ViewCoordinates.RDF, static=True)
    if world_mode:
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([positions], colors=[TRAJECTORY_COLOR], radii=0.002),
            static=True,
        )
    if log_image:
        K = episode.camera_params["K_left_front"]
        W, H = (int(v) for v in episode.camera_params["res_left_front"])
        rr.log(
            f"{camera_entity}/image",
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
            rr.log(f"{camera_entity}/image", rr.EncodedImage.from_fields(opacity=image_opacity), static=True)

    fps = episode.fps
    worst_gap_ns = 0
    world_from_camera = None
    for i in tqdm(range(len(episode)), desc="visualizing hands 3d", leave=False):
        if i % fps_downsample > 0:
            continue

        frame = episode[i]

        rr.set_time("frame_idx", sequence=frame.frame_idx)
        rr.set_time("time", duration=frame.frame_idx / fps)

        if world_mode:
            source_frame = getattr(episode, "source_frame_start", 0) + frame.frame_idx
            t_ns = frame_times.get(source_frame)
            if t_ns is not None:
                j = _nearest_pose_index(pose_times_ns, t_ns)
                worst_gap_ns = max(worst_gap_ns, abs(int(pose_times_ns[j]) - t_ns))
                world_from_camera = (rotations[j], positions[j])
                rr.log(camera_entity, rr.Transform3D(translation=positions[j], mat3x3=rotations[j]))

        if log_image and frame.left_front_rgb is not None:
            rr.log(f"{camera_entity}/image", rr.Image(frame.left_front_rgb).compress(jpeg_quality=85))

        # in world mode, a frame before the first matched pose has nowhere to
        # place camera-frame geometry; clear the hands instead of logging them
        # at the world origin
        left_hand = frame.left_hand if not world_mode or world_from_camera else None
        right_hand = frame.right_hand if not world_mode or world_from_camera else None
        log_hand_to_rerun(f"{root}/left_hand", left_hand, LEFT_HAND_COLOR, log_vertices, world_from_camera)
        log_hand_to_rerun(f"{root}/right_hand", right_hand, RIGHT_HAND_COLOR, log_vertices, world_from_camera)

    if world_mode and worst_gap_ns > POSE_MATCH_WARN_GAP_S * 1e9:
        print(
            f"Warning: worst frame-to-pose timestamp gap is {worst_gap_ns / 1e6:.0f} ms; "
            "the trajectory may not fully cover this episode."
        )
    print(f"Saved Rerun visualizer file to: {output_path}")
