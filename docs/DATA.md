# data

This document specifies the data exposed by the hand tracking readers in
`src/grounded/data/ego_dataset.py`. For installation and usage, see
`README.md` and `docs/ASSET_EPISODE_FLOW.md`.

## hardware configuration

All data is collected with the RoboCap: two hardware-synchronized stereo
pairs — forward-facing (`left_front`, `right_front`) and downward-facing
(`left_eye`, `right_eye`) — plus IMU. The published streams are undistorted
pinhole videos at 1920x1080; they are **not** stereo-rectified.

## data ontology

- `HandEpisode` iterates one full segment; `ClippedHandEpisode` iterates one
  published episode clip. Both yield a `HandFrameData` per frame.
- All 3D hand geometry is metric and expressed in the **unrectified
  `left_front` camera frame** (x right, y down, z forward).

### `HandFrameData`

| Field | Type | Description |
| --- | --- | --- |
| `frame_idx` | `int` | Frame index into the recording (source-frame index for full segments, clip-local for episodes). |
| `{left,right}_{front,eye}_rgb` | `(1080, 1920, 3) uint8` | RGB-ordered undistorted pinhole image; `None` for cameras not in `active_cameras`. |
| `left_hand`, `right_hand` | `HandPose` | MANO fit for that hand, or `None` when the tracker produced none (or, with `valid_only=True`, when cleaning rejected it). |

### `HandPose`

All arrays are `float32`, in the unrectified `left_front` frame.

| Field | Shape | Description |
| --- | --- | --- |
| `keypoints3d` | `(21, 3)` | MANO-21 joints, meters; `keypoints3d[0]` is the wrist and equals `transl`. |
| `vertices` | `(778, 3)` | MANO mesh vertices, meters. |
| `global_orient` | `(3, 3)` | Root rotation. |
| `transl` | `(3,)` | Root translation (wrist). |
| `hand_pose` | `(15, 3, 3)` | Articulated joint rotations. |
| `betas` | `(10,)` | Per-frame fitted MANO shape. |
| `source_view` | `str` | Camera the fit was primarily sourced from. |
| `inlier_mask` | `(4, 21) bool` | Per-view keypoint inliers; view axis follows `HAND_CAMS`. |
| `is_detected` | `bool` | `False` when pose cleaning rejected this fit. |
| `reason` | `str` | Cleaning rejection reason, `""` when valid. |
| `hand_frame_idx` | `int` | Frame the fit was sourced from (differs from `frame_idx` only when borrowed). |

### `episode.camera_params`

Loaded once per episode from `camera_params.npz`. Per camera: `K_{cam}`
`(3, 3)` pinhole intrinsics, `T_{cam}_from_left_front` `(4, 4)` extrinsics,
and `res_{cam}` `(2,)` width/height. Use
`episode.project_to_camera(points, cam)` to project unrectified-left-front
points into any stream. The producer's stereo-rectification arrays
(`{front,eye}_{P1,P2,R1,R2,Q}`, baselines, `T_front_to_eye`) are also
present but are sized for the producer's rectified canvas, not these streams.

## SLAM lane

`trajectory.txt` is TUM format (`timestamp tx ty tz qx qy qz qw`): per-capture
**camera-to-world** poses of the unrectified `left_front` camera, timestamped
in sensor seconds. The world frame is **gravity-aligned with +z up**. The hand
lane's `timebase.json` maps each frame to its sensor timestamp
(`captures[i].source_frame`, `captures[i].t` in ns) for matching frames to
poses.

## `episode.caption`

Optional natural-language description of the episode's action. Attached from
the episode manifest record when opening via `ProcessingClient.open_hand`, or
from the captions JSONL next to a `HandManifest`; `None` when absent.
