# ego dataset

This document specifies the contents of every numpy array exposed by the dataset. For installation and usage, see `README.md`.

## hardware configuration

Currently, all data is collected with the RoboCap. We fuse stereo views from the front cameras (i.e. `left-front`, `right-front`), eye cameras (i.e. `left-eye`, `right-eye`), and IMU to provide dataset labels. All sensor streams are hardware-synchronized and aligned atomically per frame.

## data ontology

The dataset is currently represented at 3 levels of abstraction:
- `EgoDataset` contains the metadata and entire dataset of discrete episodes
- `EgoEpisode` is a PyTorch dataset that iterates over episode frames and metadata (i.e. stereo rectification parameters)
- `Framedata` is a Python dataclass with fields that index each type of sensor data and label (i.e. RGB, depth, hand pose, head pose)

### `FrameData`

Each `frame = episode[i]` is a `FrameData` dataclass with the following fields.

| Field               | Shape              | Dtype     | Units         | Reference Frame                 | Description |
|---------------------|--------------------|-----------|---------------|---------------------------------|-------------|
| `timestamp_ns`      | scalar             | `int`     | ns            | device clock                    | Capture time of this frame. |
| `left_front_rgb`    | `(1080, 1920, 3)`  | `uint8`   | —             | rectified `left-front` (image)    | RGB-ordered, stereo-rectified. `None` if `left-front` not in `active_cameras`. |
| `right_front_rgb`   | `(1080, 1920, 3)`  | `uint8`   | —             | rectified `right-front` (image)   | RGB-ordered, stereo-rectified. |
| `left_eye_rgb`      | `(1080, 1920, 3)`  | `uint8`   | —             | rectified `left-eye` (image)      | RGB-ordered, stereo-rectified. |
| `right_eye_rgb`     | `(1080, 1920, 3)`  | `uint8`   | —             | rectified `right-eye` (image)     | RGB-ordered, stereo-rectified. |
| `left_front_depth`  | `(216, 384)`       | `float32` | meters        | rectified `left-front`            | Downsampled depth map |
| `left_eye_depth`    | `(216, 384)`       | `float32` | meters        | rectified `left-eye`              | Downsampled depth map |
| `left_hand_kp`      | `(21, 3)`          | `float32` | meters        | rectified `left-front`            | MANO-21 3D keypoints of left hand, or `None` if not recoverable for this frame. |
| `right_hand_kp`     | `(21, 3)`          | `float32` | meters        | rectified `left-front`            | MANO-21 3D keypoints of right hand, or `None` if not recoverable. |
| `c2w`               | `(7,)`             | `float64` | m / quat      | rectified `left-front`    | SLAM pose as `[tx, ty, tz, qx, qy, qz, qw]`. Maps the rectified `left-front` frame into the world frame. World = rectified `left-front` at SLAM init. |

## `episode.stereo_params`

A dict of numpy arrays loaded once per episode from `stereo_params.npz`. The four cameras form two stereo pairs: forward-facing (`front`) and downward-facing (`eye`).

Calibration was performed in the **unrectified** camera, while the image streams and depth maps are stored after rectification.

| Key               | Shape    | Geometric meaning |
|-------------------|----------|-------------------|
| `front_P1`        | `(3, 4)` | Projection matrix of the rectified `left-front` camera |
| `front_P2`        | `(3, 4)` | Projection matrix of the rectified `right-front` camera, expressed in the rectified `left-front` frame. Has the form `[K \| −K·b]` with `b = [baseline, 0, 0]ᵀ` |
| `front_R1`        | `(3, 3)` | Rotation matrix that rectifies the `left-front` camera |
| `front_R2`        | `(3, 3)` | Rotation matrix that rectifies the `right-front` camera |
| `front_Q`         | `(4, 4)` | Disparity-to-depth reprojection matrix of the rectified`left-front` camera that maps: pixel + disparity ➝ 3D point |
| `front_baseline`  | `scalar` | Stereo baseline of the front pair, equivalent to `-P2[0,3] / P2[0,0]` |
| `eye_P1`          | `(3, 4)` | Projection matrix of that maps a 3D point in the rectified `left-eye` frame to homogeneous pixel coordinates |
| `eye_P2`          | `(3, 4)` | Projection matrix of the rectified `right-eye` camera, expressed in the rectified `left-eye` camera. Has the form `[K \| −K·b]` with `b = [baseline, 0, 0]ᵀ` and `baseline = -P2[0,3] / P2[0,0]` |
| `eye_R1`          | `(3, 3)` | Rotation matrix that rectifies the `left-eye` camera |
| `eye_R2`          | `(3, 3)` | Rotation matrix that rectifies the `right-eye` camera |
| `eye_Q`           | `(4, 4)` | Disparity-to-depth reprojection matrix of the rectified`left-eye` camera that maps: pixel + disparity ➝ 3D point |
| `eye_baseline`    | `scalar` | Stereo baseline of the eye pair, equivalent to `-P2[0,3] / P2[0,0]` |
| `T_front_to_eye`  | `(4, 4)` | Extrinsic calibration matrix that maps unrectified `left-front` to unrectified `left-eye` |
