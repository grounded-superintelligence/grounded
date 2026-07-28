"""
Visualization utilities for rendering HandEpisode recordings to mp4.

Projects the unrectified left-front-frame keypoints into every active camera
(via each camera's pinhole intrinsics + extrinsics) and draws MANO-21
skeletons over the video streams, tiled into a grid.

Single entry point: visualize_hand_episode_to_mp4. Pass num_workers > 1 to
render in parallel: the episode is split into contiguous frame ranges, each
worker process renders its range to a temporary mp4 with its own decoders
(one seek to the chunk start, then pure sequential reads), and the chunks
are concatenated with ffmpeg's concat demuxer using stream copy — no
re-encode, and every chunk starts on a keyframe, so the joins are lossless.
"""

import os
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from tempfile import TemporaryDirectory
from typing import Callable, Optional, Sequence, Union

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from grounded.data.ego_dataset import HAND_CAMS, HandEpisode, HandPose

# The episode can be given as the object itself, a recording path, or a
# zero-arg callable that builds one (for constructors that need more than a
# path).
EpisodeSource = Union[HandEpisode, str, os.PathLike, Callable[[], HandEpisode]]

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


def format_caption_for_display(caption: Optional[str]) -> Optional[str]:
    """Capitalize the first letter for display without changing the source caption."""
    if not caption:
        return caption
    for index, character in enumerate(caption):
        if character.isalpha():
            return caption[:index] + character.upper() + caption[index + 1 :]
    return caption


def draw_uv_skeleton(image: np.ndarray, uvs: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Draws the MANO-21 bone edges.

    An edge is drawn only when BOTH endpoints are finite and lie inside the
    image; bones with an off-screen endpoint are skipped entirely rather than
    clipped at the border. Set inplace=True to skip the image copy.
    """
    if len(uvs) == 0:
        return image
    img = image if inplace else image.copy()
    h, w = img.shape[:2]

    for i, j in HAND_EDGES:
        if i >= len(uvs) or j >= len(uvs):
            continue
        u1, v1 = uvs[i]
        u2, v2 = uvs[j]
        if not np.all(np.isfinite([u1, v1, u2, v2])):
            continue
        p1 = (int(round(u1)), int(round(v1)))
        p2 = (int(round(u2)), int(round(v2)))
        if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
            continue
        cv2.line(img, p1, p2, JOINTS_COLOR, 3, cv2.LINE_AA)
    return img


def draw_uv_points(image: np.ndarray, uvs: np.ndarray, color: tuple, inplace: bool = False) -> np.ndarray:
    """Draws the joint dots; points outside the image are skipped.

    Set inplace=True to skip the image copy.
    """
    if len(uvs) == 0:
        return image
    img = image if inplace else image.copy()
    h, w = img.shape[:2]
    for u, v in uvs:
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        x, y = int(round(u)), int(round(v))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
    return img


def _draw_hand(img_bgr: np.ndarray, episode: HandEpisode, hand: Optional[HandPose], camera: str, base_color: tuple):
    """Draws one hand in place on img_bgr (a per-frame scratch buffer)."""
    if hand is None:
        return img_bgr
    T = episode.camera_params[f"T_{camera}_from_left_front"]
    pts_h = np.concatenate([hand.keypoints3d, np.ones((len(hand.keypoints3d), 1))], axis=-1)
    z_cam = (T @ pts_h.T).T[:, 2]
    behind = z_cam <= 1e-6
    if np.all(behind):
        return img_bgr

    uvs = episode.project_to_camera(hand.keypoints3d, camera)
    # a behind-camera point projects THROUGH the principal point and can land
    # inside the image at a plausible-looking position; mask such points so
    # they (and their bones) are skipped like any other invalid point
    uvs = np.where(behind[:, None], np.nan, uvs)

    color = base_color if hand.is_detected else INVALID_HAND_COLOR
    img_bgr = draw_uv_points(img_bgr, uvs, color, inplace=True)
    return draw_uv_skeleton(img_bgr, uvs, inplace=True)


def _append_caption_bar(img_bgr: np.ndarray, caption: str) -> np.ndarray:
    """Appends a black bar with the word-wrapped caption below the frame.

    The bar height depends only on the caption and frame width, so every frame
    of an episode keeps identical dimensions.
    """
    caption = format_caption_for_display(caption) or ""
    w = img_bgr.shape[1]
    scale = max(0.4, w / 1600.0)
    thickness = max(1, int(round(2 * scale)))
    pad = max(4, int(round(10 * scale)))

    lines, line = [], ""
    for word in caption.split():
        trial = f"{line} {word}".strip()
        text_w = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if line and text_w > w - 2 * pad:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)

    line_h = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][1] + pad
    bar = np.zeros((pad + line_h * len(lines), w, 3), dtype=img_bgr.dtype)
    for k, text in enumerate(lines):
        cv2.putText(
            bar,
            text,
            (pad, (k + 1) * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return np.vstack([img_bgr, bar])


def _render_frame(
    episode: HandEpisode,
    frame,
    cams: Sequence[str],
    grid_cols: int,
    downsample: int,
    show_invalid: bool,
    caption: Optional[str] = None,
) -> np.ndarray:
    """Renders one output frame (RGB, ready for the writer)."""
    rgb_by_cam = {
        "left_front": frame.left_front_rgb,
        "right_front": frame.right_front_rgb,
        "left_eye": frame.left_eye_rgb,
        "right_eye": frame.right_eye_rgb,
    }

    panels = []
    for cam in cams:
        img_bgr = cv2.cvtColor(rgb_by_cam[cam], cv2.COLOR_RGB2BGR)  # fresh buffer; safe to draw on
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

    if caption:
        stacked = _append_caption_bar(stacked, caption)

    return cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)


def _open_writer(output_path: str, fps: float, preset: str):
    kwargs = dict(codec="libx264", fps=fps)
    if preset:
        try:
            return imageio.get_writer(output_path, output_params=["-preset", preset], **kwargs)
        except TypeError:  # older imageio without output_params passthrough
            pass
    return imageio.get_writer(output_path, **kwargs)


def _render_range(
    episode: HandEpisode,
    start: int,
    stop: int,
    output_path: str,
    fps: float,
    downsample: int,
    grid_cols: int,
    show_invalid: bool,
    preset: str,
    progress: bool = True,
    caption: Optional[str] = None,
) -> int:
    """Renders frames [start, stop) to output_path. Returns frames written."""
    cams = [c for c in HAND_CAMS if c in episode.active_cameras]

    writer = None
    indices = range(start, stop)
    if progress:
        indices = tqdm(indices, desc="visualizing hands", leave=False)
    n_written = 0
    try:
        for i in indices:
            out_rgb = _render_frame(episode, episode[i], cams, grid_cols, downsample, show_invalid, caption)
            if writer is None:
                writer = _open_writer(output_path, fps, preset)
            writer.append_data(out_rgb)
            n_written += 1
    finally:
        if writer is not None:
            writer.close()
    return n_written


def _as_episode(source: EpisodeSource) -> HandEpisode:
    """Materializes an episode from any accepted source form."""
    if isinstance(source, HandEpisode):
        return source
    if isinstance(source, (str, os.PathLike)):
        return HandEpisode(os.fspath(source))
    if callable(source):
        return source()
    raise TypeError(
        "episode must be a HandEpisode, a recording path, or a zero-arg "
        f"callable that returns one; got {type(source).__name__}"
    )


def _worker_spec(source: EpisodeSource, episode: HandEpisode):
    """Builds the picklable recipe each worker uses to obtain its own episode."""
    if isinstance(source, (str, os.PathLike)):
        spec = ("path", os.fspath(source))
    elif isinstance(source, HandEpisode):
        try:
            spec = ("pickled", pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as err:
            raise TypeError(
                "num_workers > 1 needs to copy the episode into each worker "
                f"process, but this HandEpisode does not pickle ({err}). Pass "
                "the recording path instead — visualize_hand_episode_to_mp4("
                "path, output_path, num_workers=...) — and each worker will "
                "open its own copy."
            ) from err
    else:  # zero-arg callable, validated in _as_episode
        spec = ("loader", source)

    try:
        pickle.dumps(spec)  # catch unpicklable loaders (lambdas, closures) early
    except Exception as err:
        raise TypeError(
            "the episode loader must be picklable to reach worker processes: "
            "use a module-level function or functools.partial, not a lambda "
            f"or locally defined function ({err})"
        ) from err
    return spec


def _episode_from_spec(spec) -> HandEpisode:
    kind, payload = spec
    if kind == "path":
        return HandEpisode(payload)
    if kind == "pickled":
        return pickle.loads(payload)
    return payload()  # loader


def _render_chunk(
    spec,
    start: int,
    stop: int,
    output_path: str,
    fps: float,
    downsample: int,
    grid_cols: int,
    show_invalid: bool,
    preset: str,
    cv2_threads: int,
    caption: Optional[str] = None,
) -> str:
    """Worker: obtains its own episode and renders frames [start, stop)."""
    cv2.setNumThreads(cv2_threads)  # avoid oversubscribing cores across workers
    episode = _episode_from_spec(spec)
    _render_range(
        episode,
        start,
        stop,
        output_path,
        fps=fps,
        downsample=downsample,
        grid_cols=grid_cols,
        show_invalid=show_invalid,
        preset=preset,
        progress=False,
        caption=caption,
    )
    return output_path


def _concat_mp4_chunks(chunk_paths: Sequence[str], output_path: str) -> None:
    """Losslessly concatenates mp4 chunks (stream copy, no re-encode)."""
    from imageio_ffmpeg import get_ffmpeg_exe  # already a dep via the libx264 writer

    list_path = os.path.join(os.path.dirname(os.path.abspath(chunk_paths[0])), "concat_list.txt")
    with open(list_path, "w") as f:
        for p in chunk_paths:
            escaped = os.path.abspath(p).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        os.path.abspath(output_path),
    ]
    subprocess.run(cmd, check=True)


def _visualize_hand_episode_to_mp4(
    episode: EpisodeSource,
    output_path: str,
    downsample: int = 2,
    fps: Optional[float] = None,
    grid_cols: int = 2,
    show_invalid: bool = True,
    num_workers: int = 1,
    preset: str = "medium",
    caption: Optional[str] = None,
):
    """Renders the hand tracking of an entire episode over its video streams.

    Args:
        episode: episode to render. Accepts the HandEpisode itself, the
            recording path, or a zero-arg callable returning a HandEpisode
            (for constructors that need more than a path).
        output_path: output .mp4 path.
        downsample: spatial downsample factor applied to the tiled grid.
        fps: output framerate; defaults to the recording's framerate.
        grid_cols: number of cameras per grid row.
        show_invalid: draw cleaning-rejected fits in gray instead of skipping them.
        num_workers: 1 renders in-process, decoding frames sequentially so the
            full recording never seeks (original behavior). >1 splits the
            episode into contiguous chunks rendered by separate processes and
            losslessly concatenated; each worker seeks once to its chunk
            start, then reads sequentially. Passing the HandEpisode object
            with workers requires it to be picklable — if it is not, pass the
            recording path and each worker opens its own copy. Call from
            under `if __name__ == "__main__":` when using workers.
        preset: libx264 speed/quality preset; "veryfast" trades a little
            bitrate efficiency for a much faster encode.
        caption: optional text drawn in a bar below the camera grid on every
            frame (e.g. the episode's manifest caption).
    """
    episode_obj = _as_episode(episode)
    if len(episode_obj) == 0:
        print("Error: The provided episode is empty.")
        return
    if not episode_obj.active_cameras:
        print("Error: episode has no active_cameras; nothing to render over.")
        return

    fps = float(fps) if fps is not None else episode_obj.fps
    n = len(episode_obj)
    num_workers = max(1, min(int(num_workers), n))

    if num_workers == 1:
        written = _render_range(
            episode_obj,
            0,
            n,
            output_path,
            fps=fps,
            downsample=downsample,
            grid_cols=grid_cols,
            show_invalid=show_invalid,
            preset=preset,
            progress=True,
            caption=caption,
        )
        if written > 0:
            print(f"Saved to {output_path}")
        else:
            print("Error: No frames were written to the video.")
        return

    spec = _worker_spec(episode, episode_obj)
    del episode_obj  # parent holds no decoders while workers run

    bounds = np.linspace(0, n, num_workers + 1).astype(int)
    # split cv2's internal threading budget across workers
    cv2_threads = max(1, (os.cpu_count() or num_workers) // num_workers)

    with TemporaryDirectory(prefix="hand_viz_chunks_") as tmp_dir:
        chunk_paths = [os.path.join(tmp_dir, f"chunk_{k:04d}.mp4") for k in range(num_workers)]

        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _render_chunk,
                    spec,
                    int(bounds[k]),
                    int(bounds[k + 1]),
                    chunk_paths[k],
                    fps,
                    downsample,
                    grid_cols,
                    show_invalid,
                    preset,
                    cv2_threads,
                    caption,
                )
                for k in range(num_workers)
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="rendering chunks", leave=False):
                fut.result()  # re-raise worker exceptions immediately

        _concat_mp4_chunks(chunk_paths, output_path)

    print(f"Saved to {output_path}")


def visualize_hand_episode_to_mp4(episode: EpisodeSource, output_path: str, num_workers: int = 8, **kwargs):
    """Backward-compatible alias; prefer visualize_hand_episode_to_mp4(..., num_workers=N)."""
    return _visualize_hand_episode_to_mp4(episode, output_path, num_workers=num_workers, **kwargs)
