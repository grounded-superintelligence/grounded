from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from grounded.data.hand_dataset import ClippedHandEpisode, HandEpisode
from grounded.processing import (
    AssetDownload,
    DownloadedAssetFile,
    DownloadedEpisodeFile,
    EpisodeDownload,
    EpisodeLaneDownload,
)

EPISODE_ID = "ep_v1_test_clipped_hand"
ASSET_ID = "ast_v1_test_clipped_hand"
START_NS = 10_000_000_000
END_NS = 10_066_666_666
SOURCE_FRAME_START = 100
SOURCE_FRAME_END = 102


def _write_pose(path: Path, source_frame: int) -> None:
    np.savez(
        path,
        sides=np.asarray(["left"]),
        keypoints3d=np.full((1, 21, 3), source_frame, dtype=np.float32),
        vertices=np.full((1, 778, 3), source_frame, dtype=np.float32),
        global_orients=np.eye(3, dtype=np.float32)[None, ...],
        transls=np.asarray([[source_frame, 0, 1]], dtype=np.float32),
        hand_poses=np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], 15, axis=1),
        betas=np.zeros((1, 10), dtype=np.float32),
        source_views=np.asarray(["left_front"]),
        inlier_masks=np.ones((1, 4, 21), dtype=bool),
        is_detected=np.asarray([True]),
        reasons=np.asarray([""]),
        hand_frame_idxs=np.asarray([source_frame]),
    )


def _build_flat_hand_lane(tmp_path: Path, *, malicious_member: bool = False) -> Path:
    lane_dir = tmp_path / "episodes" / EPISODE_ID / "hand"
    lane_dir.mkdir(parents=True)

    camera_values = {}
    for camera in ("left_front", "right_front", "left_eye", "right_eye"):
        camera_values[f"K_{camera}"] = np.eye(3, dtype=np.float32)
        camera_values[f"T_{camera}_from_left_front"] = np.eye(4, dtype=np.float32)
        camera_values[f"res_{camera}"] = np.asarray([8, 8])
    np.savez(lane_dir / "camera_params.npz", **camera_values)
    (lane_dir / "left_front.mp4").write_bytes(b"test video bytes; decoder is replaced in the unit test")
    (lane_dir / "source_yield.json").write_text(json.dumps({"total_frames": 900}))
    (lane_dir / "source_continuous_intervals.json").write_text(
        json.dumps(
            {
                "left_intervals": [{"start": 80, "end": 140}],
                "both_intervals": [{"start": 99, "end": 100}],
            }
        )
    )

    pose_sources = tmp_path / "pose_sources"
    pose_sources.mkdir()
    with tarfile.open(lane_dir / "pose_frames.tar", mode="w") as archive:
        for source_frame in range(SOURCE_FRAME_START, SOURCE_FRAME_END):
            pose_path = pose_sources / f"frame_{source_frame:06d}.npz"
            _write_pose(pose_path, source_frame)
            archive.add(
                pose_path,
                arcname=f"hand/pose_interpolation/params/{pose_path.name}",
                recursive=False,
            )
        if malicious_member:
            payload = b"must not escape"
            member = tarfile.TarInfo("../../escape.npz")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    clip_manifest = {
        "schema_version": "grounded.episode.hand_clip.v1alpha1",
        "episode_id": EPISODE_ID,
        "asset_id": ASSET_ID,
        "interval": {
            "frame_start": SOURCE_FRAME_START,
            "frame_end": SOURCE_FRAME_END,
            "start_ns": START_NS,
            "end_ns": END_NS,
        },
        "videos": {
            "left_front.mp4": {
                "source_frame_count": 900,
                "output_frame_count": SOURCE_FRAME_END - SOURCE_FRAME_START,
            }
        },
        "poses": {
            "source_frame_start": SOURCE_FRAME_START,
            "source_frame_end": SOURCE_FRAME_END,
            "episode_local_frame_start": 0,
            "episode_local_frame_end": SOURCE_FRAME_END - SOURCE_FRAME_START,
            "pose_file_count": SOURCE_FRAME_END - SOURCE_FRAME_START,
            "pose_file_names_preserve_source_frame_indices": True,
        },
        "source_sidecars": ["camera_params.npz", "source_yield.json", "source_continuous_intervals.json"],
        "missing": [],
    }
    (lane_dir / "clip_manifest.json").write_text(json.dumps(clip_manifest))
    return lane_dir


def _build_full_hand_archive(tmp_path: Path, *, malicious_member: bool = False) -> Path:
    hand_dir = tmp_path / "full_hand_source"
    params_dir = hand_dir / "pose_interpolation" / "params"
    save_dataset_dir = hand_dir / "save_dataset"
    params_dir.mkdir(parents=True)
    save_dataset_dir.mkdir(parents=True)

    for frame_index in range(2):
        _write_pose(params_dir / f"frame_{frame_index:06d}.npz", frame_index)
    camera_values = {}
    for camera in ("left_front", "right_front", "left_eye", "right_eye"):
        camera_values[f"K_{camera}"] = np.eye(3, dtype=np.float32)
        camera_values[f"T_{camera}_from_left_front"] = np.eye(4, dtype=np.float32)
        camera_values[f"res_{camera}"] = np.asarray([8, 8])
        (save_dataset_dir / f"{camera}.mp4").write_bytes(b"test video bytes")
    np.savez(save_dataset_dir / "camera_params.npz", **camera_values)
    (save_dataset_dir / "yield.json").write_text(json.dumps({"total_frames": 2}))
    (save_dataset_dir / "continuous_intervals.json").write_text(
        json.dumps({"both_intervals": [{"start": 0, "end": 1}]})
    )

    archive_path = tmp_path / "hand_v2_outputs.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        for source_path in sorted(path for path in hand_dir.rglob("*") if path.is_file()):
            archive.add(
                source_path,
                arcname=f"hand/{source_path.relative_to(hand_dir).as_posix()}",
                recursive=False,
            )
        if malicious_member:
            payload = b"must not escape"
            member = tarfile.TarInfo("../../escape.txt")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return archive_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_download(
    lane_dir: Path,
    *,
    status: str = "available",
    episode_id: str = EPISODE_ID,
    exclude_names: set[str] | None = None,
) -> EpisodeDownload:
    excluded = exclude_names or set()
    files = tuple(
        DownloadedEpisodeFile(
            episode_id=episode_id,
            asset_id=ASSET_ID,
            lane="hand",
            source_uri=f"s3://episodes/{episode_id}/hand/{path.name}",
            local_path=str(path),
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
            size_verified=True,
            sha256_verified=True,
        )
        for path in sorted(lane_dir.iterdir())
        if path.is_file() and path.name not in excluded
    )
    lane = EpisodeLaneDownload(
        lane="hand",
        status=status,
        files=files,
        run_id="hand-run",
        job_id="hand-job",
        message="" if status == "available" else f"hand lane {status}",
    )
    return EpisodeDownload(
        episode_id=episode_id,
        asset_id=ASSET_ID,
        start_ns=START_NS,
        end_ns=END_NS,
        root_dir=str(lane_dir.parent),
        lanes=(lane,),
        files=files,
    )


def _asset_download(archive_path: Path, root_dir: Path) -> AssetDownload:
    downloaded_file = DownloadedAssetFile(
        asset_id=ASSET_ID,
        lane="hand",
        run_id="hand-run",
        source_uri=f"s3://assets/{ASSET_ID}/hand/{archive_path.name}",
        local_path=str(archive_path),
        size_bytes=archive_path.stat().st_size,
        sha256=_sha256(archive_path),
        size_verified=True,
        sha256_verified=True,
    )
    return AssetDownload(
        asset_id=ASSET_ID,
        root_dir=str(root_dir),
        files=(downloaded_file,),
    )


class _FakeVideoReader:
    fps = 30.0
    num_frames = SOURCE_FRAME_END - SOURCE_FRAME_START

    def __init__(self) -> None:
        self.read_indices: list[int] = []

    def read_rgb(self, frame_idx: int) -> np.ndarray:
        self.read_indices.append(frame_idx)
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def test_from_download_maps_local_video_frames_to_source_pose_indices(tmp_path: Path) -> None:
    lane_dir = _build_flat_hand_lane(tmp_path)
    download = _episode_download(lane_dir)

    episode = ClippedHandEpisode.from_download(
        download,
        active_cameras=["left_front"],
        caption="wipe the work surface",
    )

    assert isinstance(episode, HandEpisode)
    assert len(episode) == 2
    assert episode.episode_id == EPISODE_ID
    assert episode.asset_id == ASSET_ID
    assert episode.caption == "wipe the work surface"
    assert episode.lane_status == "available"
    assert episode.run_id == "hand-run"
    assert episode.job_id == "hand-job"
    assert episode.source_yield_stats["total_frames"] == 900
    assert episode.yield_stats["total_frames"] == 2
    assert episode.continuous_intervals == [
        {
            "start": 0,
            "end": 0,
            "n_present": 1,
            "holes": [],
            "source_start": 100,
            "source_end": 100,
        }
    ]
    assert episode.source_frame_index(0) == 100
    assert episode.source_frame_index(1) == 101
    assert episode.episode_local_frame_index(101) == 1
    assert Path(episode.path_manager.param_file(0)).name == "frame_000100.npz"

    reader = _FakeVideoReader()
    episode._video_readers["left_front"] = reader
    frame = episode[1]

    assert reader.read_indices == [1]
    assert frame.frame_idx == 1
    assert frame.left_hand is not None
    assert frame.left_hand.hand_frame_idx == 101
    assert frame.left_hand.keypoints3d[0, 0] == 101


def test_pose_extraction_is_reused_only_after_complete_atomic_promotion(tmp_path: Path) -> None:
    lane_dir = _build_flat_hand_lane(tmp_path)
    download = _episode_download(lane_dir)

    first = ClippedHandEpisode.from_download(download, active_cameras=[])
    first_cache = Path(first.pose_cache_dir)
    second = ClippedHandEpisode.from_download(download, active_cameras=[])

    assert Path(second.pose_cache_dir) == first_cache
    assert (first_cache / "extraction.json").is_file()
    assert not [path for path in first_cache.parent.iterdir() if path.is_dir() and path.name.startswith(".pf-")]


def test_full_asset_download_opens_entire_hand_segment(tmp_path: Path) -> None:
    archive_path = _build_full_hand_archive(tmp_path)
    download = _asset_download(archive_path, tmp_path / "asset_cache")

    episode = HandEpisode.from_asset_download(download, segment=2, active_cameras=[])

    assert type(episode) is HandEpisode
    assert len(episode) == 2
    assert episode.asset_id == ASSET_ID
    assert episode.episode_id == ""
    assert episode.segment == 2
    assert episode.run_id == "hand-run"
    assert episode.continuous_intervals == [{"start": 0, "end": 1}]
    assert episode[1].left_hand is not None
    assert episode[1].left_hand.hand_frame_idx == 1
    assert Path(episode.session_dir).is_relative_to(Path(download.root_dir))


def test_full_asset_extraction_is_content_cached(tmp_path: Path) -> None:
    archive_path = _build_full_hand_archive(tmp_path)
    download = _asset_download(archive_path, tmp_path / "asset_cache")

    first = HandEpisode.from_asset_download(download, segment=2, active_cameras=[])
    second = HandEpisode.from_asset_download(download, segment=2, active_cameras=[])

    assert second.session_dir == first.session_dir
    assert Path(first.session_dir).parent.joinpath("extraction.json").is_file()


def test_full_asset_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive_path = _build_full_hand_archive(tmp_path, malicious_member=True)
    download = _asset_download(archive_path, tmp_path / "asset_cache")

    with pytest.raises(ValueError, match="Unsafe path"):
        HandEpisode.from_asset_download(download, segment=2, active_cameras=[])

    assert not (tmp_path / "escape.txt").exists()
    extraction_root = Path(download.root_dir) / ".grounded"
    assert not [path for path in extraction_root.iterdir() if path.is_dir()]


def test_pose_extraction_rejects_traversal_without_writing_outside_lane(tmp_path: Path) -> None:
    lane_dir = _build_flat_hand_lane(tmp_path, malicious_member=True)
    download = _episode_download(lane_dir)

    with pytest.raises(ValueError, match="Unsafe path"):
        ClippedHandEpisode.from_download(download, active_cameras=[])

    assert not (tmp_path / "escape.npz").exists()
    extraction_root = lane_dir / ".grounded"
    assert not [path for path in extraction_root.iterdir() if path.is_dir()]


def test_from_download_reports_unavailable_hand_lane_without_creating_files(tmp_path: Path) -> None:
    download = EpisodeDownload(
        episode_id=EPISODE_ID,
        asset_id=ASSET_ID,
        start_ns=START_NS,
        end_ns=END_NS,
        root_dir=str(tmp_path),
        lanes=(
            EpisodeLaneDownload(
                lane="hand",
                status="failed",
                files=(),
                message="pose clipping failed",
            ),
        ),
        files=(),
    )

    with pytest.raises(ValueError, match="hand lane is failed: pose clipping failed"):
        ClippedHandEpisode.from_download(download)

    assert not list(tmp_path.iterdir())


def test_from_download_rejects_identity_mismatch_before_pose_extraction(tmp_path: Path) -> None:
    lane_dir = _build_flat_hand_lane(tmp_path)
    download = _episode_download(lane_dir, episode_id="ep_v1_wrong")

    with pytest.raises(ValueError, match="episode_id does not match"):
        ClippedHandEpisode.from_download(download, active_cameras=[])

    assert not (lane_dir / ".grounded").exists()


def test_from_download_rejects_declared_camera_missing_from_exact_download(tmp_path: Path) -> None:
    lane_dir = _build_flat_hand_lane(tmp_path)
    # The file remains in the shared lane directory but is absent from this exact download result.
    download = _episode_download(lane_dir, exclude_names={"left_front.mp4"})

    with pytest.raises(FileNotFoundError, match="manifest declaration.*left_front.mp4"):
        ClippedHandEpisode.from_download(download)

    assert not (lane_dir / ".grounded").exists()


def test_from_download_rejects_declared_sidecar_missing_from_exact_download(tmp_path: Path) -> None:
    lane_dir = _build_flat_hand_lane(tmp_path)
    download = _episode_download(lane_dir, exclude_names={"source_yield.json"})

    with pytest.raises(FileNotFoundError, match="manifest declaration.*source_yield.json"):
        ClippedHandEpisode.from_download(download, active_cameras=[])

    assert not (lane_dir / ".grounded").exists()


def test_from_download_does_not_read_stale_unlisted_sidecar(tmp_path: Path) -> None:
    lane_dir = _build_flat_hand_lane(tmp_path)
    clip_manifest_path = lane_dir / "clip_manifest.json"
    clip_manifest = json.loads(clip_manifest_path.read_text())
    clip_manifest["source_sidecars"].remove("source_yield.json")
    clip_manifest_path.write_text(json.dumps(clip_manifest))
    stale_metrics = lane_dir / "source_pose_cleaning_metrics.json"
    stale_metrics.write_text(json.dumps({"must_not_be_read": True}))
    # Both files remain on disk but are neither declared nor part of the exact download.
    download = _episode_download(
        lane_dir,
        exclude_names={"source_yield.json", "source_pose_cleaning_metrics.json"},
    )

    episode = ClippedHandEpisode.from_download(download, active_cameras=[])

    assert episode.source_yield_stats == {}
    assert "source_yield.json" not in episode.declared_source_sidecars
    assert episode.pose_cleaning_metrics == {}
