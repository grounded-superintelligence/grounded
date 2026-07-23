from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from grounded.processing import (
    EPISODE_CONTRACT_VERSION,
    EpisodeLaneNotFoundError,
    EpisodeNotFoundError,
    HttpEpisodeResolver,
    JsonEpisodeResolver,
    ProcessingClient,
    ProcessingError,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _episode_contract() -> tuple[dict, dict[str, bytes]]:
    asset_id = "ast_v1_example"
    start_ns = 9_000_000_000_000
    end_ns = 9_001_000_000_000
    objects = {
        "s3://episodes/example/hand/left_front.mp4": b"clipped hand video",
        "s3://episodes/example/slam/trajectory.txt": b"9000.0 0 0 0 0 0 0 1\n",
    }
    return (
        {
            "schema_version": EPISODE_CONTRACT_VERSION,
            "episode": {
                "episode_id": "ep_v1_e24d2a5465d99de11f62ed24cf533944",
                "asset_id": asset_id,
                "segment": 2,
                "caption": "rolls dough with both hands",
                "interval": {
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "clock": "sensor_ns",
                    "bounds": "[start_ns,end_ns)",
                },
                "lanes": [
                    {
                        "lane": "hand",
                        "status": "available",
                        "run_id": "hand-run",
                        "job_id": "hand-job",
                        "files": [
                            {
                                "relative_path": "left_front.mp4",
                                "uri": "s3://episodes/example/hand/left_front.mp4",
                                "size_bytes": len(objects["s3://episodes/example/hand/left_front.mp4"]),
                                "sha256": _sha256(objects["s3://episodes/example/hand/left_front.mp4"]),
                            }
                        ],
                    },
                    {
                        "lane": "slam",
                        "status": "partial",
                        "message": "trajectory clipped; visualization not produced",
                        "provenance": {"run_id": "slam-run", "job_id": "slam-job"},
                        "files": [
                            {
                                "relative_path": "trajectory.txt",
                                "uri": "s3://episodes/example/slam/trajectory.txt",
                                "size_bytes": len(objects["s3://episodes/example/slam/trajectory.txt"]),
                                "sha256": _sha256(objects["s3://episodes/example/slam/trajectory.txt"]),
                            }
                        ],
                    },
                    {
                        "lane": "depth",
                        "status": "not_processed",
                        "message": "depth was not processed for the parent asset",
                        "files": [],
                    },
                ],
            },
        },
        objects,
    )


class _MemoryTransport:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.downloads: list[str] = []

    def download(self, uri: str, target: Path) -> None:
        self.downloads.append(uri)
        target.write_bytes(self.objects[uri])


def test_json_episode_resolver_preserves_identity_bounds_states_and_lineage() -> None:
    contract, _ = _episode_contract()
    resolver = JsonEpisodeResolver(contract)
    episode_id = contract["episode"]["episode_id"]

    episode = resolver.episode(episode_id)

    assert episode.asset_id == "ast_v1_example"
    assert (episode.start_ns, episode.end_ns) == (9_000_000_000_000, 9_001_000_000_000)
    assert [lane.lane for lane in episode.lanes] == ["hand", "slam", "depth"]
    assert episode.lanes[0].run_id == "hand-run"
    assert episode.lanes[1].run_id == "slam-run"
    assert episode.lanes[2].status == "not_processed"

    with pytest.raises(EpisodeNotFoundError):
        resolver.episode("ep_v1_missing")


def test_processing_client_from_manifest_lists_episodes(tmp_path: Path) -> None:
    contract, _ = _episode_contract()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    client = ProcessingClient.from_manifest(path)
    episodes = client.list_episodes()

    assert len(episodes) == 1
    assert episodes[0].asset_id == "ast_v1_example"


def test_open_hand_dispatches_to_episode_reader_for_episode_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    contract, _ = _episode_contract()
    client = ProcessingClient(episode_resolver=JsonEpisodeResolver(contract))
    episode_id = contract["episode"]["episode_id"]
    calls = []

    def open_episode(identifier: str, **kwargs):
        calls.append((identifier, kwargs))
        return "clipped-hand-view"

    monkeypatch.setattr(client, "open_hand_episode", open_episode)

    result = client.open_hand(episode_id, active_cameras=["left_front"])

    assert result == "clipped-hand-view"
    assert calls[0][0] == episode_id
    assert calls[0][1]["active_cameras"] == ["left_front"]


def test_download_episode_defaults_to_all_lane_states_and_exact_existing_files(tmp_path: Path) -> None:
    contract, objects = _episode_contract()
    transport = _MemoryTransport(objects)
    resolver = JsonEpisodeResolver(contract)
    client = ProcessingClient(episode_resolver=resolver)
    episode_id = contract["episode"]["episode_id"]

    result = client.download_episode(episode_id, target_dir=str(tmp_path), transport=transport)

    assert transport.downloads == list(objects)
    assert [(lane.lane, lane.status, len(lane.files)) for lane in result.lanes] == [
        ("depth", "not_processed", 0),
        ("hand", "available", 1),
        ("slam", "partial", 1),
    ]
    assert result.lanes[0].message == "depth was not processed for the parent asset"
    assert result.lanes[1].run_id == "hand-run"
    assert result.lanes[2].run_id == "slam-run"
    assert len(result.files) == 2
    for item in result.files:
        assert Path(item.local_path).read_bytes() == objects[item.source_uri]
        assert item.size_verified
        assert item.sha256_verified


def test_download_episode_requested_unprocessed_lane_returns_status_without_files(tmp_path: Path) -> None:
    contract, objects = _episode_contract()
    transport = _MemoryTransport(objects)
    client = ProcessingClient(episode_resolver=JsonEpisodeResolver(contract))
    episode_id = contract["episode"]["episode_id"]

    result = client.download_episode(episode_id, lane="DEPTH", target_dir=str(tmp_path), transport=transport)

    assert [(item.lane, item.status, item.files) for item in result.lanes] == [
        ("depth", "not_processed", ()),
    ]
    assert result.files == ()
    assert transport.downloads == []
    assert not list(tmp_path.rglob("*"))


def test_download_episode_requested_lane_does_not_download_other_lanes(tmp_path: Path) -> None:
    contract, objects = _episode_contract()
    transport = _MemoryTransport(objects)
    client = ProcessingClient(episode_resolver=JsonEpisodeResolver(contract))
    episode_id = contract["episode"]["episode_id"]

    result = client.download_episode(episode_id, lane="hand", target_dir=str(tmp_path), transport=transport)

    assert [item.lane for item in result.lanes] == ["hand"]
    assert transport.downloads == ["s3://episodes/example/hand/left_front.mp4"]

    with pytest.raises(EpisodeLaneNotFoundError, match="publishes no audio lane state"):
        client.download_episode(episode_id, lane="audio", target_dir=str(tmp_path), transport=transport)


@pytest.mark.parametrize("status", ["not_processed", "failed"])
def test_download_episode_unavailable_statuses_never_create_files(tmp_path: Path, status: str) -> None:
    contract, objects = _episode_contract()
    depth = contract["episode"]["lanes"][2]
    depth.update(status=status, message=f"depth {status}")
    transport = _MemoryTransport(objects)
    client = ProcessingClient(episode_resolver=JsonEpisodeResolver(contract))

    result = client.download_episode(
        contract["episode"]["episode_id"],
        lane="depth",
        target_dir=str(tmp_path),
        transport=transport,
    )

    assert result.lanes[0].status == status
    assert result.lanes[0].message == f"depth {status}"
    assert result.files == ()
    assert transport.downloads == []
    assert not list(tmp_path.rglob("*"))


def test_partial_episode_lane_may_report_no_files(tmp_path: Path) -> None:
    contract, objects = _episode_contract()
    slam = contract["episode"]["lanes"][1]
    slam.update(files=[], message="SLAM clipping produced no complete output")
    transport = _MemoryTransport(objects)
    client = ProcessingClient(episode_resolver=JsonEpisodeResolver(contract))

    result = client.download_episode(
        contract["episode"]["episode_id"],
        lane="slam",
        target_dir=str(tmp_path),
        transport=transport,
    )

    assert [(item.status, item.files) for item in result.lanes] == [("partial", ())]
    assert transport.downloads == []


def test_download_episode_can_require_published_sha256(tmp_path: Path) -> None:
    contract, objects = _episode_contract()
    contract["episode"]["lanes"][0]["files"][0]["sha256"] = None
    transport = _MemoryTransport(objects)
    client = ProcessingClient(episode_resolver=JsonEpisodeResolver(contract))

    with pytest.raises(ProcessingError, match="has no published SHA-256"):
        client.download_episode(
            contract["episode"]["episode_id"],
            lane="hand",
            target_dir=str(tmp_path),
            transport=transport,
            require_sha256=True,
        )

    assert transport.downloads == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda episode: episode["interval"].update(clock="slam_seconds"), "clock must be sensor_ns"),
        (lambda episode: episode["interval"].update(bounds="inclusive"), "bounds must be"),
        (lambda episode: episode.update(episode_id="ep_v1_wrong"), "episode_id does not match"),
        (lambda episode: episode["lanes"].pop(), "missing required lane states: depth"),
        (lambda episode: episode["lanes"][0].update(status="completed"), "unsupported episode lane status"),
        (lambda episode: episode["lanes"][0].update(files=[]), "available but publishes no exact files"),
        (lambda episode: episode["lanes"][2].update(files=episode["lanes"][0]["files"]), "not_processed but publishes files"),
    ],
)
def test_episode_contract_rejects_ambiguous_or_impossible_states(mutate, message: str) -> None:
    contract, _ = _episode_contract()
    mutate(contract["episode"])

    with pytest.raises(ProcessingError, match=message):
        JsonEpisodeResolver(contract)


def test_episode_contract_rejects_fractional_nanosecond_bounds() -> None:
    contract, _ = _episode_contract()
    contract["episode"]["interval"]["start_ns"] = 1.5

    with pytest.raises(ProcessingError, match="start_ns must be an integer"):
        JsonEpisodeResolver(contract)


class _JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_http_episode_resolver_calls_episode_endpoint() -> None:
    contract, _ = _episode_contract()
    requests = []

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        return _JsonResponse(json.dumps(contract).encode())

    resolver = HttpEpisodeResolver(
        "https://api.gsi.example/",
        bearer_token="token",
        timeout_seconds=12.5,
        request_opener=open_request,
    )
    episode_id = contract["episode"]["episode_id"]

    assert resolver.episode(episode_id).episode_id == episode_id
    request, timeout = requests[0]
    assert request.full_url == f"https://api.gsi.example/v1/episodes/{episode_id}"
    assert request.get_header("Authorization") == "Bearer token"
    assert timeout == 12.5
