from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from grounded.data.processing import (
    ASSET_CONTRACT_VERSION,
    ASSET_LIST_CONTRACT_VERSION,
    PROCESS_RECEIPT_CONTRACT_VERSION,
    PROCESS_REQUEST_CONTRACT_VERSION,
    AmbiguousAssetArtifactError,
    AssetArtifactNotReadyError,
    AssetNotFoundError,
    HttpAssetResolver,
    HttpEpisodeResolver,
    JsonAssetResolver,
    ProcessingClient,
    ProcessingError,
)


def _asset_contract(*, second_hand_attempt: bool = False) -> dict:
    artifacts = [
        {
            "lane": "hand",
            "run_id": "hand-run-1",
            "job_id": "hand-job-1",
            "status": "succeeded",
            "output_uri": "s3://processed/session/processed-segment2/hand/",
            "artifact_uri": "s3://processed/session/processed-segment2/hand/hand_v2_outputs.tar",
            "manifest_uri": "s3://processed/session/processed-segment2/hand/upload_manifest.json",
        },
        {
            "lane": "slam",
            "run_id": "slam-run-1",
            "job_id": "slam-job-1",
            "status": "running",
            "output_uri": "s3://processed/session/processed-segment2/slam/",
        },
    ]
    if second_hand_attempt:
        artifacts.append(
            {
                "lane": "hand",
                "run_id": "hand-run-2",
                "job_id": "hand-job-2",
                "status": "succeeded",
                "output_uri": "s3://processed/session/processed-segment2/hand-retry/",
            }
        )
    return {
        "schema_version": ASSET_CONTRACT_VERSION,
        "asset": {
            "asset_id": "asset-segment-2",
            "segment": 2,
            "source_uri": "s3://raw/session/",
            "timebase": {
                "reference_stream": "left_front",
                "mapping_status": "awaiting-authoritative-timestamps",
            },
            "artifacts": artifacts,
        },
    }


def _public_asset_contract(*, asset_id: str = "asset-segment-2") -> dict:
    contract = _asset_contract()
    asset = contract["asset"]
    asset.update(
        {
            "asset_id": asset_id,
        }
    )
    asset["artifacts"] = [
        {
            "lane": "hand",
            "status": "available",
            "run_id": "hand-run-1",
            "job_id": "hand-job-1",
            "output_uri": "s3://processed/session/processed-segment2/hand/",
            "files": [
                {
                    "relative_path": "hand_v2_outputs.tar",
                    "uri": "s3://processed/session/processed-segment2/hand/hand_v2_outputs.tar",
                    "size_bytes": 1234,
                }
            ],
        },
        {"lane": "slam", "status": "not_processed", "files": []},
        {"lane": "depth", "status": "not_processed", "files": []},
    ]
    return contract


def test_json_asset_resolver_preserves_asset_id_and_timebase(tmp_path: Path) -> None:
    contract_path = tmp_path / "asset.json"
    contract_path.write_text(json.dumps(_asset_contract()), encoding="utf-8")

    resolver = JsonAssetResolver(contract_path)
    asset = resolver.asset("asset-segment-2")

    assert asset.asset_id == "asset-segment-2"
    assert asset.segment == 2
    assert asset.timebase["mapping_status"] == "awaiting-authoritative-timestamps"
    assert [item.lane for item in asset.artifacts] == ["hand", "slam"]


def test_json_asset_resolver_accepts_resolved_public_lane_states_without_fake_lineage() -> None:
    contract = _public_asset_contract()
    asset = contract["asset"]
    asset["provenance"] = {"legacy_run_id": "old-run-1"}
    asset["artifacts"][2] = {"lane": "depth", "status": "failed", "message": "depth worker failed", "files": []}

    resolved = JsonAssetResolver(contract, strict=True).asset("asset-segment-2")

    assert resolved.provenance["legacy_run_id"] == "old-run-1"
    assert resolved.artifacts[1].run_id == ""
    assert resolved.artifacts[1].job_id == ""
    assert resolved.artifacts[1].output_uri == ""
    assert (
        ProcessingClient(asset_resolver=JsonAssetResolver(contract, strict=True))
        .resolve_asset_artifact("asset-segment-2", lane="hand")
        .status
        == "available"
    )


def test_json_asset_resolver_does_not_require_internal_source_uri() -> None:
    contract = _public_asset_contract()
    contract["asset"].pop("source_uri")

    resolved = JsonAssetResolver(contract, strict=True).asset("asset-segment-2")

    assert resolved.source_uri == ""


def test_json_asset_resolver_rejects_files_for_unavailable_lane_state() -> None:
    contract, _ = _downloadable_asset_contract()
    contract["asset"]["artifacts"][0]["status"] = "not_processed"

    with pytest.raises(ProcessingError, match="not_processed but publishes files"):
        JsonAssetResolver(contract)


def test_processing_client_rejects_duplicate_resolved_lane_states() -> None:
    asset_resolver = JsonAssetResolver(_asset_contract(second_hand_attempt=True))
    client = ProcessingClient(asset_resolver=asset_resolver)

    with pytest.raises(AmbiguousAssetArtifactError, match="multiple resolved hand states"):
        client.resolve_asset_artifact("asset-segment-2", lane="hand")


def test_processing_client_rejects_incomplete_asset_artifact() -> None:
    client = ProcessingClient(asset_resolver=JsonAssetResolver(_asset_contract()))

    with pytest.raises(AssetArtifactNotReadyError, match="is running"):
        client.resolve_asset_artifact("asset-segment-2", lane="slam")


def test_json_asset_resolver_rejects_missing_and_duplicate_ids() -> None:
    resolver = JsonAssetResolver(_asset_contract())
    with pytest.raises(AssetNotFoundError):
        resolver.asset("missing")

    duplicated = _asset_contract()
    duplicated["assets"] = [duplicated.pop("asset")] * 2
    with pytest.raises(ProcessingError, match="duplicate asset_id"):
        JsonAssetResolver(duplicated)


def test_json_asset_resolver_requires_versioned_contract() -> None:
    contract = _asset_contract()
    del contract["schema_version"]

    with pytest.raises(ProcessingError, match="unsupported asset contract version"):
        JsonAssetResolver(contract)


class _JsonResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int = 200):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_http_asset_resolver_calls_versioned_asset_endpoint() -> None:
    requests = []

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        return _JsonResponse(json.dumps(_public_asset_contract()).encode())

    resolver = HttpAssetResolver(
        "https://api.gsi.example/",
        bearer_token="test-token",
        timeout_seconds=12.5,
        request_opener=open_request,
    )

    asset = resolver.asset("asset-segment-2")

    assert asset.asset_id == "asset-segment-2"
    request, timeout = requests[0]
    assert request.full_url == "https://api.gsi.example/v1/assets/asset-segment-2"
    assert request.get_header("Authorization") == "Bearer test-token"
    assert timeout == 12.5


def test_http_asset_resolver_lists_assets_with_explicit_filters_and_cursor() -> None:
    requests = []
    asset = _public_asset_contract()["asset"]
    payload = {
        "schema_version": ASSET_LIST_CONTRACT_VERSION,
        "assets": [asset],
        "next_cursor": "opaque-next",
    }

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        return _JsonResponse(json.dumps(payload).encode())

    resolver = HttpAssetResolver("https://api.gsi.example", bearer_token="token", request_opener=open_request)
    page = resolver.list_assets(
        lane="hand",
        status="available",
        cursor="opaque-current",
        limit=25,
    )

    assert [item.asset_id for item in page.assets] == ["asset-segment-2"]
    assert page.next_cursor == "opaque-next"
    request, _ = requests[0]
    query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
    assert query == {
        "lane": ["hand"],
        "status": ["available"],
        "cursor": ["opaque-current"],
        "limit": ["25"],
    }
    assert request.get_header("Authorization") == "Bearer token"


def test_http_asset_resolver_submits_versioned_process_request() -> None:
    requests = []
    response = {
        "schema_version": PROCESS_RECEIPT_CONTRACT_VERSION,
        "request_id": "proc_test",
        "asset_id": "asset-segment-2",
        "lanes": [
            {"lane": "hand", "state": "already_available", "run_id": "hand-run-1"},
            {"lane": "slam", "state": "accepted", "run_id": "slam-run-2", "job_id": "slam-job-2"},
        ],
    }

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        return _JsonResponse(json.dumps(response).encode(), status=202)

    resolver = HttpAssetResolver("https://api.gsi.example", bearer_token="token", request_opener=open_request)
    receipt = resolver.process_asset(
        "asset-segment-2",
        lanes=["hand", "SLAM", "hand"],
        idempotency_key="client-retry-1",
    )

    assert receipt.request_id == "proc_test"
    assert [(item.lane, item.state) for item in receipt.lanes] == [
        ("hand", "already_available"),
        ("slam", "accepted"),
    ]
    request, _ = requests[0]
    assert request.method == "POST"
    assert request.full_url == "https://api.gsi.example/v1/assets/asset-segment-2/process"
    assert json.loads(request.data) == {
        "schema_version": PROCESS_REQUEST_CONTRACT_VERSION,
        "lanes": ["hand", "slam"],
        "idempotency_key": "client-retry-1",
        "retry_failed": False,
        "rerun_available": False,
    }


def test_processing_client_exposes_http_asset_catalog_and_process_methods() -> None:
    payloads = [
        {
            "schema_version": ASSET_LIST_CONTRACT_VERSION,
            "assets": [_public_asset_contract()["asset"]],
            "next_cursor": None,
        },
        {
            "schema_version": PROCESS_RECEIPT_CONTRACT_VERSION,
            "request_id": "proc_test",
            "asset_id": "asset-segment-2",
            "lanes": [{"lane": "depth", "state": "not_supported"}],
        },
    ]

    def open_request(request, *, timeout):
        return _JsonResponse(json.dumps(payloads.pop(0)).encode())

    resolver = HttpAssetResolver("https://api.gsi.example", request_opener=open_request)
    client = ProcessingClient(asset_resolver=resolver)

    assert client.list_assets(limit=1).assets[0].asset_id == "asset-segment-2"
    assert client.process_asset("asset-segment-2", lanes=["depth"]).lanes[0].state == "not_supported"


def test_http_asset_resolver_preserves_versioned_api_error() -> None:
    body = json.dumps(
        {
            "schema_version": "grounded.error.v1alpha1",
            "error": {"code": "asset_not_found", "message": "asset does not exist", "request_id": "req-1"},
        }
    ).encode()

    def open_request(request, *, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, _JsonResponse(body))

    resolver = HttpAssetResolver("https://api.gsi.example", request_opener=open_request)

    with pytest.raises(AssetNotFoundError, match="asset does not exist.*req-1"):
        resolver.asset("missing-asset")


def test_asset_only_client_resolves_without_run_registry() -> None:
    client = ProcessingClient(asset_resolver=JsonAssetResolver(_asset_contract()))

    assert client.get_asset("asset-segment-2").segment == 2


def test_processing_client_from_manifest_configures_offline_asset_resolver(tmp_path: Path) -> None:
    path = tmp_path / "assets.json"
    path.write_text(json.dumps(_asset_contract()), encoding="utf-8")

    client = ProcessingClient.from_manifest(path)

    assert client.get_asset("asset-segment-2").asset_id == "asset-segment-2"


def test_open_hand_dispatches_to_asset_reader_for_asset_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ProcessingClient(asset_resolver=JsonAssetResolver(_asset_contract()))
    calls = []

    def open_asset(asset_id: str, **kwargs):
        calls.append((asset_id, kwargs))
        return "full-hand-view"

    monkeypatch.setattr(client, "open_hand_asset", open_asset)

    result = client.open_hand("asset-segment-2", active_cameras=["left_front"])

    assert result == "full-hand-view"
    assert calls[0][0] == "asset-segment-2"
    assert calls[0][1]["active_cameras"] == ["left_front"]


def test_processing_client_from_api_configures_asset_and_episode_resolvers() -> None:
    client = ProcessingClient.from_api("https://api.gsi.example", bearer_token="token")

    assert isinstance(client.asset_resolver, HttpAssetResolver)
    assert isinstance(client.episode_resolver, HttpEpisodeResolver)


class _MemoryTransport:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.downloads: list[str] = []

    def download(self, uri: str, target: Path) -> None:
        self.downloads.append(uri)
        target.write_bytes(self.objects[uri])


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _downloadable_asset_contract() -> tuple[dict, dict[str, bytes]]:
    objects = {
        "s3://processed/hand/hand_v2_outputs.tar": b"hand archive",
        "s3://processed/slam/trajectory.csv": b"timestamp,x,y,z\n100,1,2,3\n",
        "s3://processed/slam/sync_manifest.json": b'{"clock":"sensor_ns"}',
    }
    contract = _asset_contract()
    hand, slam = contract["asset"]["artifacts"]
    hand["files"] = [
        {
            "relative_path": "hand_v2_outputs.tar",
            "uri": "s3://processed/hand/hand_v2_outputs.tar",
            "size_bytes": len(objects["s3://processed/hand/hand_v2_outputs.tar"]),
            "sha256": _sha256(objects["s3://processed/hand/hand_v2_outputs.tar"]),
        }
    ]
    slam["status"] = "succeeded"
    slam["files"] = [
        {
            "relative_path": "trajectory.csv",
            "uri": "s3://processed/slam/trajectory.csv",
            "size_bytes": len(objects["s3://processed/slam/trajectory.csv"]),
            "sha256": _sha256(objects["s3://processed/slam/trajectory.csv"]),
        },
        {
            "relative_path": "sync/sync_manifest.json",
            "uri": "s3://processed/slam/sync_manifest.json",
            "size_bytes": len(objects["s3://processed/slam/sync_manifest.json"]),
            "sha256": _sha256(objects["s3://processed/slam/sync_manifest.json"]),
        },
    ]
    return contract, objects


def test_download_asset_defaults_to_real_files_and_skips_unavailable_lane_states(tmp_path: Path) -> None:
    contract = _public_asset_contract()
    hand_file = contract["asset"]["artifacts"][0]["files"][0]
    payload = b"hand archive"
    hand_file["size_bytes"] = len(payload)
    hand_file["sha256"] = _sha256(payload)
    transport = _MemoryTransport({hand_file["uri"]: payload})
    client = ProcessingClient(asset_resolver=JsonAssetResolver(contract, strict=True))

    result = client.download_asset("asset-segment-2", target_dir=str(tmp_path), transport=transport)

    assert transport.downloads == [hand_file["uri"]]
    assert [item.lane for item in result.files] == ["hand"]
    assert not list(tmp_path.rglob("*not_processed*"))


def test_download_asset_explicit_unavailable_lane_creates_no_file(tmp_path: Path) -> None:
    contract = _public_asset_contract()
    transport = _MemoryTransport({})
    client = ProcessingClient(asset_resolver=JsonAssetResolver(contract, strict=True))

    with pytest.raises(AssetArtifactNotReadyError, match="is not_processed"):
        client.download_asset("asset-segment-2", lanes=["depth"], target_dir=str(tmp_path), transport=transport)

    assert transport.downloads == []
    assert not list(tmp_path.rglob("*"))


def test_download_asset_fetches_only_exact_manifest_objects(tmp_path: Path) -> None:
    contract, objects = _downloadable_asset_contract()
    transport = _MemoryTransport(objects)
    client = ProcessingClient(asset_resolver=JsonAssetResolver(contract))

    result = client.download_asset("asset-segment-2", target_dir=str(tmp_path), transport=transport)

    assert transport.downloads == list(objects)
    relative_paths = {item.source_uri: Path(item.local_path).relative_to(result.root_dir).as_posix() for item in result.files}
    assert relative_paths["s3://processed/hand/hand_v2_outputs.tar"].endswith("/hand_v2_outputs.tar")
    assert relative_paths["s3://processed/slam/trajectory.csv"].endswith("/trajectory.csv")
    assert relative_paths["s3://processed/slam/sync_manifest.json"].endswith("/sync/sync_manifest.json")
    for item in result.files:
        assert Path(item.local_path).read_bytes() == objects[item.source_uri]
        assert item.sha256 == _sha256(objects[item.source_uri])
        assert item.size_verified
        assert item.sha256_verified


@pytest.mark.parametrize("relative_path", ["../escape", "/absolute", "C:/escape", "lane:file"])
def test_asset_contract_rejects_unsafe_download_paths(relative_path: str) -> None:
    contract, _ = _downloadable_asset_contract()
    contract["asset"]["artifacts"][0]["files"][0]["relative_path"] = relative_path

    with pytest.raises(ProcessingError, match="unsafe relative path"):
        JsonAssetResolver(contract)


@pytest.mark.parametrize("relative_path", ["a\\b", "a//b", "a/./b"])
def test_strict_asset_contract_rejects_noncanonical_posix_paths(relative_path: str) -> None:
    contract = _public_asset_contract()
    contract["asset"]["artifacts"][0]["files"][0]["relative_path"] = relative_path

    with pytest.raises(ProcessingError, match="unsafe relative path"):
        JsonAssetResolver(contract, strict=True)


@pytest.mark.parametrize("size_bytes", [True, "123", -1])
def test_strict_asset_contract_rejects_invalid_file_sizes(size_bytes) -> None:
    contract = _public_asset_contract()
    contract["asset"]["artifacts"][0]["files"][0]["size_bytes"] = size_bytes

    with pytest.raises(ProcessingError, match="integer|non-negative"):
        JsonAssetResolver(contract, strict=True)


def test_strict_asset_contract_rejects_malformed_sha256() -> None:
    contract = _public_asset_contract()
    contract["asset"]["artifacts"][0]["files"][0]["sha256"] = "not-a-digest"

    with pytest.raises(ProcessingError, match="64 hexadecimal"):
        JsonAssetResolver(contract, strict=True)


def test_download_asset_rejects_checksum_mismatch_and_removes_partial_file(tmp_path: Path) -> None:
    contract, objects = _downloadable_asset_contract()
    contract["asset"]["artifacts"][0]["files"][0]["sha256"] = "0" * 64
    transport = _MemoryTransport(objects)
    client = ProcessingClient(asset_resolver=JsonAssetResolver(contract))

    with pytest.raises(ProcessingError, match="SHA-256 mismatch"):
        client.download_asset("asset-segment-2", lanes=["hand"], target_dir=str(tmp_path), transport=transport)

    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("hand_v2_outputs.tar"))


def test_download_asset_can_require_published_sha256(tmp_path: Path) -> None:
    contract, objects = _downloadable_asset_contract()
    contract["asset"]["artifacts"][0]["files"][0]["sha256"] = None
    transport = _MemoryTransport(objects)
    client = ProcessingClient(asset_resolver=JsonAssetResolver(contract))

    with pytest.raises(ProcessingError, match="has no published SHA-256"):
        client.download_asset(
            "asset-segment-2",
            lanes=["hand"],
            target_dir=str(tmp_path),
            transport=transport,
            require_sha256=True,
        )

    assert transport.downloads == []
