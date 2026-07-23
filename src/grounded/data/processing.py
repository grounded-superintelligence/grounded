"""Resolve and download immutable Grounded assets and multimodal episodes."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Protocol

ASSET_CONTRACT_VERSION = "grounded.asset.v1alpha1"
ASSET_LIST_CONTRACT_VERSION = "grounded.asset-list.v1alpha1"
ERROR_CONTRACT_VERSION = "grounded.error.v1alpha1"
EPISODE_CONTRACT_VERSION = "grounded.episode.v1alpha1"
PROCESS_REQUEST_CONTRACT_VERSION = "grounded.process-request.v1alpha1"
PROCESS_RECEIPT_CONTRACT_VERSION = "grounded.process-receipt.v1alpha1"
DEFAULT_ASSET_CACHE = "~/.cache/grounded/data"

SUPPORTED_ASSET_LANES = frozenset({"hand", "slam", "depth"})
ASSET_LANE_STATUSES = frozenset({"available", "partial", "queued", "running", "not_processed", "failed"})
LEGACY_AVAILABLE_ASSET_STATUSES = frozenset({"succeeded", "completed"})
READY_ASSET_STATUSES = frozenset({"available", *LEGACY_AVAILABLE_ASSET_STATUSES})
DOWNLOADABLE_ASSET_STATUSES = frozenset({*READY_ASSET_STATUSES, "partial"})
EMPTY_ASSET_STATUSES = frozenset({"not_processed", "failed"})
EPISODE_LANE_STATUSES = frozenset({"available", "partial", "not_processed", "failed"})
DOWNLOADABLE_EPISODE_LANE_STATUSES = frozenset({"available", "partial"})
REQUIRED_EPISODE_LANES = frozenset({"hand", "slam", "depth"})
PROCESS_RECEIPT_STATES = frozenset({"accepted", "already_running", "already_available", "retry_required", "not_supported"})


def _expected_episode_id(*, asset_id: str, start_ns: Any, end_ns: Any) -> str:
    """Validate the producer's opaque episode ID without exposing ID construction."""

    def parse_bound(value: Any, name: str) -> int:
        if isinstance(value, Integral) and not isinstance(value, bool):
            return int(value)
        raise ProcessingError(f"{name} must be an integer")

    start = parse_bound(start_ns, "start_ns")
    end = parse_bound(end_ns, "end_ns")
    if start < 0:
        raise ProcessingError("start_ns must be non-negative")
    if end <= start:
        raise ProcessingError("end_ns must be greater than start_ns")
    identity = {
        "scheme": "grounded.episode.v1",
        "asset_id": _required_text(asset_id, field_name="asset_id"),
        "start_ns": start,
        "end_ns": end,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"ep_v1_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]}"


def _cache_component(value: str, *, visible_chars: int) -> str:
    visible = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")[:visible_chars] or "item"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{visible}-{digest}"


class ProcessingError(RuntimeError):
    """Base error for asset and episode resolution or artifact access."""


class AssetNotFoundError(ProcessingError):
    """Raised when a segment-level asset ID is absent from the resolver."""


class AssetApiError(ProcessingError):
    """Versioned error returned by the public asset API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: str = "",
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id


class AssetArtifactNotFoundError(ProcessingError):
    """Raised when an asset has no artifact matching the requested lane."""


class AssetArtifactNotReadyError(ProcessingError):
    """Raised when the selected asset artifact is not complete."""


class AmbiguousAssetArtifactError(ProcessingError):
    """Raised when artifact selection would silently choose between retries."""


class EpisodeNotFoundError(ProcessingError):
    """Raised when an episode ID is absent from the configured resolver."""


class EpisodeLaneNotFoundError(ProcessingError):
    """Raised when an episode does not publish a requested lane state."""


@dataclass(frozen=True)
class AssetFile:
    """One exact downloadable object within a lane artifact."""

    relative_path: str
    uri: str
    size_bytes: Optional[int] = None
    sha256: str = ""


@dataclass(frozen=True)
class AssetArtifact:
    """One lane output for a segment-level asset.

    ``run_id`` identifies the lane execution or retry. ``job_id`` remains in
    the contract for diagnostics and exact registry correlation; normal SDK
    consumers select by ``asset_id`` and lane.
    """

    lane: str
    run_id: str
    job_id: str
    status: str
    output_uri: str
    artifact_uri: str = ""
    manifest_uri: str = ""
    checksum_sha256: str = ""
    files: tuple[AssetFile, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetRecord:
    """One segment-level asset and its published lane states."""

    asset_id: str
    segment: int
    source_uri: str
    artifacts: tuple[AssetArtifact, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    timebase: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetPage:
    """One API page of immutable segment assets."""

    assets: tuple[AssetRecord, ...]
    next_cursor: Optional[str] = None


@dataclass(frozen=True)
class LaneProcessReceipt:
    """The control-plane result for one requested lane."""

    lane: str
    state: str
    run_id: str = ""
    job_id: str = ""
    message: str = ""


@dataclass(frozen=True)
class ProcessReceipt:
    """Idempotent response from ``POST /v1/assets/{asset_id}/process``."""

    request_id: str
    asset_id: str
    lanes: tuple[LaneProcessReceipt, ...]


@dataclass(frozen=True)
class EpisodeLane:
    """One clipped-lane state for an episode.

    ``run_id`` and ``job_id`` are optional lineage fields. Consumers normally
    select the episode and lane, not the processing attempt that produced it.
    """

    lane: str
    status: str
    files: tuple[AssetFile, ...] = ()
    run_id: str = ""
    job_id: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRecord:
    """One captioned interval within a parent asset."""

    episode_id: str
    asset_id: str
    start_ns: int
    end_ns: int
    lanes: tuple[EpisodeLane, ...]
    segment: Optional[int] = None
    legacy_key: str = ""
    caption: str = ""
    activity: str = ""
    interval: dict[str, Any] = field(default_factory=dict)
    timebase: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AssetResolver(Protocol):
    """Backend contract for segment-level multimodal asset metadata."""

    def asset(self, asset_id: str) -> AssetRecord: ...


class EpisodeResolver(Protocol):
    """Backend contract for multimodal episode metadata."""

    def episode(self, episode_id: str) -> EpisodeRecord: ...


class ArtifactTransport(Protocol):
    """Transport used for exact object downloads returned by the API."""

    def download(self, uri: str, target: Path) -> None: ...


@dataclass(frozen=True)
class DownloadedAssetFile:
    asset_id: str
    lane: str
    run_id: str
    source_uri: str
    local_path: str
    size_bytes: int
    sha256: str
    size_verified: bool
    sha256_verified: bool


@dataclass(frozen=True)
class AssetDownload:
    asset_id: str
    root_dir: str
    files: tuple[DownloadedAssetFile, ...]


@dataclass(frozen=True)
class DownloadedEpisodeFile:
    episode_id: str
    asset_id: str
    lane: str
    source_uri: str
    local_path: str
    size_bytes: int
    sha256: str
    size_verified: bool
    sha256_verified: bool


@dataclass(frozen=True)
class EpisodeLaneDownload:
    lane: str
    status: str
    files: tuple[DownloadedEpisodeFile, ...]
    run_id: str = ""
    job_id: str = ""
    message: str = ""

    @property
    def available(self) -> bool:
        """Whether this lane published at least one downloadable clipped file."""

        return bool(self.files)


@dataclass(frozen=True)
class EpisodeDownload:
    episode_id: str
    asset_id: str
    start_ns: int
    end_ns: int
    root_dir: str
    lanes: tuple[EpisodeLaneDownload, ...]
    files: tuple[DownloadedEpisodeFile, ...]


def _required_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ProcessingError(f"contract field is required: {field_name}")
    return text


def _safe_relative_path(value: Any, *, field_name: str, strict: bool = False) -> str:
    raw_text = _required_text(value, field_name=field_name)
    if strict and "\\" in raw_text:
        raise ProcessingError(f"unsafe relative path in asset contract: {raw_text}")
    text = raw_text.replace("\\", "/")
    path = PurePosixPath(text)
    parts = [part for part in path.parts if part not in {"", "."}]
    if (
        not parts
        or path.is_absolute()
        or text.startswith("/")
        or any(part == ".." or ":" in part or "\x00" in part for part in parts)
        or (strict and "/".join(parts) != text)
    ):
        raise ProcessingError(f"unsafe relative path in asset contract: {text}")
    return "/".join(parts)


def _asset_file_from_mapping(value: Mapping[str, Any], *, strict: bool = False) -> AssetFile:
    raw_size = value.get("size_bytes", value.get("bytes"))
    if strict:
        if raw_size is not None and (not isinstance(raw_size, int) or isinstance(raw_size, bool)):
            raise ProcessingError("asset contract field must be an integer: artifacts[].files[].size_bytes")
        size_bytes = raw_size
    else:
        try:
            size_bytes = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError) as exc:
            raise ProcessingError("asset contract field must be an integer: artifacts[].files[].size_bytes") from exc
    if size_bytes is not None and size_bytes < 0:
        raise ProcessingError("asset contract file size must be non-negative")
    sha256 = str(value.get("sha256") or "").strip()
    if strict and sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise ProcessingError("asset contract file sha256 must be 64 hexadecimal characters")
    return AssetFile(
        relative_path=_safe_relative_path(
            value.get("relative_path"),
            field_name="artifacts[].files[].relative_path",
            strict=strict,
        ),
        uri=_required_text(value.get("uri"), field_name="artifacts[].files[].uri"),
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _asset_artifact_from_mapping(value: Mapping[str, Any], *, strict: bool = False) -> AssetArtifact:
    lane = _required_text(value.get("lane"), field_name="artifacts[].lane").lower()
    status = _required_text(value.get("status"), field_name="artifacts[].status").lower()
    allowed_statuses = ASSET_LANE_STATUSES if strict else ASSET_LANE_STATUSES | LEGACY_AVAILABLE_ASSET_STATUSES
    if status not in allowed_statuses:
        choices = ", ".join(sorted(allowed_statuses))
        raise ProcessingError(f"unsupported asset lane status {status!r}; expected one of: {choices}")
    if strict and lane not in SUPPORTED_ASSET_LANES:
        raise ProcessingError(f"unsupported asset lane: {lane}")
    raw_files = value.get("files") or []
    if not isinstance(raw_files, list):
        raise ProcessingError("asset contract field must be a list: artifacts[].files")
    files: list[AssetFile] = []
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise ProcessingError("asset contract file entries must be objects")
        files.append(_asset_file_from_mapping(item, strict=strict))
    relative_paths = [item.relative_path for item in files]
    if len(relative_paths) != len(set(relative_paths)):
        raise ProcessingError("duplicate relative path in asset contract artifact files")
    if strict and status == "available" and not files:
        raise ProcessingError(f"asset lane {lane} is available but publishes no exact files")
    if status in EMPTY_ASSET_STATUSES and files:
        raise ProcessingError(f"asset lane {lane} is {status} but publishes files")
    return AssetArtifact(
        lane=lane,
        run_id=str(value.get("run_id") or ""),
        job_id=str(value.get("job_id") or ""),
        status=status,
        output_uri=str(value.get("output_uri") or ""),
        artifact_uri=str(value.get("artifact_uri") or ""),
        manifest_uri=str(value.get("manifest_uri") or ""),
        checksum_sha256=str(value.get("checksum_sha256") or ""),
        files=tuple(files),
        metadata=dict(value.get("metadata") or {}),
    )


def _asset_from_mapping(value: Mapping[str, Any], *, strict: bool = False) -> AssetRecord:
    try:
        segment = int(value["segment"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProcessingError("asset contract field must be an integer: segment") from exc
    if segment < 0:
        raise ProcessingError("asset contract field must be non-negative: segment")

    raw_artifacts = value.get("artifacts") or []
    if not isinstance(raw_artifacts, list):
        raise ProcessingError("asset contract field must be a list: artifacts")
    artifacts = tuple(_asset_artifact_from_mapping(item, strict=strict) for item in raw_artifacts)
    if strict:
        lane_names = [item.lane for item in artifacts]
        if len(lane_names) != len(set(lane_names)):
            raise ProcessingError("asset contract publishes duplicate resolved lane states")
        missing_lanes = sorted(SUPPORTED_ASSET_LANES - set(lane_names))
        if missing_lanes:
            raise ProcessingError(f"asset contract is missing required lane states: {', '.join(missing_lanes)}")
    return AssetRecord(
        asset_id=_required_text(value.get("asset_id"), field_name="asset_id"),
        segment=segment,
        source_uri=_required_text(value.get("source_uri"), field_name="source_uri"),
        artifacts=artifacts,
        provenance=dict(value.get("provenance") or {}),
        timebase=dict(value.get("timebase") or {}),
        metadata=dict(value.get("metadata") or {}),
    )


def _asset_page_from_mapping(value: Mapping[str, Any]) -> AssetPage:
    if value.get("schema_version") != ASSET_LIST_CONTRACT_VERSION:
        raise ProcessingError(
            f"unsupported asset list contract version: {value.get('schema_version') or '<missing>'}; "
            f"expected {ASSET_LIST_CONTRACT_VERSION}"
        )
    raw_assets = value.get("assets")
    if not isinstance(raw_assets, list):
        raise ProcessingError("asset list contract field must be a list: assets")
    assets: list[AssetRecord] = []
    seen_ids: set[str] = set()
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, Mapping):
            raise ProcessingError("asset list entries must be objects")
        asset = _asset_from_mapping(raw_asset, strict=True)
        if asset.asset_id in seen_ids:
            raise ProcessingError(f"duplicate asset_id in asset list contract: {asset.asset_id}")
        seen_ids.add(asset.asset_id)
        assets.append(asset)
    raw_cursor = value.get("next_cursor")
    if raw_cursor is not None and not isinstance(raw_cursor, str):
        raise ProcessingError("asset list next_cursor must be a string or null")
    return AssetPage(assets=tuple(assets), next_cursor=raw_cursor or None)


def _process_receipt_from_mapping(value: Mapping[str, Any], *, expected_asset_id: str) -> ProcessReceipt:
    if value.get("schema_version") != PROCESS_RECEIPT_CONTRACT_VERSION:
        raise ProcessingError(
            f"unsupported process receipt contract version: {value.get('schema_version') or '<missing>'}; "
            f"expected {PROCESS_RECEIPT_CONTRACT_VERSION}"
        )
    asset_id = _required_text(value.get("asset_id"), field_name="asset_id")
    if asset_id != expected_asset_id:
        raise ProcessingError(f"process receipt asset_id mismatch: expected {expected_asset_id}, got {asset_id}")
    request_id = _required_text(value.get("request_id"), field_name="request_id")
    raw_lanes = value.get("lanes")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        raise ProcessingError("process receipt lanes must be a non-empty list")
    lanes: list[LaneProcessReceipt] = []
    seen_lanes: set[str] = set()
    for raw_lane in raw_lanes:
        if not isinstance(raw_lane, Mapping):
            raise ProcessingError("process receipt lane entries must be objects")
        lane = _required_text(raw_lane.get("lane"), field_name="lanes[].lane").lower()
        if lane not in SUPPORTED_ASSET_LANES:
            raise ProcessingError(f"unsupported process receipt lane: {lane}")
        if lane in seen_lanes:
            raise ProcessingError(f"duplicate process receipt lane: {lane}")
        seen_lanes.add(lane)
        state = _required_text(raw_lane.get("state"), field_name="lanes[].state")
        if state not in PROCESS_RECEIPT_STATES:
            raise ProcessingError(f"unsupported process receipt state: {state}")
        lanes.append(
            LaneProcessReceipt(
                lane=lane,
                state=state,
                run_id=str(raw_lane.get("run_id") or ""),
                job_id=str(raw_lane.get("job_id") or ""),
                message=str(raw_lane.get("message") or ""),
            )
        )
    return ProcessReceipt(request_id=request_id, asset_id=asset_id, lanes=tuple(lanes))


def _episode_lane_from_mapping(value: Mapping[str, Any]) -> EpisodeLane:
    lane = _required_text(value.get("lane"), field_name="lanes[].lane").lower()
    status = _required_text(value.get("status"), field_name="lanes[].status").lower()
    if status not in EPISODE_LANE_STATUSES:
        choices = ", ".join(sorted(EPISODE_LANE_STATUSES))
        raise ProcessingError(f"unsupported episode lane status {status!r}; expected one of: {choices}")
    raw_files = value.get("files") or []
    if not isinstance(raw_files, list):
        raise ProcessingError("episode contract field must be a list: lanes[].files")
    files: list[AssetFile] = []
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise ProcessingError("episode contract file entries must be objects")
        files.append(_asset_file_from_mapping(item))
    relative_paths = [item.relative_path for item in files]
    if len(relative_paths) != len(set(relative_paths)):
        raise ProcessingError("duplicate relative path in episode lane files")
    if status == "available" and not files:
        raise ProcessingError(f"episode lane {lane} is {status} but publishes no exact files")
    if status in {"not_processed", "failed"} and files:
        raise ProcessingError(f"episode lane {lane} is {status} but publishes files")
    provenance = dict(value.get("provenance") or {})
    return EpisodeLane(
        lane=lane,
        status=status,
        files=tuple(files),
        run_id=str(value.get("run_id") or provenance.get("run_id") or ""),
        job_id=str(value.get("job_id") or provenance.get("job_id") or ""),
        message=str(value.get("message") or value.get("reason") or value.get("error") or ""),
        metadata=dict(value.get("metadata") or {}),
        provenance=provenance,
    )


def _episode_from_mapping(value: Mapping[str, Any]) -> EpisodeRecord:
    raw_interval = value.get("interval")
    if not isinstance(raw_interval, Mapping):
        raise ProcessingError("episode contract field must be an object: interval")
    interval = dict(raw_interval)
    try:
        raw_start_ns = interval["start_ns"]
        raw_end_ns = interval["end_ns"]
    except KeyError as exc:
        raise ProcessingError("episode contract interval bounds must be integer nanoseconds") from exc
    if str(interval.get("clock") or "").lower() != "sensor_ns":
        raise ProcessingError("episode contract interval clock must be sensor_ns")
    if str(interval.get("bounds") or "") != "[start_ns,end_ns)":
        raise ProcessingError("episode contract interval bounds must be [start_ns,end_ns)")

    asset_id = _required_text(value.get("asset_id"), field_name="asset_id")
    episode_id = _required_text(value.get("episode_id"), field_name="episode_id")
    expected_episode_id = _expected_episode_id(asset_id=asset_id, start_ns=raw_start_ns, end_ns=raw_end_ns)
    start_ns = int(raw_start_ns)
    end_ns = int(raw_end_ns)
    if episode_id != expected_episode_id:
        raise ProcessingError(
            f"episode_id does not match asset_id and canonical bounds: expected {expected_episode_id}, got {episode_id}"
        )

    raw_lanes = value.get("lanes") or []
    if not isinstance(raw_lanes, list):
        raise ProcessingError("episode contract field must be a list: lanes")
    lanes: list[EpisodeLane] = []
    for item in raw_lanes:
        if not isinstance(item, Mapping):
            raise ProcessingError("episode contract lane entries must be objects")
        lanes.append(_episode_lane_from_mapping(item))
    if not lanes:
        raise ProcessingError(f"episode {episode_id} publishes no lane states")
    lane_names = [item.lane for item in lanes]
    if len(lane_names) != len(set(lane_names)):
        raise ProcessingError(f"episode {episode_id} publishes duplicate lane states")
    missing_lanes = sorted(REQUIRED_EPISODE_LANES - set(lane_names))
    if missing_lanes:
        raise ProcessingError(f"episode {episode_id} is missing required lane states: {', '.join(missing_lanes)}")

    raw_segment = value.get("segment")
    try:
        segment = int(raw_segment) if raw_segment is not None else None
    except (TypeError, ValueError) as exc:
        raise ProcessingError("episode contract field must be an integer: segment") from exc
    if segment is not None and segment < 0:
        raise ProcessingError("episode contract field must be non-negative: segment")

    return EpisodeRecord(
        episode_id=episode_id,
        asset_id=asset_id,
        start_ns=start_ns,
        end_ns=end_ns,
        lanes=tuple(lanes),
        segment=segment,
        legacy_key=str(value.get("legacy_key") or ""),
        caption=str(value.get("caption") or ""),
        activity=str(value.get("activity") or ""),
        interval=interval,
        timebase=dict(value.get("timebase") or {}),
        provenance=dict(value.get("provenance") or {}),
        metadata=dict(value.get("metadata") or {}),
    )


class JsonAssetResolver:
    """Read a local asset manifest."""

    def __init__(self, document: str | Path | Mapping[str, Any], *, strict: bool = False) -> None:
        if isinstance(document, (str, Path)):
            with Path(document).expanduser().open(encoding="utf-8") as stream:
                payload = json.load(stream)
        else:
            payload = dict(document)

        version = str(payload.get("schema_version") or "")
        if version != ASSET_CONTRACT_VERSION:
            raise ProcessingError(
                f"unsupported asset contract version: {version or '<missing>'}; expected {ASSET_CONTRACT_VERSION}"
            )

        if isinstance(payload.get("asset"), Mapping):
            raw_assets = [payload["asset"]]
        else:
            raw_assets = payload.get("assets") or []
        if not isinstance(raw_assets, list):
            raise ProcessingError("asset contract field must be a list: assets")

        self._assets: dict[str, AssetRecord] = {}
        for raw_asset in raw_assets:
            if not isinstance(raw_asset, Mapping):
                raise ProcessingError("asset contract entries must be objects")
            asset = _asset_from_mapping(raw_asset, strict=strict)
            if asset.asset_id in self._assets:
                raise ProcessingError(f"duplicate asset_id in asset contract: {asset.asset_id}")
            self._assets[asset.asset_id] = asset

    def asset(self, asset_id: str) -> AssetRecord:
        try:
            return self._assets[asset_id]
        except KeyError as exc:
            raise AssetNotFoundError(f"asset not found: {asset_id}") from exc

    def assets(self) -> list[AssetRecord]:
        return [self._assets[key] for key in sorted(self._assets)]


class JsonEpisodeResolver:
    """Read a versioned producer episode handoff document."""

    def __init__(self, document: str | Path | Mapping[str, Any]) -> None:
        if isinstance(document, (str, Path)):
            with Path(document).expanduser().open(encoding="utf-8") as stream:
                payload = json.load(stream)
        else:
            payload = dict(document)

        version = str(payload.get("schema_version") or "")
        if version != EPISODE_CONTRACT_VERSION:
            raise ProcessingError(
                f"unsupported episode contract version: {version or '<missing>'}; expected {EPISODE_CONTRACT_VERSION}"
            )

        if isinstance(payload.get("episode"), Mapping):
            raw_episodes = [payload["episode"]]
        else:
            raw_episodes = payload.get("episodes") or []
        if not isinstance(raw_episodes, list):
            raise ProcessingError("episode contract field must be a list: episodes")

        self._episodes: dict[str, EpisodeRecord] = {}
        self._episode_order: list[str] = []
        for raw_episode in raw_episodes:
            if not isinstance(raw_episode, Mapping):
                raise ProcessingError("episode contract entries must be objects")
            episode = _episode_from_mapping(raw_episode)
            if episode.episode_id in self._episodes:
                raise ProcessingError(f"duplicate episode_id in episode contract: {episode.episode_id}")
            self._episodes[episode.episode_id] = episode
            self._episode_order.append(episode.episode_id)

    def episode(self, episode_id: str) -> EpisodeRecord:
        try:
            return self._episodes[episode_id]
        except KeyError as exc:
            raise EpisodeNotFoundError(f"episode not found: {episode_id}") from exc

    def episodes(self) -> list[EpisodeRecord]:
        return [self._episodes[key] for key in self._episode_order]


class HttpAssetResolver:
    """Resolve the public ``GET /v1/assets/{asset_id}`` contract."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: Optional[str] = None,
        timeout_seconds: float = 30.0,
        request_opener: Any = None,
    ) -> None:
        self.base_url = _required_text(base_url, field_name="base_url").rstrip("/")
        if urllib.parse.urlparse(self.base_url).scheme not in {"http", "https"}:
            raise ProcessingError("asset API base_url must use http or https")
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        self._open = request_opener or urllib.request.urlopen

    def _request_json(self, request: urllib.request.Request, *, context: str) -> tuple[Mapping[str, Any], int]:
        try:
            with self._open(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
                status_code = int(getattr(response, "status", 0) or 0)
        except urllib.error.HTTPError as exc:
            code = ""
            message = f"asset API returned HTTP {exc.code}"
            request_id = ""
            try:
                error_payload = json.load(exc)
                if isinstance(error_payload, Mapping) and error_payload.get("schema_version") == ERROR_CONTRACT_VERSION:
                    error = error_payload.get("error")
                    if isinstance(error, Mapping):
                        code = str(error.get("code") or "")
                        message = str(error.get("message") or message)
                        request_id = str(error.get("request_id") or "")
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                pass
            suffix = f" (request_id={request_id})" if request_id else ""
            if exc.code == 404 or code == "asset_not_found":
                raise AssetNotFoundError(f"{context}: {message}{suffix}") from exc
            raise AssetApiError(
                f"{context}: {message}{suffix}",
                status_code=exc.code,
                code=code,
                request_id=request_id,
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProcessingError(f"{context}: asset API request failed: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ProcessingError(f"{context}: asset API response must be a JSON object")
        return payload, status_code

    def _request(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: Optional[Mapping[str, Any]] = None,
    ) -> urllib.request.Request:
        headers = {"Accept": "application/json", "User-Agent": "grounded-python-sdk"}
        body = None
        if payload is not None:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        if self.bearer_token:
            request.add_header("Authorization", f"Bearer {self.bearer_token}")
        return request

    def asset(self, asset_id: str) -> AssetRecord:
        normalized_asset_id = _required_text(asset_id, field_name="asset_id")
        quoted_id = urllib.parse.quote(normalized_asset_id, safe="")
        payload, _ = self._request_json(
            self._request(f"{self.base_url}/v1/assets/{quoted_id}"),
            context=f"get asset {normalized_asset_id}",
        )
        return JsonAssetResolver(payload, strict=True).asset(normalized_asset_id)

    def list_assets(
        self,
        *,
        lane: Optional[str] = None,
        status: Optional[str] = None,
        updated_after: Optional[str] = None,
        updated_before: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> AssetPage:
        """Return one API page without inferring assets from storage prefixes."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ProcessingError("asset list limit must be an integer from 1 through 200")
        query: dict[str, str] = {"limit": str(limit)}
        for key, value in (
            ("lane", lane),
            ("status", status),
            ("updated_after", updated_after),
            ("updated_before", updated_before),
            ("cursor", cursor),
        ):
            if value is not None:
                query[key] = _required_text(value, field_name=key)
        url = f"{self.base_url}/v1/assets?{urllib.parse.urlencode(query)}"
        payload, _ = self._request_json(self._request(url), context="list assets")
        return _asset_page_from_mapping(payload)

    def process_asset(
        self,
        asset_id: str,
        *,
        lanes: list[str] | tuple[str, ...],
        idempotency_key: Optional[str] = None,
        retry_failed: bool = False,
        rerun_available: bool = False,
    ) -> ProcessReceipt:
        """Submit lane work while retaining run/job IDs as receipt lineage only."""

        normalized_asset_id = _required_text(asset_id, field_name="asset_id")
        normalized_lanes: list[str] = []
        for raw_lane in lanes:
            lane = _required_text(raw_lane, field_name="lanes[]").lower()
            if lane not in SUPPORTED_ASSET_LANES:
                raise ProcessingError(f"unsupported asset lane: {lane}")
            if lane not in normalized_lanes:
                normalized_lanes.append(lane)
        if not normalized_lanes:
            raise ProcessingError("at least one process lane is required")
        request_payload: dict[str, Any] = {
            "schema_version": PROCESS_REQUEST_CONTRACT_VERSION,
            "lanes": normalized_lanes,
            "retry_failed": bool(retry_failed),
            "rerun_available": bool(rerun_available),
        }
        if idempotency_key is not None:
            request_payload["idempotency_key"] = _required_text(idempotency_key, field_name="idempotency_key")
        quoted_id = urllib.parse.quote(normalized_asset_id, safe="")
        payload, _ = self._request_json(
            self._request(
                f"{self.base_url}/v1/assets/{quoted_id}/process",
                method="POST",
                payload=request_payload,
            ),
            context=f"process asset {normalized_asset_id}",
        )
        return _process_receipt_from_mapping(payload, expected_asset_id=normalized_asset_id)


class HttpEpisodeResolver:
    """Resolve the public ``GET /v1/episodes/{episode_id}`` contract."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: Optional[str] = None,
        timeout_seconds: float = 30.0,
        request_opener: Any = None,
    ) -> None:
        self.base_url = _required_text(base_url, field_name="base_url").rstrip("/")
        if urllib.parse.urlparse(self.base_url).scheme not in {"http", "https"}:
            raise ProcessingError("episode API base_url must use http or https")
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        self._open = request_opener or urllib.request.urlopen

    def episode(self, episode_id: str) -> EpisodeRecord:
        quoted_id = urllib.parse.quote(_required_text(episode_id, field_name="episode_id"), safe="")
        request = urllib.request.Request(
            f"{self.base_url}/v1/episodes/{quoted_id}",
            headers={"Accept": "application/json", "User-Agent": "grounded-python-sdk"},
            method="GET",
        )
        if self.bearer_token:
            request.add_header("Authorization", f"Bearer {self.bearer_token}")
        try:
            with self._open(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise EpisodeNotFoundError(f"episode not found: {episode_id}") from exc
            raise ProcessingError(f"episode API returned HTTP {exc.code} for {episode_id}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProcessingError(f"episode API request failed for {episode_id}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ProcessingError("episode API response must be a JSON object")
        return JsonEpisodeResolver(payload).episode(episode_id)


class DefaultArtifactTransport:
    """Download exact S3 or HTTP(S) objects without listing prefixes."""

    def __init__(self, *, aws_profile: Optional[str] = None, session: Any = None) -> None:
        self.aws_profile = aws_profile
        self._session = session

    def download(self, uri: str, target: Path) -> None:
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme == "s3":
            session = self._session
            if session is None:
                import boto3

                session = boto3.Session(profile_name=self.aws_profile)
                self._session = session
            session.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), str(target))
            return
        if parsed.scheme in {"http", "https"}:
            with urllib.request.urlopen(uri) as response, target.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            return
        raise ProcessingError(f"unsupported artifact URI scheme: {parsed.scheme or '<missing>'}")


class ProcessingClient:
    """Resolve and download segment assets and multimodal episodes."""

    def __init__(
        self,
        *,
        asset_resolver: Optional[AssetResolver] = None,
        episode_resolver: Optional[EpisodeResolver] = None,
        aws_profile: Optional[str] = None,
    ) -> None:
        self.asset_resolver = asset_resolver
        self.episode_resolver = episode_resolver
        self.aws_profile = aws_profile

    @classmethod
    def from_api(
        cls,
        base_url: str,
        *,
        bearer_token: Optional[str] = None,
        timeout_seconds: float = 30.0,
        aws_profile: Optional[str] = None,
    ) -> ProcessingClient:
        """Create a client for a deployed Grounded HTTP API."""

        return cls(
            asset_resolver=HttpAssetResolver(
                base_url,
                bearer_token=bearer_token,
                timeout_seconds=timeout_seconds,
            ),
            episode_resolver=HttpEpisodeResolver(
                base_url,
                bearer_token=bearer_token,
                timeout_seconds=timeout_seconds,
            ),
            aws_profile=aws_profile,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path,
        *,
        aws_profile: Optional[str] = None,
    ) -> ProcessingClient:
        """Create an offline client from an asset or episode manifest."""

        path = Path(manifest).expanduser()
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProcessingError(f"cannot read manifest {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ProcessingError("manifest must contain a JSON object")
        version = payload.get("schema_version")
        if version == ASSET_CONTRACT_VERSION:
            return cls(asset_resolver=JsonAssetResolver(payload), aws_profile=aws_profile)
        if version == EPISODE_CONTRACT_VERSION:
            return cls(episode_resolver=JsonEpisodeResolver(payload), aws_profile=aws_profile)
        raise ProcessingError(f"unsupported manifest schema: {version or '<missing>'}")

    def get_asset(self, asset_id: str) -> AssetRecord:
        """Resolve one immutable segment and its processing lane states."""

        if self.asset_resolver is None:
            raise ProcessingError("no asset resolver configured")
        return self.asset_resolver.asset(asset_id)

    def list_assets(
        self,
        *,
        lane: Optional[str] = None,
        status: Optional[str] = None,
        updated_after: Optional[str] = None,
        updated_before: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> AssetPage:
        """List assets through a deployed HTTP API."""

        if not isinstance(self.asset_resolver, HttpAssetResolver):
            raise ProcessingError("asset listing requires an HTTP asset resolver")
        return self.asset_resolver.list_assets(
            lane=lane,
            status=status,
            updated_after=updated_after,
            updated_before=updated_before,
            cursor=cursor,
            limit=limit,
        )

    def process_asset(
        self,
        asset_id: str,
        *,
        lanes: list[str] | tuple[str, ...],
        idempotency_key: Optional[str] = None,
        retry_failed: bool = False,
        rerun_available: bool = False,
    ) -> ProcessReceipt:
        """Request processing through a deployed HTTP API."""

        if not isinstance(self.asset_resolver, HttpAssetResolver):
            raise ProcessingError("processing submission requires an HTTP asset resolver")
        return self.asset_resolver.process_asset(
            asset_id,
            lanes=lanes,
            idempotency_key=idempotency_key,
            retry_failed=retry_failed,
            rerun_available=rerun_available,
        )

    def list_episodes(self) -> list[EpisodeRecord]:
        """List episodes from a local manifest."""

        if not isinstance(self.episode_resolver, JsonEpisodeResolver):
            raise ProcessingError("episode listing requires a local episode manifest")
        return self.episode_resolver.episodes()

    def get_episode(self, episode_id: str) -> EpisodeRecord:
        """Resolve one canonical interval and all of its lane states."""

        if self.episode_resolver is None:
            raise ProcessingError("no episode resolver configured")
        return self.episode_resolver.episode(episode_id)

    def list_episode_lanes(self, episode_id: str) -> list[EpisodeLane]:
        """List every producer-published lane state for an episode."""

        return sorted(self.get_episode(episode_id).lanes, key=lambda item: item.lane)

    def resolve_episode_lane(self, episode_id: str, *, lane: str) -> EpisodeLane:
        """Resolve one published lane state without selecting by run/job lineage."""

        normalized_lane = _required_text(lane, field_name="lane").lower()
        for item in self.list_episode_lanes(episode_id):
            if item.lane == normalized_lane:
                return item
        raise EpisodeLaneNotFoundError(f"episode {episode_id} publishes no {normalized_lane} lane state")

    def download_episode(
        self,
        episode_id: str,
        *,
        lane: Optional[str] = None,
        target_dir: str = DEFAULT_ASSET_CACHE,
        transport: Optional[ArtifactTransport] = None,
        require_sha256: bool = False,
    ) -> EpisodeDownload:
        """Download exact clipped files and preserve every selected lane state.

        With no ``lane``, all published lanes are returned and every exact file
        from ``available`` or ``partial`` lanes is downloaded. ``not_processed``
        and ``failed`` lanes remain visible with empty file lists. A requested
        unavailable lane is likewise returned without falling back to an asset
        artifact or creating a placeholder file.
        """

        episode = self.get_episode(episode_id)
        selected = [self.resolve_episode_lane(episode_id, lane=lane)] if lane else self.list_episode_lanes(episode_id)
        cache_root = Path(target_dir).expanduser() / "episodes" / _cache_component(episode_id, visible_chars=32)
        downloader = transport or DefaultArtifactTransport(aws_profile=self.aws_profile)
        lane_results: list[EpisodeLaneDownload] = []
        all_files: list[DownloadedEpisodeFile] = []

        for lane_state in selected:
            downloaded: list[DownloadedEpisodeFile] = []
            if lane_state.status in DOWNLOADABLE_EPISODE_LANE_STATUSES:
                lane_root = cache_root / _cache_component(lane_state.lane, visible_chars=16)
                for episode_file in lane_state.files:
                    destination = lane_root / Path(episode_file.relative_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    expected_sha = episode_file.sha256
                    if require_sha256 and not expected_sha:
                        raise ProcessingError(
                            f"episode {episode_id} {lane_state.lane} file "
                            f"{episode_file.relative_path} has no published SHA-256"
                        )
                    if not _local_file_matches(
                        destination,
                        size_bytes=episode_file.size_bytes,
                        sha256=expected_sha,
                    ):
                        temporary = destination.with_name(f".{destination.name}.part")
                        if temporary.exists():
                            temporary.unlink()
                        try:
                            downloader.download(episode_file.uri, temporary)
                            _validate_download(
                                temporary,
                                size_bytes=episode_file.size_bytes,
                                sha256=expected_sha,
                            )
                            temporary.replace(destination)
                        finally:
                            if temporary.exists():
                                temporary.unlink()
                    result_file = DownloadedEpisodeFile(
                        episode_id=episode_id,
                        asset_id=episode.asset_id,
                        lane=lane_state.lane,
                        source_uri=episode_file.uri,
                        local_path=str(destination),
                        size_bytes=destination.stat().st_size,
                        sha256=_sha256_file(destination),
                        size_verified=episode_file.size_bytes is not None,
                        sha256_verified=bool(expected_sha),
                    )
                    downloaded.append(result_file)
                    all_files.append(result_file)
            lane_results.append(
                EpisodeLaneDownload(
                    lane=lane_state.lane,
                    status=lane_state.status,
                    files=tuple(downloaded),
                    run_id=lane_state.run_id,
                    job_id=lane_state.job_id,
                    message=lane_state.message,
                )
            )

        return EpisodeDownload(
            episode_id=episode.episode_id,
            asset_id=episode.asset_id,
            start_ns=episode.start_ns,
            end_ns=episode.end_ns,
            root_dir=str(cache_root),
            lanes=tuple(lane_results),
            files=tuple(all_files),
        )

    def list_asset_artifacts(self, asset_id: str, *, lane: Optional[str] = None) -> list[AssetArtifact]:
        """List resolved lane states for a segment-level asset."""

        asset = self.get_asset(asset_id)
        normalized_lane = lane.lower() if lane else None
        return sorted(
            (artifact for artifact in asset.artifacts if normalized_lane is None or artifact.lane == normalized_lane),
            key=lambda item: (item.lane, item.run_id, item.job_id),
        )

    def resolve_asset_artifact(
        self,
        asset_id: str,
        *,
        lane: str,
    ) -> AssetArtifact:
        """Resolve one lane state without treating run/job lineage as identity."""

        normalized_lane = lane.strip().lower()
        matches = [artifact for artifact in self.list_asset_artifacts(asset_id, lane=normalized_lane)]
        if not matches:
            raise AssetArtifactNotFoundError(f"asset {asset_id} has no artifact matching lane={normalized_lane}")
        if len(matches) > 1:
            choices = ", ".join(f"run_id={item.run_id} job_id={item.job_id}" for item in matches)
            raise AmbiguousAssetArtifactError(
                f"asset {asset_id} has multiple resolved {normalized_lane} states; producer must publish one: {choices}"
            )
        artifact = matches[0]
        if artifact.status.lower() not in DOWNLOADABLE_ASSET_STATUSES:
            raise AssetArtifactNotReadyError(
                f"asset {asset_id} {normalized_lane} run {artifact.run_id} is {artifact.status}; output is not ready"
            )
        return artifact

    def download_asset(
        self,
        asset_id: str,
        *,
        lanes: Optional[list[str]] = None,
        target_dir: str = DEFAULT_ASSET_CACHE,
        transport: Optional[ArtifactTransport] = None,
        require_sha256: bool = False,
    ) -> AssetDownload:
        """Download every exact object selected for an asset's completed lanes.

        The resolver/API must provide an explicit file list or one exact
        ``artifact_uri`` per lane. This method never lists or recursively
        downloads a shared output prefix.
        """

        asset = self.get_asset(asset_id)
        selected_lanes = (
            sorted({lane.strip().lower() for lane in lanes if lane.strip()})
            if lanes
            else sorted(
                {
                    artifact.lane
                    for artifact in asset.artifacts
                    if artifact.status.lower() in DOWNLOADABLE_ASSET_STATUSES and (artifact.files or artifact.artifact_uri)
                }
            )
        )
        if not selected_lanes:
            raise AssetArtifactNotFoundError(f"asset {asset_id} has no downloadable lanes")

        chosen: list[AssetArtifact] = []
        for lane in selected_lanes:
            chosen.append(self.resolve_asset_artifact(asset_id, lane=lane))

        cache_root = Path(target_dir).expanduser() / "assets" / _cache_component(asset_id, visible_chars=32)
        downloader = transport or DefaultArtifactTransport(aws_profile=self.aws_profile)
        downloaded: list[DownloadedAssetFile] = []
        for artifact in chosen:
            files = artifact.files or _single_artifact_file(artifact)
            if not files:
                raise ProcessingError(f"asset {asset_id} {artifact.lane} run {artifact.run_id} has no exact artifact files")
            for asset_file in files:
                destination = cache_root / _cache_component(artifact.lane, visible_chars=16) / Path(asset_file.relative_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                expected_sha = asset_file.sha256 or (artifact.checksum_sha256 if len(files) == 1 else "")
                if require_sha256 and not expected_sha:
                    raise ProcessingError(
                        f"asset {asset_id} {artifact.lane} file {asset_file.relative_path} has no published SHA-256"
                    )
                if not _local_file_matches(destination, size_bytes=asset_file.size_bytes, sha256=expected_sha):
                    temporary = destination.with_name(f".{destination.name}.part")
                    if temporary.exists():
                        temporary.unlink()
                    try:
                        downloader.download(asset_file.uri, temporary)
                        _validate_download(temporary, size_bytes=asset_file.size_bytes, sha256=expected_sha)
                        temporary.replace(destination)
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                downloaded.append(
                    DownloadedAssetFile(
                        asset_id=asset_id,
                        lane=artifact.lane,
                        run_id=artifact.run_id,
                        source_uri=asset_file.uri,
                        local_path=str(destination),
                        size_bytes=destination.stat().st_size,
                        sha256=_sha256_file(destination),
                        size_verified=asset_file.size_bytes is not None,
                        sha256_verified=bool(expected_sha),
                    )
                )
        return AssetDownload(asset_id=asset_id, root_dir=str(cache_root), files=tuple(downloaded))

    def open_hand_asset(
        self,
        asset_id: str,
        *,
        target_dir: str = DEFAULT_ASSET_CACHE,
        active_cameras: Optional[list[str]] = None,
        valid_only: bool = False,
        transport: Optional[ArtifactTransport] = None,
        require_sha256: bool = False,
    ) -> Any:
        """Download and open the full Hand recording for one segment asset."""

        from grounded.data.ego_dataset import HandEpisode

        asset = self.get_asset(asset_id)
        download = self.download_asset(
            asset_id,
            lanes=["hand"],
            target_dir=target_dir,
            transport=transport,
            require_sha256=require_sha256,
        )
        return HandEpisode.from_asset_download(
            download,
            segment=asset.segment,
            active_cameras=active_cameras,
            valid_only=valid_only,
        )

    def open_hand_episode(
        self,
        episode_id: str,
        *,
        target_dir: str = DEFAULT_ASSET_CACHE,
        active_cameras: Optional[list[str]] = None,
        valid_only: bool = False,
        transport: Optional[ArtifactTransport] = None,
        require_sha256: bool = False,
    ) -> Any:
        """Download and open one clipped Hand episode, including its caption."""

        from grounded.data.ego_dataset import ClippedHandEpisode

        record = self.get_episode(episode_id)
        download = self.download_episode(
            episode_id,
            lane="hand",
            target_dir=target_dir,
            transport=transport,
            require_sha256=require_sha256,
        )
        return ClippedHandEpisode.from_download(
            download,
            active_cameras=active_cameras,
            valid_only=valid_only,
            caption=record.caption or None,
        )

    def open_hand(
        self,
        asset_or_episode_id: str,
        *,
        target_dir: str = DEFAULT_ASSET_CACHE,
        active_cameras: Optional[list[str]] = None,
        valid_only: bool = False,
        transport: Optional[ArtifactTransport] = None,
        require_sha256: bool = False,
    ) -> Any:
        """Open a full Hand asset or clipped episode using the configured manifest/API."""

        identifier = _required_text(asset_or_episode_id, field_name="asset_or_episode_id")
        common = {
            "target_dir": target_dir,
            "active_cameras": active_cameras,
            "valid_only": valid_only,
            "transport": transport,
            "require_sha256": require_sha256,
        }
        if self.episode_resolver is not None and self.asset_resolver is None:
            return self.open_hand_episode(identifier, **common)
        if self.asset_resolver is not None and self.episode_resolver is None:
            return self.open_hand_asset(identifier, **common)
        if identifier.startswith("ep_v1_"):
            return self.open_hand_episode(identifier, **common)
        if identifier.startswith("ast_v1_"):
            return self.open_hand_asset(identifier, **common)
        raise ProcessingError(
            "cannot determine whether the identifier is an asset or episode; "
            "use open_hand_asset(...) or open_hand_episode(...)"
        )


def _single_artifact_file(artifact: AssetArtifact) -> tuple[AssetFile, ...]:
    if not artifact.artifact_uri:
        return ()
    parsed = urllib.parse.urlparse(artifact.artifact_uri)
    name = Path(parsed.path).name or f"{artifact.lane}-artifact"
    size = artifact.metadata.get("size_bytes")
    try:
        size_bytes = int(size) if size is not None else None
    except (TypeError, ValueError) as exc:
        raise ProcessingError(f"invalid artifact size for {artifact.lane} run {artifact.run_id}") from exc
    return (
        AssetFile(
            relative_path=_safe_relative_path(name, field_name="artifact_uri filename"),
            uri=artifact.artifact_uri,
            size_bytes=size_bytes,
            sha256=artifact.checksum_sha256,
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_file_matches(path: Path, *, size_bytes: Optional[int], sha256: str) -> bool:
    if not path.is_file():
        return False
    if size_bytes is None and not sha256:
        return False
    if size_bytes is not None and path.stat().st_size != size_bytes:
        return False
    return not sha256 or _sha256_file(path) == sha256.removeprefix("sha256:").lower()


def _validate_download(path: Path, *, size_bytes: Optional[int], sha256: str) -> None:
    if not path.is_file():
        raise ProcessingError(f"artifact transport did not create a file: {path}")
    if size_bytes is not None and path.stat().st_size != size_bytes:
        raise ProcessingError(f"artifact size mismatch for {path.name}: expected {size_bytes}, got {path.stat().st_size}")
    if sha256:
        expected = sha256.removeprefix("sha256:").lower()
        actual = _sha256_file(path)
        if actual != expected:
            raise ProcessingError(f"artifact SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")
