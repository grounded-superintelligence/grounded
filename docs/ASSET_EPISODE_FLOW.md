# Asset and episode flow

The SDK accepts either:

- an **asset manifest**, where `asset_id` identifies one full enriched segment; or
- an **episode manifest**, where `episode_id` identifies one captioned clip within a segment.

Both paths return a `HandEpisode`-compatible object and use the same visualizer.
No HTTP server is required when a manifest is supplied.

## Files and functions

| File | Function or class | Purpose |
| --- | --- | --- |
| `src/grounded/processing.py` | `ProcessingClient.from_manifest(...)` | Opens an asset or episode manifest. |
| `src/grounded/processing.py` | `download_asset(...)` | Downloads every available Hand, SLAM, and depth file for a full segment, or selected lanes. |
| `src/grounded/processing.py` | `download_episode(...)` | Downloads every available clipped lane for an episode, or one selected lane. |
| `src/grounded/processing.py` | `open_hand(...)` | Downloads and opens either a full Hand asset or a clipped Hand episode. |
| `src/grounded/data/hand_dataset.py` | `HandEpisode.from_asset_download(...)` | Safely extracts and opens a full-segment Hand archive. |
| `src/grounded/data/hand_dataset.py` | `ClippedHandEpisode.from_download(...)` | Opens the flat Hand files for one clipped episode. |
| `src/grounded/data/visualize_hand.py` | `visualize_hand_episode_to_mp4(...)` | Renders either reader to MP4. |

`processing.py` handles identity, manifest resolution, downloading, caching, and
integrity checks. `hand_dataset.py` turns downloaded Hand files into frame data
the visualizer can read.

## Control flow

```mermaid
flowchart TD
    manifest["Asset or episode manifest"] --> client["ProcessingClient.from_manifest"]
    client --> kind{"Identifier"}
    kind -->|"asset_id"| asset["download_asset hand lane"]
    kind -->|"episode_id"| episode["download_episode hand lane"]
    asset --> full["HandEpisode.from_asset_download"]
    episode --> clip["ClippedHandEpisode.from_download"]
    full --> visualizer["visualize_hand_episode_to_mp4"]
    clip --> visualizer
```

An `asset_id` renders the entire enriched segment. An `episode_id` renders only
the producer-published clip and carries its caption.

## Quick validation

After installing the SDK from a wheel or source checkout:

```bash
python demo_v1.py --manifest /path/to/manifest.json --episode 0
python demo_v1.py \
  --manifest /path/to/manifest.json \
  --asset-id ast_v1_... \
  --cameras left_front \
  --downsample 4 \
  --num-workers 4
```

The demo downloads every available Hand, SLAM, and depth lane before rendering
Hand. Unavailable lanes are reported without placeholder files. It writes an
MP4 and Rerun `.rrd` file under `outputs/`, and reuses downloads cached under
`~/.cache/grounded/data`. Numeric episode indexes follow manifest row order.

For a presigned manifest, no AWS credentials are required. For a manifest with
`s3://` file URIs, use the normal AWS credential chain or pass
`--aws-profile PROFILE`. Credentials are not stored in either manifest type.

Add `--allow-missing-sha256` only when validating a legacy manifest whose
producer did not publish checksums.

## Visualize a captioned episode

```python
from grounded.data.visualize_hand import visualize_hand_episode_to_mp4
from grounded.processing import ProcessingClient

client = ProcessingClient.from_manifest("episodes.json")
record = client.list_episodes()[0]
hand = client.open_hand(record.episode_id, active_cameras=["left_front"])

try:
    visualize_hand_episode_to_mp4(
        hand,
        "episode.mp4",
        caption=hand.caption,
    )
finally:
    hand.close()
```

## Visualize a full segment

```python
from grounded.data.visualize_hand import visualize_hand_episode_to_mp4
from grounded.processing import ProcessingClient

asset_id = "ast_v1_..."
client = ProcessingClient.from_manifest("assets.json")
hand = client.open_hand(asset_id, active_cameras=["left_front"])

try:
    visualize_hand_episode_to_mp4(hand, "full-segment.mp4")
finally:
    hand.close()
```

## Download all enrichment lanes

```python
# Every available full-segment lane.
asset_download = client.download_asset(asset_id)

# Every available clipped lane. Unavailable lanes remain visible in
# episode_download.lanes without fake files.
episode_download = client.download_episode(record.episode_id)
```

Pass `lanes=["hand"]` to `download_asset(...)` or `lane="hand"` to
`download_episode(...)` when only one lane is needed.
