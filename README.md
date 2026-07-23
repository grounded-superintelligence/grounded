```
 ______   _____    ______   _    _   _    _   _____    ______   _____          
/ _____/ |  __ \  /  __  \ | |  | | | \  | | | ___ \  |  ____| | ___ \         
| |__  | | |__) | | (__) | | |__| | |  \ | | | |__) | |  ___|  | |__) |        
\______/ |_|  \_\ \______/  \____/  |_|  \_| |_____/  |______| |_____/        

 _______  _    _   ______   ______   _____    ______   _    _   ______   ______ 
/ _____/ | |  | | |  ___ \ |  ____| |  __ \  |__  __| | \  | | |__  __| |  ____|
\_____ \ | |__| | | |__/ / |  ___|  | |__) |   |  |   |  \ | |   |  |   |  ___| 
/______/  \____/  | |      |______| |_|  \_\ |______| |_|  \_|   |__|   |______|

          _        _        ______   ______   ______   _    _   ______   ______ 
         | |      | |      |__  __| / _____/ |  ____| | \  | | / _____/ |  ____|
         | |____  | |____    |  |   | |__  | |  ___|  |  \ | | | |____  |  ___| 
         |______| |______| |______| \______/ |______| |_|  \_| \______\ |______|
```

## setup

Grounded requires Python 3.10 or newer. Install it from a source checkout:

```bash
python -m pip install .
# or
python -m pip install git+https://github.com/grounded-superintelligence/grounded.git
```

## usage

A delivery includes a JSON manifest plus access to the exact files it
references. No API server is required.

The quickest validation uses the existing Python demo:

```bash
python demo.py --manifest /path/to/manifest.json --episode 0
```

Both commands download every available enrichment lane, report unavailable or
partial lanes, and write an MP4 (camera grid with hand skeletons) and a Rerun
`.rrd` file (world-frame when the SLAM lane is available) under `outputs/`.
Downloads are cached under `~/.cache/grounded/data`.

Numeric episode indexes follow the row order in the supplied manifest.

If a legacy manifest does not include SHA-256 checksums, add
`--allow-missing-sha256`. New manifests should include checksums.

### manifest access

The SDK supports two delivery options:

- A manifest containing presigned HTTPS URLs works without AWS credentials.
  The URLs stop working at their stated expiration time.
- A manifest containing `s3://` URIs uses the normal AWS credential chain.
  Pass `--aws-profile PROFILE` to the demo when using a named local profile.

Credentials are never stored in the manifest or packaged in the SDK.

The same flow can be used directly from Python:

```python
# see demo.py for reference
from grounded.data.processing import ProcessingClient
from grounded.data.visualize_hand import visualize_hand_episode_to_mp4

client = ProcessingClient.from_manifest("manifest.json")
record = client.list_episodes()[0]

episode = client.open_hand(record.episode_id, active_cameras=["left_front"])
try:
    visualize_hand_episode_to_mp4(
        episode,
        "episode.mp4",
        caption=episode.caption,
    )
finally:
    episode.close()
```

To download every published enrichment for a full segment from an asset
manifest:

```python
client = ProcessingClient.from_manifest("assets.json")
download = client.download_asset("ast_v1_...")

# Or download selected lanes only.
hand = client.download_asset("ast_v1_...", lanes=["hand"])

# Open and visualize the full Hand segment.
episode = client.open_hand("ast_v1_...", active_cameras=["left_front"])
```

`ProcessingClient.from_api(...)` is available if Grounded later provides an
HTTP API URL. It is not required for manifest-based delivery. Storage access is
granted separately and is never embedded in the SDK.

See [`docs/ASSET_EPISODE_FLOW.md`](docs/ASSET_EPISODE_FLOW.md) for the complete
asset and episode control flow, and [`docs/DATA.md`](docs/DATA.md) for the
exact specification of every array the readers expose.
