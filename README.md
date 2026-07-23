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

Install the API and manifest client from a source checkout:

```bash
git clone https://github.com/grounded-superintelligence/grounded.git
cd grounded
git checkout v1.0.0
python -m pip install -e .
```

Install the visualization dependencies when rendering MP4 or Rerun files:

```bash
python -m pip install -e ".[visualization]"
```

The prepared PyPI distribution name is `grounded-sdk`. After its first public
release, install it with `python -m pip install grounded-sdk` or
`python -m pip install "grounded-sdk[visualization]"`.

## usage

### v1

A manifest can contain captioned episodes or full enriched segments. Render an
episode by its row number or `episode_id`:

```bash
python demo.py \
  --manifest /path/to/manifest.json \
  --episode 0
```

To render a full segment from an asset manifest:

```bash
python demo.py \
  --manifest /path/to/manifest.json \
  --asset-id ast_v1_... \
  --cameras left_front \
  --downsample 4 \
  --num-workers 4
```

The demo downloads the published enrichment files, reports missing or partial
lanes, and renders Hand tracking to MP4 and Rerun `.rrd` files under `outputs/`.
Downloads are cached under `~/.cache/grounded/data`.

Presigned manifests require no AWS credentials. Manifests containing `s3://`
URIs require the `s3` or `visualization` extra and use the normal AWS credential
chain; pass `--aws-profile PROFILE` when using a named profile. Add
`--allow-missing-sha256` only for legacy manifests without checksums.

The same client can connect to a hosted API:

```python
import os

from grounded.data.processing import ProcessingClient

client = ProcessingClient.from_api(
    os.environ["GROUNDED_API_URL"],
    bearer_token=os.environ["GROUNDED_API_TOKEN"],
)

assets = client.list_assets(lane="hand", status="available")
asset = client.get_asset(assets.assets[0].asset_id)
download = client.download_asset(asset.asset_id)

episode = client.get_episode("ep_v1_...")
hand_only = client.download_episode(episode.episode_id, lane="hand")
```

The API or manifest supplies exact files and reports each lane as available,
partial, not processed, or failed. Storage credentials are never packaged in
the SDK.

See [`docs/ASSET_EPISODE_FLOW.md`](docs/ASSET_EPISODE_FLOW.md) for the complete
asset, episode, download, and visualization flow. Refer to
[`docs/DATA.md`](docs/DATA.md) for the data structures exposed by the SDK.
