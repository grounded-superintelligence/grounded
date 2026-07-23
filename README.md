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
release, the equivalent commands will be `python -m pip install grounded-sdk`
and `python -m pip install "grounded-sdk[visualization]"`.

## usage

### v1

A manifest can contain captioned episodes or full enriched segments. Render an
episode by its row number or `episode_id`:

```bash
python demo_v1.py \
  --manifest /path/to/manifest.json \
  --episode 0
```

To render a full segment from an asset manifest:

```bash
python demo_v1.py \
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

from grounded.processing import ProcessingClient

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

### v0
You should be given an `index.json` and `captions.jsonl` for your proprietary
dataset. Dataset access is granted separately through the standard AWS
credential chain (for example, an instance role or named local profile);
credentials are never packaged with the index, API, or SDK. Below is a basic
snippet of the modules present in the `grounded` SDK:

```python
import os

from grounded.data.ego_dataset import EgoDataset, EgoEpisode
from grounded.data.visualize import visualize_episode_to_mp4
from grounded.data.visualize_3d import visualize_episode_to_rerun

INDEX_JSON = "index.json"  # change this to your path
CAPTIONS_JSONL = "captions.jsonl"  # change this to your path
EPISODE_IDX = 0

# load dataset & episode
dataset = EgoDataset(
    index_path=INDEX_JSON,
    captions_path=CAPTIONS_JSONL,
    active_cameras=["left-front", "right-front"],
    target_dir="~/.cache/grounded/data",
    min_duration_sec=4,
)
episode = dataset[EPISODE_IDX]

os.makedirs("outputs/", exist_ok=True)

# print caption
print(dataset.get_caption(EPISODE_IDX))
# print(dataset[EPISODE_IDX].caption)  # alternate way to get caption, but will download all episode files

# generate mp4 render
visualize_episode_to_mp4(
    episode=episode,
    output_path=f"outputs/sdkvis{EPISODE_IDX}.mp4",
    downsample=4,
    fps=30,
    max_workers=16,
    max_depth=5,
)

# generate rerun 3d
visualize_episode_to_rerun(
    episode=episode,
    output_path=f"outputs/sdkvis{EPISODE_IDX}.rrd",
)
```

Refer to `docs/DATA.md` for the exact specifications of all parameters used in this library.
