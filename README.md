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

First download `conda` or your preferred Python environment manager.

```bash
conda create -n grounded python=3.10
conda activate grounded
python -m pip install ./grounded-1.0.0-py3-none-any.whl
```

The wheel can also be published to PyPI later; that only changes the install
command.

## usage

### v1

An `asset_id` identifies one enriched segment. An `episode_id` identifies one
captioned clip inside that segment.

A delivery includes `manifest.json` plus access to the files it references.
No API server is required:

```python
from grounded.data.visualize_hand import visualize_hand_episode_to_mp4
from grounded.processing import ProcessingClient

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
asset and episode control flow.

The included Python demo downloads every available lane and creates both MP4
and Rerun visualizations:

```bash
python demo_v1.py --manifest episodes.json --episode 0
python demo_v1.py --manifest assets.json --asset-id ast_v1_...
```

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
