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
python -m pip install git+https://github.com/grounded-superintelligence/grounded.git
```

## usage
You should be given an `index.json` and `credentials` that corresponds your proprietary dataset. Add the contents of `credentials` to your `~/.aws/credentials` file. Below is a basic snippet of the basic modules present in `grounded` SDK:

```python
from grounded.data.ego_dataset import EgoDataset, EgoEpisode
from grounded.data.visualize import visualize_episode_to_mp4

INDEX_JSON = "index.json"  # change this to your path
EPISODE_IDX = 0

dataset = EgoDataset(
    index_path=INDEX_JSON,
    active_cameras=["left-front", "right-front", "left-eye", "right-eye"],
    target_dir="~/.cache/grounded/data",
    min_duration_sec=4,
)
os.makedirs("outputs/", exist_ok=True)
episode = dataset[EPISODE_IDX]
visualize_episode_to_mp4(
    episode=episode,
    output_path=f"outputs/sdkvis{EPISODE_IDX}.mp4",
    downsample=4,
    fps=30,
    max_workers=16,
    max_depth=5,
)
```
