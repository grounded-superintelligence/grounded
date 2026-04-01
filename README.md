```
 ______   _____    ______   _    _   _    _   _____    ______   _____          
/ _____/ |  __ \  /  __  \ | |  | | | \  | | | ___ \  |  ____| | ___ \         
| |__  | | |__) | | (__) | | |__| | |  \ | | | |__) | |  ___|  | |__) |        
\______/ |_|  \_\ \______/  \____/  |_|  \_| |_____/  |______| |_____/        
                                                                            
 ______  _    _   ______   ______   _____    ______   _    _   ______   ______ 
/ ____/ | |  | | |  ___ \ |  ____| |  __ \  |__  __| | \  | | |__  __| |  ____|
\____ \ | |__| | | |__/ / |  ___|  | |__) |   |  |   |  \ | |   |  |   |  ___| 
/_____/  \____/  | |      |______| |_|  \_\ |______| |_|  \_|   |__|   |______|
                                                                        
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

Install the AWS CLI binary
```bash
sudo apt update && sudo apt install unzip curl -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws/
```

## usage
You should be given an `index.json` and `credentials` that corresponds your proprietary dataset. Add the contents of `credentials` to your `~/.aws/credentials` file. Below is a basic snippet of the basic modules present in `grounded` SDK:

```python
from grounded.data.ego_dataset import EgoDataset, EgoEpisode
from grounded.data.visualize import visualize_episode_to_mp4

# initialize dataset
# you target_dir defaults to your cache, so make sure to change this for each dataset
dataset = EgoDataset("index.json", aws_profile="grounded", target_dir="~/.cache/grounded/data")
print(f"Found {len(dataset)} episodes")

# initialize episode
dataset.download_episode(0)
episode: EgoEpisode = dataset[0]

# visualize episode
visualize_episode_to_mp4(episode, downsample=4, fps=30, overlay_text=False, output="test.mp4")
```
