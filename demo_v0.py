import os

from grounded.data.ego_dataset import EgoDataset, EgoEpisode
from grounded.data.visualize import visualize_episode_to_mp4
from grounded.data.visualize_3d import visualize_episode_to_rerun

INDEX_JSON = "index.json"
CAPTIONS_JSONL = "captions.jsonl"
EPISODE_IDX = 0


if __name__ == "__main__":
    dataset = EgoDataset(
        index_path=INDEX_JSON,
        captions_path=None,
        active_cameras=["left-front", "right-front"],
        aws_profile="mecha",
        target_dir="~/.cache/grounded/data",
        min_duration_sec=5,
    )
    os.makedirs("outputs/", exist_ok=True)
    episode: EgoEpisode = dataset[EPISODE_IDX]
    print(dataset.get_caption(EPISODE_IDX))

    visualize_episode_to_mp4(
        episode=episode,
        output_path=f"outputs/sdkvis{EPISODE_IDX}.mp4",
        downsample=4,
        fps=30,
        max_workers=16,
        max_depth=5,
    )
    visualize_episode_to_rerun(episode, f"outputs/sdkvis{EPISODE_IDX}.rrd", pcd_downsample=8)
