"""Load a dataset manifest and render its captioned hand tracking episodes.

The manifest is produced by the manifest-creation post-process (candidate hand
intervals chopped into discrete captioned episodes by a VLM); see
HandManifest for the on-disk format.

Usage:
    # render one episode (by index or by key) to mp4 + rerun
    python demo_v1.py --manifest manifest.json --episode 3
    python demo_v1.py --manifest manifest.json --captions captions.jsonl --episode {key}
"""

import argparse
import os

from grounded.data.hand_dataset import HAND_CAMS, HandManifest
from grounded.data.visualize_hand import visualize_hand_episode_to_mp4
from grounded.data.visualize_hand_3d import visualize_hand_episode_to_rerun


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="path to a manifest JSON built over downloaded sessions")
    parser.add_argument("--episode", required=True,
                        help="episode index or key to render")
    parser.add_argument("--captions", default=None,
                        help="path to the captions JSONL (defaults to {manifest stem}.captions.jsonl)")
    parser.add_argument("--sessions-root", default=None,
                        help="dir containing the session folders (defaults to the manifest's own directory)")
    parser.add_argument("--cameras", nargs="+", default=HAND_CAMS, choices=HAND_CAMS)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    manifest = HandManifest(args.manifest, sessions_root=args.sessions_root, captions_path=args.captions)

    episode_ref = int(args.episode) if args.episode.lstrip("-").isdigit() else args.episode
    os.makedirs(args.out_dir, exist_ok=True)

    with manifest.open(episode_ref, active_cameras=args.cameras) as episode:
        entry = episode.manifest_entry
        print(
            f"Loaded {entry['key']}: session {entry['session']}, segment {entry['segment']}, "
            f"frames [{entry['frame_start']}, {entry['frame_end']}) = "
            f"{len(episode)} frames @ {episode.fps:.0f} fps"
        )
        print(f"Caption: {episode.caption}")

        visualize_hand_episode_to_mp4(
            episode,
            os.path.join(args.out_dir, f"{entry['key']}.mp4"),
            downsample=args.downsample,
            num_workers=args.num_workers,
            caption=episode.caption,
        )
        visualize_hand_episode_to_rerun(
            episode,
            os.path.join(args.out_dir, f"{entry['key']}.rrd"),
        )


if __name__ == "__main__":
    main()
