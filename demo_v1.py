"""Load an asset or episode manifest and render Hand tracking.

Usage:
    python demo_v1.py --manifest episodes.json --episode 0
    python demo_v1.py --manifest episodes.json --episode ep_v1_...
    python demo_v1.py --manifest assets.json --asset-id ast_v1_...
"""

import argparse
from pathlib import Path

from grounded.data.hand_dataset import HAND_CAMS
from grounded.data.visualize_hand import visualize_hand_episode_to_mp4
from grounded.data.visualize_hand_3d import visualize_hand_episode_to_rerun
from grounded.processing import ProcessingClient


def _resolve_episode(client: ProcessingClient, reference: str):
    try:
        index = int(reference)
    except ValueError:
        return client.get_episode(reference)

    episodes = client.list_episodes()
    try:
        return episodes[index]
    except IndexError as exc:
        raise SystemExit(f"episode index {index} is outside a manifest with {len(episodes)} episode(s)") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="asset or episode manifest")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--episode", help="episode index or episode_id")
    identity.add_argument("--asset-id", help="asset_id for a full enriched segment")
    parser.add_argument("--aws-profile", help="optional named AWS profile for s3:// file URIs")
    parser.add_argument("--cameras", nargs="+", default=HAND_CAMS, choices=HAND_CAMS)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--target-dir", default="~/.cache/grounded/data")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--allow-missing-sha256", action="store_true")
    args = parser.parse_args()

    client = ProcessingClient.from_manifest(args.manifest, aws_profile=args.aws_profile)
    require_sha256 = not args.allow_missing_sha256

    if args.episode is not None:
        record = _resolve_episode(client, args.episode)
        identifier = record.episode_id
        caption = record.caption or None
        download = client.download_episode(
            identifier,
            target_dir=args.target_dir,
            require_sha256=require_sha256,
        )
        for lane in download.lanes:
            print(f"{lane.lane}: {lane.status}, {len(lane.files)} downloaded file(s)")
    else:
        identifier = args.asset_id
        record = client.get_asset(identifier)
        caption = None
        for lane in client.list_asset_artifacts(identifier):
            print(f"{lane.lane}: {lane.status}")
        download = client.download_asset(
            identifier,
            target_dir=args.target_dir,
            require_sha256=require_sha256,
        )
        print(f"Downloaded {len(download.files)} full-segment enrichment file(s)")

    hand = client.open_hand(
        identifier,
        target_dir=args.target_dir,
        active_cameras=args.cameras,
        require_sha256=require_sha256,
    )
    output_dir = Path(args.out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Loaded {identifier}: {len(hand)} Hand frame(s)")
        if caption:
            print(f"Caption: {caption}")
        visualize_hand_episode_to_mp4(
            hand,
            str(output_dir / f"{identifier}.mp4"),
            downsample=args.downsample,
            num_workers=args.num_workers,
            caption=caption,
        )
        visualize_hand_episode_to_rerun(
            hand,
            str(output_dir / f"{identifier}.rrd"),
        )
    finally:
        hand.close()


if __name__ == "__main__":
    main()
