"""Download and visualize one full Hand asset or one captioned Hand episode.

Examples:
    grounded-demo --manifest manifest.json --episode 0
    grounded-demo --manifest manifest.json --episode ep_v1_...
    grounded-demo --manifest assets.json --asset-id ast_v1_...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grounded.data.hand_dataset import HAND_CAMS
from grounded.data.visualize_hand import visualize_hand_episode_to_mp4
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="asset or final episode manifest")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--episode", help="episode index or episode_id")
    identity.add_argument("--asset-id", help="asset_id for a full enriched segment")
    parser.add_argument("--aws-profile", help="optional named AWS profile for s3:// artifact URIs")
    parser.add_argument("--target-dir", default="~/.cache/grounded/data", help="download cache")
    parser.add_argument("--cameras", nargs="+", default=["left_front"], choices=HAND_CAMS)
    parser.add_argument("--output", help="output MP4 path; defaults to outputs/{id}.mp4")
    parser.add_argument("--download-all", action="store_true", help="download every available lane before rendering Hand")
    parser.add_argument(
        "--allow-missing-sha256",
        action="store_true",
        help="allow files without a producer-published SHA-256",
    )
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    client = ProcessingClient.from_manifest(args.manifest, aws_profile=args.aws_profile)
    require_sha256 = not args.allow_missing_sha256

    if args.episode is not None:
        record = _resolve_episode(client, args.episode)
        identifier = record.episode_id
        caption = record.caption or None
        if args.download_all:
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
        if args.download_all:
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
    output = Path(args.output or f"outputs/{identifier}.mp4").expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Loaded {identifier}: {len(hand)} Hand frame(s)")
        if caption:
            print(f"Caption: {caption}")
        visualize_hand_episode_to_mp4(
            hand,
            str(output),
            downsample=args.downsample,
            num_workers=args.num_workers,
            caption=caption,
        )
    finally:
        hand.close()


if __name__ == "__main__":
    main()
