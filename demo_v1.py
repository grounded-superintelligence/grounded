"""Download, load, and render hand tracking v2 for one segment recording.

Usage:
    python examples/render_hand_tracking.py s3://bucket/path/to/{session} --segment 0
    python examples/render_hand_tracking.py /local/path/to/{session} --segment 0
"""

import argparse
import os

from grounded.data.hand_dataset import HAND_CAMS, HandEpisode, download_hand_segment
from grounded.data.visualize_hand import visualize_hand_episode_to_mp4
from grounded.data.visualize_hand_3d import visualize_hand_episode_to_rerun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="s3:// URI or local path of the session folder")
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--cameras", nargs="+", default=HAND_CAMS, choices=HAND_CAMS)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--aws-profile", default=None)
    args = parser.parse_args()

    session_dir = args.session
    if session_dir.startswith("s3://"):
        session_dir = download_hand_segment(session_dir, args.segment, aws_profile=args.aws_profile, verbose=True)

    os.makedirs(args.out_dir, exist_ok=True)

    with HandEpisode(session_dir, segment=args.segment, active_cameras=args.cameras) as episode:
        print(
            f"Loaded segment {args.segment}: {len(episode)} frames @ {episode.fps:.0f} fps | "
            f"yield: left {episode.yield_stats['left_pct']}%, right {episode.yield_stats['right_pct']}% | "
            f"{len(episode.continuous_intervals)} continuous both-hands intervals"
        )

        visualize_hand_episode_to_mp4(
            episode,
            os.path.join(args.out_dir, f"{os.path.basename(session_dir.rstrip('/'))}-segment{args.segment}.mp4"),
            downsample=args.downsample,
        )
        visualize_hand_episode_to_rerun(
            episode,
            os.path.join(args.out_dir, f"handvis_segment{args.segment}.rrd"),
        )


if __name__ == "__main__":
    main()
