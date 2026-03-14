from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.goalie_trajectories import GOALIE_IDENTITY_STABILITY_NOTE, GOALIE_TRAJECTORY_NOTE, extract_goalie_trajectories
from src.metrics import compute_motion_efficiency
from src.schema_alignment import align_tracking_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frame-level goalie trajectory extraction and print motion summary outputs on one tracking CSV."
    )
    parser.add_argument("tracking_file", help="Path to a tracking CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_path = Path(args.tracking_file).expanduser().resolve()
    if not tracking_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_path}")

    raw_tracking = pd.read_csv(tracking_path)
    canonical_tracking = align_tracking_schema(raw_tracking)
    goalie_trajectories = extract_goalie_trajectories(canonical_tracking)
    motion_summary = compute_motion_efficiency(
        goalie_trajectories,
        group_columns=("game", "team", "player_id"),
        x_column="trajectory_x_feet",
        y_column="trajectory_y_feet",
    )
    stabilized_goalie_rows = goalie_trajectories[goalie_trajectories["is_stabilized_goalie_frame"].fillna(False)].copy()
    stabilized_motion_summary = compute_motion_efficiency(
        stabilized_goalie_rows,
        group_columns=("game", "team", "stabilized_goalie_player_id"),
        x_column="trajectory_x_feet",
        y_column="trajectory_y_feet",
    )

    preview_columns = [
        column
        for column in [
            "game",
            "team",
            "player_id",
            "frame_count",
            "path_length_feet",
            "straight_line_distance_feet",
            "motion_efficiency",
            "start_x_feet",
            "start_y_feet",
            "end_x_feet",
            "end_y_feet",
        ]
        if column in motion_summary.columns
    ]

    print(f"tracking_file: {tracking_path}")
    print(f"goalie_trajectory_rows: {len(goalie_trajectories)}")
    print(f"frame_level_unique_goalie_candidate_player_ids: {goalie_trajectories['player_id'].dropna().nunique()}")
    print(f"stabilized_goalie_identity_groups: {goalie_trajectories[['game', 'team', 'stability_rule_passed']].drop_duplicates()['stability_rule_passed'].fillna(False).sum()}")
    print(f"stabilized_goalie_frame_rows: {int(goalie_trajectories['is_stabilized_goalie_frame'].fillna(False).sum())}")
    print(f"frame_level_motion_summary_groups: {len(motion_summary)}")
    print(f"stabilized_motion_summary_groups: {len(stabilized_motion_summary)}")
    print("motion_summary_preview:")
    if motion_summary.empty:
        print("<no motion summary rows>")
    else:
        print(motion_summary.loc[:, preview_columns].head(10).to_string(index=False))
    print(f"caveat: {GOALIE_TRAJECTORY_NOTE}")
    print(f"identity_caveat: {GOALIE_IDENTITY_STABILITY_NOTE}")


if __name__ == "__main__":
    main()