from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.goalie_trajectories import GOALIE_IDENTITY_STABILITY_NOTE, GOALIE_TRAJECTORY_NOTE, extract_goalie_trajectories, infer_stable_goalie_identities
from src.metrics import compute_motion_efficiency
from src.schema_alignment import align_tracking_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frame-level goalie trajectory extraction and print motion summary outputs on one tracking CSV."
    )
    parser.add_argument("tracking_file", help="Path to a tracking CSV file.")
    return parser.parse_args()


def _fmt_pct(value: object) -> str:
    """Format a share value as a percentage string, or 'N/A' when unavailable."""
    try:
        return f"{float(value) * 100:.1f}%"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "N/A"


def _print_stability_diagnostics(goalie_trajectories: pd.DataFrame) -> None:
    """Print a per-game/team stability diagnostic block."""

    # Derive per-group stability summary from the annotated trajectory frame.
    stability_cols = [
        "game", "team",
        "frame_level_candidate_rows", "frame_level_unique_candidate_player_ids",
        "stability_player_share", "stability_margin_frames", "stability_rule_passed_by_player_id",
        "stability_by_jersey_number_rule_passed", "frame_level_unique_candidate_jersey_numbers",
        "stability_jersey_share", "stability_jersey_margin_frames",
        "stability_rule_passed", "goalie_identity_inference_level",
        "stabilized_goalie_player_id", "stabilized_goalie_jersey_number",
        "stabilized_goalie_frame_count",
    ]
    available = [c for c in stability_cols if c in goalie_trajectories.columns]
    stability_rows = (
        goalie_trajectories[available]
        .drop_duplicates(subset=["game", "team"])
        .sort_values(["game", "team"])
        .reset_index(drop=True)
    )

    print("--- Identity Stabilization Diagnostics (per game/team) ---")
    for _, row in stability_rows.iterrows():
        game = row.get("game", "?")
        team = row.get("team", "?")
        print(f"  game={game}  team={team}")
        print(f"    candidate_rows={row.get('frame_level_candidate_rows', 'N/A')}")

        # Player-id path
        player_share = _fmt_pct(row.get("stability_player_share"))
        player_margin = row.get("stability_margin_frames", "N/A")
        player_unique = row.get("frame_level_unique_candidate_player_ids", "N/A")
        player_passed = bool(row.get("stability_rule_passed_by_player_id", False))
        player_status = "PASS" if player_passed else "FAIL"
        print(f"    player_id:   unique_ids={player_unique}  share={player_share}  margin={player_margin}  rule={player_status}")

        # Jersey-number path (only print when column present)
        if "stability_by_jersey_number_rule_passed" in row.index:
            jersey_share = _fmt_pct(row.get("stability_jersey_share"))
            jersey_margin = row.get("stability_jersey_margin_frames", "N/A")
            jersey_unique = row.get("frame_level_unique_candidate_jersey_numbers", "N/A")
            jersey_passed = bool(row.get("stability_by_jersey_number_rule_passed", False))
            jersey_status = "PASS (fallback)" if jersey_passed else "FAIL"
            print(f"    jersey_num:  unique_jerseys={jersey_unique}  share={jersey_share}  margin={jersey_margin}  rule={jersey_status}")

        # Outcome
        level = row.get("goalie_identity_inference_level", "unresolved")
        frame_count = row.get("stabilized_goalie_frame_count", 0)
        pid = row.get("stabilized_goalie_player_id")
        jersey = row.get("stabilized_goalie_jersey_number")
        if level == "stabilized":
            print(f"    => outcome=stabilized  by=player_id  stabilized_player_id={pid}  stabilized_frames={frame_count}")
        elif level == "stabilized_by_jersey_number":
            print(f"    => outcome=stabilized_by_jersey_number  stabilized_jersey={jersey}  stabilized_frames={frame_count}")
        else:
            print(f"    => outcome=unresolved")


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

    # For stabilized motion summary, group by the resolved goalie identity:
    # player_id-stabilized groups have a meaningful stabilized_goalie_player_id;
    # jersey-stabilized groups do not, so fall back to stabilized_goalie_jersey_number.
    if "stabilized_goalie_jersey_number" in stabilized_goalie_rows.columns:
        stabilized_goalie_rows["resolved_goalie_identity"] = stabilized_goalie_rows["stabilized_goalie_player_id"].where(
            stabilized_goalie_rows["stabilized_goalie_player_id"].notna(),
            stabilized_goalie_rows["stabilized_goalie_jersey_number"],
        )
        stabilized_motion_summary = compute_motion_efficiency(
            stabilized_goalie_rows,
            group_columns=("game", "team", "resolved_goalie_identity"),
            x_column="trajectory_x_feet",
            y_column="trajectory_y_feet",
        )
    else:
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
    _print_stability_diagnostics(goalie_trajectories)
    print("motion_summary_preview:")
    if motion_summary.empty:
        print("<no motion summary rows>")
    else:
        print(motion_summary.loc[:, preview_columns].head(10).to_string(index=False))
    print("stabilized_motion_summary:")
    if stabilized_motion_summary.empty:
        print("<no stabilized motion summary rows>")
    else:
        stab_preview = [c for c in preview_columns if c in stabilized_motion_summary.columns]
        # Replace 'player_id' with the actual identity column used
        if "player_id" not in stabilized_motion_summary.columns and "resolved_goalie_identity" in stabilized_motion_summary.columns:
            stab_preview = ["resolved_goalie_identity" if c == "player_id" else c for c in stab_preview]
        print(stabilized_motion_summary.loc[:, stab_preview].head(10).to_string(index=False))
    print(f"caveat: {GOALIE_TRAJECTORY_NOTE}")
    print(f"identity_caveat: {GOALIE_IDENTITY_STABILITY_NOTE}")


if __name__ == "__main__":
    main()