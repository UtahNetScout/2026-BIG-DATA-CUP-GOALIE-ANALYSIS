from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.schema_alignment import align_tracking_schema, identify_goalie_rows, summarize_goalie_candidates


def _resolve_tracking_path(manual_path: str | None) -> Path:
    if manual_path:
        path = Path(manual_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Tracking file not found: {path}")
        return path

    tracking_paths = sorted(WORKSPACE_ROOT.glob("*Tracking_*.csv"))
    if not tracking_paths:
        raise FileNotFoundError(
            "No tracking CSV was found in the workspace root. Provide one explicitly with --tracking-file PATH."
        )
    return tracking_paths[0]


def _split_entities(tracking_frame):
    entity_series = tracking_frame["entity_type"].astype(str).str.casefold()
    player_rows = tracking_frame[entity_series == "player"].copy()
    puck_rows = tracking_frame[entity_series == "puck"].copy()
    return player_rows, puck_rows
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the tracking pipeline on one real tracking CSV. "
            "If --tracking-file is omitted, the script uses the first *Tracking_*.csv file in the workspace root."
        )
    )
    parser.add_argument(
        "--tracking-file",
        help="Optional path to a tracking CSV. Provide this explicitly if you want to validate a specific sample file.",
    )
    args = parser.parse_args()

    tracking_path = _resolve_tracking_path(args.tracking_file)
    raw_tracking = align_tracking_schema(pd.read_csv(tracking_path))

    player_rows, puck_rows = _split_entities(raw_tracking)
    goalie_candidate_mask = identify_goalie_rows(raw_tracking)
    goalie_candidate_summary = summarize_goalie_candidates(raw_tracking, goalie_candidate_mask)
    canonical_columns = [
        column
        for column in [
            "image_id",
            "game",
            "period",
            "game_clock",
            "entity_type",
            "team",
            "player_id",
            "x_feet",
            "y_feet",
            "z_feet",
        ]
        if column in raw_tracking.columns
    ]

    print(f"tracking_file: {tracking_path}")
    print(f"total_rows: {len(raw_tracking)}")
    print(f"player_rows: {len(player_rows)}")
    print(f"puck_rows: {len(puck_rows)}")
    print(f"goalie_candidate_frame_rows: {goalie_candidate_summary['frame_level_goalie_candidate_rows']}")
    print(f"unique_goalie_candidate_player_ids: {goalie_candidate_summary['unique_goalie_candidate_player_ids']}")
    print(f"goalie_candidate_team_distribution: {goalie_candidate_summary['goalie_candidate_team_distribution']}")
    print(f"goalie_heuristic_conditions: {goalie_candidate_summary['heuristic_conditions']}")
    print(f"unique_games: {sorted(raw_tracking['game'].dropna().unique().tolist())}")
    print(f"canonical_columns: {canonical_columns[:8]}")
    print(
        "note: goalie_candidate_frame_rows reflects placeholder frame-level heuristic output, "
        "not a finalized goalie classifier or a count of unique goalies."
    )


if __name__ == "__main__":
    main()