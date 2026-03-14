from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
	sys.path.insert(0, str(WORKSPACE_ROOT))

from src.goalie_trajectories import GOALIE_TRAJECTORY_NOTE, extract_goalie_trajectories
from src.metrics import compute_motion_efficiency
from src.schema_alignment import align_tracking_schema


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Summarize frame-level goalie candidate motion on one tracking CSV."
	)
	parser.add_argument("tracking_file", help="Path to a tracking CSV file.")
	parser.add_argument("--preview-rows", type=int, default=10, help="Number of motion summary rows to preview.")
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

	team_row_summary = goalie_trajectories.groupby(["game", "team"], dropna=False, sort=False).agg(
		frame_level_goalie_rows=("image_id", "size"),
		unique_candidate_player_ids=("player_id", "nunique"),
	).reset_index()
	team_motion_summary = motion_summary.groupby(["game", "team"], dropna=False, sort=False).agg(
		motion_summary_groups=("player_id", "size"),
		total_path_length_feet=("path_length_feet", "sum"),
		median_motion_efficiency=("motion_efficiency", "median"),
		median_frame_count=("frame_count", "median"),
	).reset_index()
	team_summary = team_row_summary.merge(team_motion_summary, on=["game", "team"], how="left")

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
	print(f"total_frame_level_goalie_rows: {len(goalie_trajectories)}")
	print(f"motion_summary_groups: {len(motion_summary)}")
	print("team_level_summary:")
	if team_summary.empty:
		print("<no team summary rows>")
	else:
		print(team_summary.to_string(index=False))
	print("motion_summary_preview:")
	if motion_summary.empty:
		print("<no motion summary rows>")
	else:
		print(motion_summary.loc[:, preview_columns].head(args.preview_rows).to_string(index=False))
	print(f"caveat: {GOALIE_TRAJECTORY_NOTE}")
	print(
		"analysis_note: motion summary groups are frame-level goalie candidate player_id groupings within a game/team; "
		"they are useful for auditing movement patterns but do not imply stabilized goalie identity."
	)


if __name__ == "__main__":
	main()