from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
	sys.path.insert(0, str(WORKSPACE_ROOT))

from src.goalie_trajectories import infer_stable_goalie_identities, select_frame_level_goalie_rows
from src.schema_alignment import (
	GOALIE_MAX_ABS_Y_FEET,
	GOALIE_MIN_ABS_X_FEET,
	GOALIE_MIN_PLAYER_ROWS_PER_TEAM_FRAME,
	align_tracking_schema,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Inspect frame-level goalie candidate churn and conservative stability evidence on one tracking CSV."
	)
	parser.add_argument("tracking_file", help="Path to a tracking CSV file.")
	parser.add_argument("--top-n", type=int, default=10, help="How many top candidate player_ids to print per game/team.")
	return parser.parse_args()


def _prepare_eligible_player_rows(tracking_frame: pd.DataFrame) -> pd.DataFrame:
	player_rows = tracking_frame[tracking_frame["entity_type"].astype(str).str.casefold() == "player"].copy()
	player_rows["player_count_in_team_frame"] = player_rows.groupby(["image_id", "team"])["image_id"].transform("size")
	player_rows["abs_x_feet"] = player_rows["x_feet"].abs()
	player_rows["abs_y_feet"] = player_rows["y_feet"].abs()
	return player_rows[
		(player_rows["player_count_in_team_frame"] >= GOALIE_MIN_PLAYER_ROWS_PER_TEAM_FRAME)
		& (player_rows["abs_x_feet"] >= GOALIE_MIN_ABS_X_FEET)
		& (player_rows["abs_y_feet"] <= GOALIE_MAX_ABS_Y_FEET)
	].copy()


def _candidate_count_distribution(goalie_rows: pd.DataFrame) -> pd.DataFrame:
	counts = (
		goalie_rows.groupby(["game", "team", "player_id"], dropna=False)
		.size()
		.rename("candidate_frame_count")
		.reset_index()
	)
	return counts.sort_values(
		by=["game", "team", "candidate_frame_count", "player_id"],
		ascending=[True, True, False, True],
		kind="stable",
	)


def _tie_diagnostics(eligible_rows: pd.DataFrame) -> pd.DataFrame:
	max_x = eligible_rows.groupby(["image_id", "team"], sort=False).agg(max_abs_x=("abs_x_feet", "max")).reset_index()
	top_x = eligible_rows.merge(max_x, on=["image_id", "team"], how="inner")
	top_x = top_x[top_x["abs_x_feet"].eq(top_x["max_abs_x"])].copy()
	top_x["min_abs_y_within_top_x"] = top_x.groupby(["image_id", "team"], sort=False)["abs_y_feet"].transform("min")
	top_xy = top_x[top_x["abs_y_feet"].eq(top_x["min_abs_y_within_top_x"])].copy()

	return pd.DataFrame(
		[
			{
				"frames_with_abs_x_ties": int(top_x.groupby(["image_id", "team"]).size().gt(1).sum()),
				"frames_with_abs_x_and_abs_y_ties": int(top_xy.groupby(["image_id", "team"]).size().gt(1).sum()),
			}
		]
	)


def _geometry_diagnostics(goalie_rows: pd.DataFrame) -> pd.DataFrame:
	working = goalie_rows.copy()
	working["abs_x_feet"] = working["x_feet"].abs()
	working["abs_y_feet"] = working["y_feet"].abs()
	return working.groupby(["game", "team"], dropna=False, sort=False).agg(
		frame_level_candidate_rows=("image_id", "size"),
		selected_abs_x_lt_85=("abs_x_feet", lambda s: int((s < 85.0).sum())),
		selected_abs_x_lt_82=("abs_x_feet", lambda s: int((s < 82.0).sum())),
		selected_abs_y_gt_12=("abs_y_feet", lambda s: int((s > 12.0).sum())),
		selected_abs_y_gt_14=("abs_y_feet", lambda s: int((s > 14.0).sum())),
	).reset_index()


def _identifier_diagnostics(goalie_rows: pd.DataFrame) -> pd.DataFrame:
	return goalie_rows.groupby(["game", "team"], dropna=False, sort=False).agg(
		frame_level_candidate_rows=("image_id", "size"),
		unique_player_ids=("player_id", "nunique"),
		unique_jersey_numbers=("player_jersey_number", "nunique"),
		missing_player_ids=("player_id", lambda s: int(s.isna().sum())),
		missing_jersey_numbers=("player_jersey_number", lambda s: int(s.isna().sum())),
	).reset_index()


def main() -> None:
	args = parse_args()
	tracking_path = Path(args.tracking_file).expanduser().resolve()
	if not tracking_path.exists():
		raise FileNotFoundError(f"Tracking file not found: {tracking_path}")

	tracking_frame = align_tracking_schema(pd.read_csv(tracking_path))
	goalie_rows = select_frame_level_goalie_rows(tracking_frame)
	eligible_rows = _prepare_eligible_player_rows(tracking_frame)
	stability = infer_stable_goalie_identities(goalie_rows)
	candidate_counts = _candidate_count_distribution(goalie_rows)
	identifier_diagnostics = _identifier_diagnostics(goalie_rows)
	geometry_diagnostics = _geometry_diagnostics(goalie_rows)
	tie_diagnostics = _tie_diagnostics(eligible_rows)
	jersey_mapping = (
		goalie_rows.dropna(subset=["player_id", "player_jersey_number"])
		.groupby(["game", "team", "player_jersey_number"], dropna=False)["player_id"]
		.nunique()
		.reset_index(name="unique_player_ids_per_jersey")
	)

	print(f"tracking_file: {tracking_path}")
	print(f"frame_level_goalie_candidate_rows: {len(goalie_rows)}")
	print("stability_summary:")
	print(stability.to_string(index=False))
	print("identifier_diagnostics:")
	print(identifier_diagnostics.to_string(index=False))
	print("geometry_diagnostics:")
	print(geometry_diagnostics.to_string(index=False))
	print("tie_diagnostics:")
	print(tie_diagnostics.to_string(index=False))

	for (game, team), group in candidate_counts.groupby(["game", "team"], sort=False):
		print(f"top_candidate_player_ids: game={game} team={team}")
		print(group.head(args.top_n).to_string(index=False))

		jersey_group = jersey_mapping[(jersey_mapping["game"] == game) & (jersey_mapping["team"] == team)]
		if not jersey_group.empty:
			print(f"top_jersey_to_player_id_churn: game={game} team={team}")
			print(
				jersey_group.sort_values(
					by=["unique_player_ids_per_jersey", "player_jersey_number"],
					ascending=[False, True],
					kind="stable",
				)
				.head(args.top_n)
				.to_string(index=False)
			)

	print(
		"note: frame-level goalie candidate selection remains separate from cross-frame identity inference; "
		"an unresolved stabilized result is preferred over an unjustified assignment."
	)


if __name__ == "__main__":
	main()