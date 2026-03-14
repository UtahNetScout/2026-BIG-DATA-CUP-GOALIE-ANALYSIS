from __future__ import annotations

from typing import Final, Sequence

import pandas as pd

from .metrics import build_tracking_frame_sequence
from .schema_alignment import goalie_heuristic_conditions, identify_goalie_rows


GOALIE_TRAJECTORY_NOTE: Final[str] = (
	"Frame-level goalie trajectory extraction uses a heuristic row selector and is not a finalized goalie identity model."
)
GOALIE_IDENTITY_STABILITY_NOTE: Final[str] = (
	"Cross-frame goalie identity inference is a conservative modal-player summary per game/team, not a finalized identity tracker."
)
STABLE_GOALIE_ID_MIN_CANDIDATE_FRAMES: Final[int] = 3
STABLE_GOALIE_ID_MIN_SHARE: Final[float] = 0.6
STABLE_GOALIE_ID_MIN_MARGIN_FRAMES: Final[int] = 2


def validate_goalie_trajectory_input(
	tracking_frame: pd.DataFrame,
	*,
	require_game_column: bool = True,
) -> None:
	required_columns = {"image_id", "team", "period", "game_clock", "entity_type", "player_id", "x_feet", "y_feet"}
	if require_game_column:
		required_columns.add("game")

	missing_columns = sorted(required_columns.difference(tracking_frame.columns))
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"tracking_frame is missing columns required for goalie trajectory extraction: {missing}")


def select_frame_level_goalie_rows(tracking_frame: pd.DataFrame) -> pd.DataFrame:
	"""Select one likely goalie row per image_id/team using the current heuristic."""

	validate_goalie_trajectory_input(tracking_frame)
	goalie_mask = identify_goalie_rows(tracking_frame)
	goalie_rows = tracking_frame.loc[goalie_mask].copy()
	goalie_rows["goalie_selection_level"] = "frame"
	goalie_rows["goalie_selector"] = "heuristic"
	goalie_rows["goalie_selector_note"] = GOALIE_TRAJECTORY_NOTE
	return goalie_rows


def infer_stable_goalie_identities(goalie_rows: pd.DataFrame) -> pd.DataFrame:
	"""Infer a conservative per-game/team goalie identity from frame-level candidates.

	The rule only resolves a stabilized identity when one player_id clearly dominates
	the frame-level candidate rows for a game/team. Ambiguous groups remain unresolved.
	"""

	required_columns = {"game", "team", "player_id"}
	missing_columns = sorted(required_columns.difference(goalie_rows.columns))
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"goalie_rows is missing columns required for identity stability inference: {missing}")

	base_groups = goalie_rows.loc[:, ["game", "team"]].drop_duplicates().reset_index(drop=True)
	if base_groups.empty:
		return pd.DataFrame(
			columns=[
				"game",
				"team",
				"frame_level_candidate_rows",
				"frame_level_unique_candidate_player_ids",
				"stabilized_goalie_player_id",
				"stabilized_goalie_frame_count",
				"stability_player_share",
				"stability_margin_frames",
				"stability_rule_passed",
				"goalie_identity_inference_level",
				"goalie_identity_inference_note",
			]
		)

	count_frame = goalie_rows.dropna(subset=["player_id"]).copy()
	if count_frame.empty:
		result = base_groups.copy()
		result["frame_level_candidate_rows"] = 0
		result["frame_level_unique_candidate_player_ids"] = 0
		result["stabilized_goalie_player_id"] = pd.NA
		result["stabilized_goalie_frame_count"] = 0
		result["stability_player_share"] = 0.0
		result["stability_margin_frames"] = 0
		result["stability_rule_passed"] = False
		result["goalie_identity_inference_level"] = "unresolved"
		result["goalie_identity_inference_note"] = GOALIE_IDENTITY_STABILITY_NOTE
		return result

	counts = (
		count_frame.groupby(["game", "team", "player_id"], dropna=False)
		.size()
		.rename("candidate_frame_count")
		.reset_index()
	)
	counts = counts.sort_values(
		by=["game", "team", "candidate_frame_count", "player_id"],
		ascending=[True, True, False, True],
		kind="stable",
	)

	group_totals = counts.groupby(["game", "team"], sort=False).agg(
		frame_level_candidate_rows=("candidate_frame_count", "sum"),
		frame_level_unique_candidate_player_ids=("player_id", "nunique"),
	).reset_index()
	top_counts = counts.groupby(["game", "team"], sort=False).head(2).copy()
	top_counts["candidate_rank"] = top_counts.groupby(["game", "team"], sort=False).cumcount() + 1

	leading = top_counts[top_counts["candidate_rank"] == 1].rename(
		columns={
			"player_id": "leading_player_id",
			"candidate_frame_count": "leading_candidate_frame_count",
		}
	)
	runner_up = top_counts[top_counts["candidate_rank"] == 2].rename(
		columns={
			"candidate_frame_count": "runner_up_candidate_frame_count",
		}
	)

	stability = group_totals.merge(
		leading.loc[:, ["game", "team", "leading_player_id", "leading_candidate_frame_count"]],
		on=["game", "team"],
		how="left",
	)
	stability = stability.merge(
		runner_up.loc[:, ["game", "team", "runner_up_candidate_frame_count"]],
		on=["game", "team"],
		how="left",
	)
	stability["runner_up_candidate_frame_count"] = stability["runner_up_candidate_frame_count"].fillna(0)
	stability["stability_player_share"] = (
		stability["leading_candidate_frame_count"] / stability["frame_level_candidate_rows"]
	)
	stability["stability_margin_frames"] = (
		stability["leading_candidate_frame_count"] - stability["runner_up_candidate_frame_count"]
	)
	stability["stability_rule_passed"] = (
		(stability["leading_candidate_frame_count"] >= STABLE_GOALIE_ID_MIN_CANDIDATE_FRAMES)
		& (stability["stability_player_share"] >= STABLE_GOALIE_ID_MIN_SHARE)
		& (stability["stability_margin_frames"] >= STABLE_GOALIE_ID_MIN_MARGIN_FRAMES)
	)
	stability["stabilized_goalie_player_id"] = stability["leading_player_id"].where(stability["stability_rule_passed"], pd.NA)
	stability["stabilized_goalie_frame_count"] = stability["leading_candidate_frame_count"].where(stability["stability_rule_passed"], 0)
	stability["goalie_identity_inference_level"] = stability["stability_rule_passed"].map({True: "stabilized", False: "unresolved"})
	stability["goalie_identity_inference_note"] = GOALIE_IDENTITY_STABILITY_NOTE
	return stability[
		[
			"game",
			"team",
			"frame_level_candidate_rows",
			"frame_level_unique_candidate_player_ids",
			"stabilized_goalie_player_id",
			"stabilized_goalie_frame_count",
			"stability_player_share",
			"stability_margin_frames",
			"stability_rule_passed",
			"goalie_identity_inference_level",
			"goalie_identity_inference_note",
		]
	]


def attach_stable_goalie_identity(goalie_rows: pd.DataFrame) -> pd.DataFrame:
	"""Annotate frame-level goalie rows with conservative cross-frame identity inference."""

	stability = infer_stable_goalie_identities(goalie_rows)
	annotated = goalie_rows.merge(stability, on=["game", "team"], how="left", validate="m:1")
	annotated["is_stabilized_goalie_frame"] = (
		annotated["stability_rule_passed"].fillna(False)
		& annotated["player_id"].eq(annotated["stabilized_goalie_player_id"])
	)
	return annotated


def sort_goalie_trajectory_frames(
	goalie_rows: pd.DataFrame,
	*,
	group_columns: Sequence[str] = ("game", "team"),
) -> pd.DataFrame:
	"""Sort goalie trajectory rows into stable frame order when frame fields are available."""

	validate_goalie_trajectory_input(goalie_rows)
	working = goalie_rows.copy()
	if "frame_sequence" not in working.columns:
		working["frame_sequence"] = build_tracking_frame_sequence(working)

	sort_columns = [*group_columns, "frame_sequence", "image_id"]
	missing_columns = sorted(set(sort_columns).difference(working.columns))
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"goalie_rows is missing columns required for trajectory sorting: {missing}")

	return working.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)


def extract_goalie_trajectories(
	tracking_frame: pd.DataFrame,
	*,
	prefer_normalized_coordinates: bool = True,
	add_frame_sequence: bool = True,
) -> pd.DataFrame:
	"""Return frame-level goalie trajectories suitable for movement metrics.

	The result keeps one heuristic goalie row per image_id/team and adds explicit
	trajectory coordinate columns so downstream metric code can consume a stable,
	reusable shape without assuming this is a finalized goalie identity model.
	"""

	goalie_rows = select_frame_level_goalie_rows(tracking_frame)
	x_column = "normalized_x_feet" if prefer_normalized_coordinates and "normalized_x_feet" in goalie_rows.columns else "x_feet"
	y_column = "normalized_y_feet" if prefer_normalized_coordinates and "normalized_y_feet" in goalie_rows.columns else "y_feet"
	z_column = "normalized_z_feet" if prefer_normalized_coordinates and "normalized_z_feet" in goalie_rows.columns else "z_feet"
	coordinate_source = "normalized_xy_feet" if x_column.startswith("normalized_") else "raw_xy_feet"

	goalie_rows["trajectory_x_feet"] = goalie_rows[x_column]
	goalie_rows["trajectory_y_feet"] = goalie_rows[y_column]
	if z_column in goalie_rows.columns:
		goalie_rows["trajectory_z_feet"] = goalie_rows[z_column]
	goalie_rows["trajectory_coordinate_source"] = coordinate_source
	goalie_rows["goalie_heuristic_conditions"] = str(goalie_heuristic_conditions())
	goalie_rows = attach_stable_goalie_identity(goalie_rows)

	if add_frame_sequence:
		goalie_rows["frame_sequence"] = build_tracking_frame_sequence(goalie_rows)
		return sort_goalie_trajectory_frames(goalie_rows)

	return goalie_rows.reset_index(drop=True)


def summarize_goalie_trajectories(goalie_trajectories: pd.DataFrame) -> dict[str, object]:
	"""Summarize frame-level goalie trajectory rows for audit and validation."""

	validate_goalie_trajectory_input(goalie_trajectories)
	team_distribution = {
		str(team): int(count)
		for team, count in goalie_trajectories["team"].value_counts(dropna=False).items()
	}
	usable_player_ids = goalie_trajectories["player_id"].dropna()
	coordinate_source = "unknown"
	if "trajectory_coordinate_source" in goalie_trajectories.columns and not goalie_trajectories.empty:
		coordinate_source = str(goalie_trajectories["trajectory_coordinate_source"].iloc[0])

	return {
		"frame_level_goalie_rows": int(len(goalie_trajectories)),
		"unique_goalie_candidate_player_ids": int(usable_player_ids.nunique()) if not usable_player_ids.empty else 0,
		"stabilized_goalie_identity_groups": int(goalie_trajectories["stability_rule_passed"].fillna(False).groupby([goalie_trajectories["game"], goalie_trajectories["team"]]).any().sum()) if "stability_rule_passed" in goalie_trajectories.columns and not goalie_trajectories.empty else 0,
		"stabilized_goalie_frame_rows": int(goalie_trajectories["is_stabilized_goalie_frame"].fillna(False).sum()) if "is_stabilized_goalie_frame" in goalie_trajectories.columns else 0,
		"team_distribution": team_distribution,
		"trajectory_coordinate_source": coordinate_source,
		"heuristic_conditions": goalie_heuristic_conditions(),
		"note": GOALIE_TRAJECTORY_NOTE,
		"goalie_identity_inference_note": GOALIE_IDENTITY_STABILITY_NOTE,
	}