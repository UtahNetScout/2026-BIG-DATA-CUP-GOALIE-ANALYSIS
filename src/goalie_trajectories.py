from __future__ import annotations

from typing import Final, Sequence

import numpy as np
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

# When player_id is too fragmented (ephemeral tracking IDs are common), fall back to
# stabilizing by player_jersey_number.  The same share/margin thresholds apply so the
# heuristic remains equally conservative regardless of which identifier is used.
STABLE_GOALIE_JERSEY_STABILIZATION_NOTE: Final[str] = (
	"player_id was too fragmented for cross-frame stabilization; jersey-number-based fallback used instead."
)


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

	The rule first tries to resolve a stabilized identity by player_id.  In many tracking
	data sources player IDs are ephemeral (the same physical goalie is assigned a different
	ID every short tracking segment), so a jersey-number-based fallback is attempted when
	the player_id approach fails.  The same share/margin thresholds apply in both cases.
	Ambiguous groups remain unresolved.

	Returned columns always include the player_id stabilization diagnostics
	(``stability_rule_passed_by_player_id``, ``stability_player_share``,
	``stability_margin_frames``) and, when ``player_jersey_number`` is available, the
	jersey-number diagnostics (``stability_by_jersey_number_rule_passed``,
	``stability_jersey_share``, ``stability_jersey_margin_frames``,
	``frame_level_unique_candidate_jersey_numbers``, ``stabilized_goalie_jersey_number``).
	The combined ``stability_rule_passed`` is True when either approach succeeds.
	"""

	required_columns = {"game", "team", "player_id"}
	missing_columns = sorted(required_columns.difference(goalie_rows.columns))
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"goalie_rows is missing columns required for identity stability inference: {missing}")

	_empty_columns = [
		"game",
		"team",
		"frame_level_candidate_rows",
		"frame_level_unique_candidate_player_ids",
		"stabilized_goalie_player_id",
		"stabilized_goalie_jersey_number",
		"stabilized_goalie_frame_count",
		"stability_player_share",
		"stability_margin_frames",
		"stability_rule_passed_by_player_id",
		"stability_by_jersey_number_rule_passed",
		"frame_level_unique_candidate_jersey_numbers",
		"stability_jersey_share",
		"stability_jersey_margin_frames",
		"stability_rule_passed",
		"goalie_identity_inference_level",
		"goalie_identity_inference_note",
	]

	base_groups = goalie_rows.loc[:, ["game", "team"]].drop_duplicates().reset_index(drop=True)
	if base_groups.empty:
		return pd.DataFrame(columns=_empty_columns)

	count_frame = goalie_rows.dropna(subset=["player_id"]).copy()
	if count_frame.empty:
		result = base_groups.copy()
		result["frame_level_candidate_rows"] = 0
		result["frame_level_unique_candidate_player_ids"] = 0
		result["stabilized_goalie_player_id"] = pd.NA
		result["stabilized_goalie_jersey_number"] = pd.NA
		result["stabilized_goalie_frame_count"] = 0
		result["stability_player_share"] = 0.0
		result["stability_margin_frames"] = 0
		result["stability_rule_passed_by_player_id"] = False
		result["stability_by_jersey_number_rule_passed"] = False
		result["frame_level_unique_candidate_jersey_numbers"] = 0
		result["stability_jersey_share"] = pd.NA
		result["stability_jersey_margin_frames"] = pd.NA
		result["stability_rule_passed"] = False
		result["goalie_identity_inference_level"] = "unresolved"
		result["goalie_identity_inference_note"] = GOALIE_IDENTITY_STABILITY_NOTE
		return result

	# ---- Player-id-based stabilization (primary path) ----
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
	stability["stability_rule_passed_by_player_id"] = (
		(stability["leading_candidate_frame_count"] >= STABLE_GOALIE_ID_MIN_CANDIDATE_FRAMES)
		& (stability["stability_player_share"] >= STABLE_GOALIE_ID_MIN_SHARE)
		& (stability["stability_margin_frames"] >= STABLE_GOALIE_ID_MIN_MARGIN_FRAMES)
	)
	stability["stabilized_goalie_player_id"] = stability["leading_player_id"].where(
		stability["stability_rule_passed_by_player_id"], pd.NA
	)

	# ---- Jersey-number-based stabilization (fallback when player_id is too fragmented) ----
	# Tracking systems often assign ephemeral player IDs (the same goalie receives a new ID
	# each tracking segment), making player_id counts too spread out to cross the share
	# threshold.  Jersey number is stable across frames and provides a reliable fallback.
	stability["stabilized_goalie_jersey_number"] = pd.NA
	stability["stability_by_jersey_number_rule_passed"] = False
	stability["frame_level_unique_candidate_jersey_numbers"] = 0
	stability["stability_jersey_share"] = pd.NA
	stability["stability_jersey_margin_frames"] = pd.NA

	if "player_jersey_number" in goalie_rows.columns:
		jersey_frame = goalie_rows.dropna(subset=["player_jersey_number"]).copy()
		if not jersey_frame.empty:
			jersey_counts = (
				jersey_frame.groupby(["game", "team", "player_jersey_number"], dropna=False)
				.size()
				.rename("jersey_candidate_frame_count")
				.reset_index()
			)
			jersey_counts = jersey_counts.sort_values(
				by=["game", "team", "jersey_candidate_frame_count", "player_jersey_number"],
				ascending=[True, True, False, True],
				kind="stable",
			)
			jersey_group_info = jersey_counts.groupby(["game", "team"], sort=False).agg(
				frame_level_unique_candidate_jersey_numbers=("player_jersey_number", "nunique"),
			).reset_index()
			jersey_top = jersey_counts.groupby(["game", "team"], sort=False).head(2).copy()
			jersey_top["jersey_rank"] = jersey_top.groupby(["game", "team"], sort=False).cumcount() + 1
			jersey_leading = jersey_top[jersey_top["jersey_rank"] == 1].rename(
				columns={
					"player_jersey_number": "leading_jersey_number",
					"jersey_candidate_frame_count": "leading_jersey_frame_count",
				}
			)
			jersey_runner_up = jersey_top[jersey_top["jersey_rank"] == 2].rename(
				columns={"jersey_candidate_frame_count": "runner_up_jersey_frame_count"}
			)

			stability = stability.merge(jersey_group_info, on=["game", "team"], how="left", suffixes=("", "_jgi"))
			# Resolve column suffix if the name collided with an earlier initialization
			if "frame_level_unique_candidate_jersey_numbers_jgi" in stability.columns:
				stability["frame_level_unique_candidate_jersey_numbers"] = stability["frame_level_unique_candidate_jersey_numbers_jgi"].fillna(0).astype(int)
				stability = stability.drop(columns=["frame_level_unique_candidate_jersey_numbers_jgi"])
			else:
				stability["frame_level_unique_candidate_jersey_numbers"] = stability["frame_level_unique_candidate_jersey_numbers"].fillna(0).astype(int)

			stability = stability.merge(
				jersey_leading.loc[:, ["game", "team", "leading_jersey_number", "leading_jersey_frame_count"]],
				on=["game", "team"],
				how="left",
			).merge(
				jersey_runner_up.loc[:, ["game", "team", "runner_up_jersey_frame_count"]],
				on=["game", "team"],
				how="left",
			)
			stability["runner_up_jersey_frame_count"] = stability["runner_up_jersey_frame_count"].fillna(0)
			# Share is computed against the total player-id candidate rows so the two
			# approaches are comparable and the jersey threshold is equally conservative.
			stability["stability_jersey_share"] = (
				stability["leading_jersey_frame_count"] / stability["frame_level_candidate_rows"]
			)
			stability["stability_jersey_margin_frames"] = (
				stability["leading_jersey_frame_count"] - stability["runner_up_jersey_frame_count"]
			)
			jersey_rule_passes = (
				(stability["leading_jersey_frame_count"].fillna(0) >= STABLE_GOALIE_ID_MIN_CANDIDATE_FRAMES)
				& (stability["stability_jersey_share"].fillna(0) >= STABLE_GOALIE_ID_MIN_SHARE)
				& (stability["stability_jersey_margin_frames"].fillna(0) >= STABLE_GOALIE_ID_MIN_MARGIN_FRAMES)
			)
			# Jersey fallback only fires when the player_id primary path failed.
			stability["stability_by_jersey_number_rule_passed"] = (
				~stability["stability_rule_passed_by_player_id"] & jersey_rule_passes
			)
			stability["stabilized_goalie_jersey_number"] = stability["leading_jersey_number"].where(
				stability["stability_by_jersey_number_rule_passed"], pd.NA
			)

	# ---- Combine ----
	stability["stability_rule_passed"] = (
		stability["stability_rule_passed_by_player_id"] | stability["stability_by_jersey_number_rule_passed"]
	)

	# stabilized_goalie_frame_count: use jersey count when the jersey path fired
	player_id_frames = stability["leading_candidate_frame_count"].where(
		stability["stability_rule_passed_by_player_id"], 0
	)
	if "leading_jersey_frame_count" in stability.columns:
		jersey_frames = stability["leading_jersey_frame_count"].where(
			stability["stability_by_jersey_number_rule_passed"], 0
		)
		stability["stabilized_goalie_frame_count"] = player_id_frames + jersey_frames
	else:
		stability["stabilized_goalie_frame_count"] = player_id_frames

	stability["goalie_identity_inference_level"] = np.select(
		[
			stability["stability_rule_passed_by_player_id"],
			stability["stability_by_jersey_number_rule_passed"],
		],
		["stabilized", "stabilized_by_jersey_number"],
		default="unresolved",
	)
	stability["goalie_identity_inference_note"] = GOALIE_IDENTITY_STABILITY_NOTE

	return_columns = [c for c in _empty_columns if c in stability.columns]
	return stability[return_columns]


def attach_stable_goalie_identity(goalie_rows: pd.DataFrame) -> pd.DataFrame:
	"""Annotate frame-level goalie rows with conservative cross-frame identity inference."""

	stability = infer_stable_goalie_identities(goalie_rows)
	annotated = goalie_rows.merge(stability, on=["game", "team"], how="left", validate="m:1")

	# A frame is stabilized when its player_id matches the resolved player_id identity …
	player_id_match = annotated["player_id"].eq(annotated["stabilized_goalie_player_id"]).fillna(False)

	# … or, when the player_id path failed, when its jersey number matches the resolved
	# jersey-number identity (jersey-based fallback).
	jersey_match = pd.Series(False, index=annotated.index, dtype=bool)
	if "stabilized_goalie_jersey_number" in annotated.columns and "player_jersey_number" in annotated.columns:
		jersey_match = (
			annotated["player_jersey_number"]
			.eq(annotated["stabilized_goalie_jersey_number"])
			.fillna(False)
		)

	annotated["is_stabilized_goalie_frame"] = player_id_match | jersey_match
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