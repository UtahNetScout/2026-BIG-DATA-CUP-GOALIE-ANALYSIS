from __future__ import annotations

from typing import Final

import pandas as pd

from .metrics import _period_to_float, compute_motion_efficiency


EVENT_ALIGNMENT_NOTE: Final[str] = (
	"Event alignment is a first-pass second-level window join on derived game, period, and clock timing. "
	"It is suitable for narrow audit windows but does not imply frame-exact event causality."
)


def _clock_to_remaining_seconds(clock_value: object) -> int:
	if pd.isna(clock_value):
		raise ValueError("clock value cannot be null for event alignment")

	minutes, seconds = str(clock_value).split(":", maxsplit=1)
	return (int(minutes) * 60) + int(seconds)


def derive_event_game_key(event_frame: pd.DataFrame) -> pd.Series:
	"""Derive the canonical tracking-style game key from aligned event rows."""

	required_columns = {"date", "home_team", "away_team"}
	missing_columns = sorted(required_columns.difference(event_frame.columns))
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"event_frame is missing columns required to derive game: {missing}")

	return (
		event_frame["date"].astype(str).str.strip()
		+ " "
		+ event_frame["away_team"].astype(str).str.strip()
		+ " @ "
		+ event_frame["home_team"].astype(str).str.strip()
	)


def validate_event_window_alignment_inputs(
	goalie_trajectories: pd.DataFrame,
	event_frame: pd.DataFrame,
) -> None:
	goalie_required_columns = {"game", "period", "game_clock", "image_id", "team"}
	event_required_columns = {"date", "home_team", "away_team", "period", "clock", "team", "event_type"}

	missing_goalie_columns = sorted(goalie_required_columns.difference(goalie_trajectories.columns))
	if missing_goalie_columns:
		missing = ", ".join(missing_goalie_columns)
		raise ValueError(f"goalie_trajectories is missing columns required for event alignment: {missing}")

	missing_event_columns = sorted(event_required_columns.difference(event_frame.columns))
	if missing_event_columns:
		missing = ", ".join(missing_event_columns)
		raise ValueError(f"event_frame is missing columns required for event alignment: {missing}")


def align_goalie_trajectories_to_event_windows(
	goalie_trajectories: pd.DataFrame,
	event_frame: pd.DataFrame,
	*,
	pre_event_seconds: int = 0,
	post_event_seconds: int = 0,
) -> pd.DataFrame:
	"""Align frame-level goalie trajectory rows to event windows using second-level timing.

	The join uses a derived event game key of `date + away_team + @ + home_team`, plus
	shared `period` and second-level elapsed time computed from event `clock` and
	tracking `game_clock`. Tracking frames remain frame-level rows; each aligned output row
	indicates that the frame fell inside the configured event window.
	"""

	if pre_event_seconds < 0 or post_event_seconds < 0:
		raise ValueError("pre_event_seconds and post_event_seconds must be non-negative")

	validate_event_window_alignment_inputs(goalie_trajectories, event_frame)

	goalie_working = goalie_trajectories.copy()
	event_working = event_frame.copy()

	goalie_working["frame_elapsed_seconds"] = (
		(goalie_working["period"].map(_period_to_float) - 1) * 1200
		+ (1200 - goalie_working["game_clock"].map(_clock_to_remaining_seconds))
	)
	event_working["game"] = derive_event_game_key(event_working)
	event_working["event_elapsed_seconds"] = (
		(event_working["period"].map(_period_to_float) - 1) * 1200
		+ (1200 - event_working["clock"].map(_clock_to_remaining_seconds))
	)
	event_working["event_window_start_seconds"] = event_working["event_elapsed_seconds"] - pre_event_seconds
	event_working["event_window_end_seconds"] = event_working["event_elapsed_seconds"] + post_event_seconds
	event_working["event_index"] = range(len(event_working))

	joined = goalie_working.merge(
		event_working,
		on=["game", "period"],
		how="inner",
		suffixes=("_goalie", "_event"),
	)
	aligned = joined[
		(joined["frame_elapsed_seconds"] >= joined["event_window_start_seconds"])
		& (joined["frame_elapsed_seconds"] <= joined["event_window_end_seconds"])
	].copy()
	if aligned.empty:
		aligned["event_alignment_note"] = pd.Series(dtype=object)
		return aligned

	aligned["event_alignment_note"] = EVENT_ALIGNMENT_NOTE
	return aligned.sort_values(
		by=["game", "period", "event_elapsed_seconds", "frame_elapsed_seconds", "image_id"],
		kind="stable",
	).reset_index(drop=True)


def summarize_aligned_goalie_motion_by_event_type(
	aligned_goalie_event_rows: pd.DataFrame,
	*,
	include_team: bool = False,
) -> pd.DataFrame:
	"""Aggregate aligned goalie-event rows into compact event-type movement summaries."""

	required_columns = {
		"game",
		"event_index",
		"event_type",
		"team_goalie",
		"player_id_goalie",
		"image_id",
		"period",
		"game_clock",
		"trajectory_x_feet",
		"trajectory_y_feet",
	}
	missing_columns = sorted(required_columns.difference(aligned_goalie_event_rows.columns))
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"aligned_goalie_event_rows is missing columns required for event-type summary: {missing}")

	group_columns = ["event_type"]
	if include_team:
		group_columns.append("team_goalie")

	if aligned_goalie_event_rows.empty:
		return pd.DataFrame(
			columns=[
				*group_columns,
				"aligned_row_count",
				"distinct_event_count",
				"motion_summary_group_count",
				"median_frame_count",
				"median_path_length_feet",
				"median_motion_efficiency",
			]
		)

	motion_summary = compute_motion_efficiency(
		aligned_goalie_event_rows,
		group_columns=("game", "event_index", "event_type", "team_goalie", "player_id_goalie"),
		x_column="trajectory_x_feet",
		y_column="trajectory_y_feet",
	)

	row_summary = aligned_goalie_event_rows.groupby(group_columns, dropna=False, sort=False).agg(
		aligned_row_count=("image_id", "size"),
		distinct_event_count=("event_index", "nunique"),
	).reset_index()
	motion_group_columns = ["event_type"]
	if include_team:
		motion_group_columns.append("team_goalie")
	motion_aggregate = motion_summary.groupby(motion_group_columns, dropna=False, sort=False).agg(
		motion_summary_group_count=("event_index", "size"),
		median_frame_count=("frame_count", "median"),
		median_path_length_feet=("path_length_feet", "median"),
		median_motion_efficiency=("motion_efficiency", "median"),
	).reset_index()
	return row_summary.merge(motion_aggregate, on=group_columns, how="left")