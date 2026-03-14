from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


_PERIOD_LABEL_MAP: dict[str, float] = {
	"OT": 4.0,
}


def _clock_to_remaining_seconds(clock_value: object) -> float:
	if pd.isna(clock_value):
		return np.nan

	minutes, seconds = str(clock_value).split(":", maxsplit=1)
	return (int(minutes) * 60) + int(seconds)


def _period_to_float(period_value: object) -> float:
	"""Convert a period label to its numeric ordering equivalent.

	Numeric values are cast directly. The label 'OT' is mapped to 4.0,
	placing overtime frames after regulation period 3 in a monotonic sequence.
	Other unrecognized labels raise ValueError.
	"""
	if pd.isna(period_value):
		return float("nan")
	s = str(period_value).strip().upper()
	if s in _PERIOD_LABEL_MAP:
		return _PERIOD_LABEL_MAP[s]
	return float(s)


def build_tracking_frame_sequence(frame: pd.DataFrame) -> pd.Series:
	required_columns = {"image_id", "period", "game_clock"}
	missing_columns = sorted(required_columns.difference(frame.columns))
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"tracking frame is missing columns required for frame ordering: {missing}")

	period_numeric = frame["period"].map(_period_to_float)
	remaining_seconds = frame["game_clock"].map(_clock_to_remaining_seconds)
	image_suffix = frame["image_id"].astype(str).str.extract(r"_(\d+)$", expand=False).astype(float)
	return ((period_numeric - 1.0) * 1200.0) + (1200.0 - remaining_seconds) + (image_suffix / 1_000_000.0)


def compute_motion_efficiency(
	tracking_frame: pd.DataFrame,
	group_columns: Sequence[str] = ("game", "team", "player_id"),
	x_column: str = "normalized_x_feet",
	y_column: str = "normalized_y_feet",
) -> pd.DataFrame:
	"""Summarize grouped motion using path length, displacement, endpoints, and efficiency."""

	required_columns = set(group_columns).union({"image_id", "period", "game_clock", x_column, y_column})
	missing_columns = sorted(required_columns.difference(tracking_frame.columns))
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"tracking_frame is missing columns required for motion efficiency: {missing}")

	working = tracking_frame.dropna(subset=list(group_columns) + [x_column, y_column]).copy()
	if working.empty:
		return pd.DataFrame(
			columns=[
				*group_columns,
				"sample_count",
				"frame_count",
				"path_length_feet",
				"net_displacement_feet",
				"straight_line_distance_feet",
				"start_x_feet",
				"start_y_feet",
				"end_x_feet",
				"end_y_feet",
				"motion_efficiency",
			]
		)

	working["frame_sequence"] = build_tracking_frame_sequence(working)
	working = working.sort_values(by=[*group_columns, "frame_sequence"], kind="stable")
	working["dx"] = working.groupby(list(group_columns), sort=False)[x_column].diff().fillna(0.0)
	working["dy"] = working.groupby(list(group_columns), sort=False)[y_column].diff().fillna(0.0)
	working["step_distance_feet"] = np.hypot(working["dx"], working["dy"])

	grouped = working.groupby(list(group_columns), dropna=False, sort=False)
	summary = grouped.agg(
		sample_count=(x_column, "size"),
		path_length_feet=("step_distance_feet", "sum"),
		start_x_feet=(x_column, "first"),
		start_y_feet=(y_column, "first"),
		end_x_feet=(x_column, "last"),
		end_y_feet=(y_column, "last"),
	).reset_index()
	summary["frame_count"] = summary["sample_count"]
	summary["net_displacement_feet"] = np.hypot(
		summary["end_x_feet"] - summary["start_x_feet"],
		summary["end_y_feet"] - summary["start_y_feet"],
	)
	summary["straight_line_distance_feet"] = summary["net_displacement_feet"]
	summary["motion_efficiency"] = np.where(
		summary["path_length_feet"] > 0.0,
		summary["net_displacement_feet"] / summary["path_length_feet"],
		1.0,
	)
	return summary[
		[
			*group_columns,
			"sample_count",
			"frame_count",
			"path_length_feet",
			"net_displacement_feet",
			"straight_line_distance_feet",
			"start_x_feet",
			"start_y_feet",
			"end_x_feet",
			"end_y_feet",
			"motion_efficiency",
		]
	]


def categorize_rebound_control(
	*,
	controlled_by_goalie: bool = False,
	controlled_by_defending_team: bool = False,
	immediate_follow_up_shot: bool = False,
) -> str:
	"""Conservative rebound control categorization.

	Only classify a rebound as controlled when control is explicit. Ambiguous cases
	stay unknown instead of overstating control quality.
	"""

	if controlled_by_goalie:
		return "freeze"
	if immediate_follow_up_shot:
		return "uncontrolled"
	if controlled_by_defending_team:
		return "controlled"
	return "unknown"


def compute_squareness(*args: object, **kwargs: object) -> pd.DataFrame:
	raise NotImplementedError(
		"Squareness is intentionally left as an honest placeholder until body-angle or net-facing data is available."
	)
