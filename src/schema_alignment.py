from __future__ import annotations

import re
from typing import Final

import pandas as pd


TRACKING_SCHEMA_MAP: Final[dict[str, str]] = {
	"Image Id": "image_id",
	"Period": "period",
	"Game Clock": "game_clock",
	"Player or Puck": "entity_type",
	"Team": "team",
	"Player Id": "player_id",
	"Player Jersey Number": "player_jersey_number",
	"Rink Location X (Feet)": "x_feet",
	"Rink Location Y (Feet)": "y_feet",
	"Rink Location Z (Feet)": "z_feet",
	"Goal Score": "goal_score",
}

EVENT_SCHEMA_MAP: Final[dict[str, str]] = {
	"Date": "date",
	"Home_Team": "home_team",
	"Away_Team": "away_team",
	"Period": "period",
	"Clock": "clock",
	"Home_Team_Skaters": "home_team_skaters",
	"Away_Team_Skaters": "away_team_skaters",
	"Home_Team_Goals": "home_team_goals",
	"Away_Team_Goals": "away_team_goals",
	"Team": "team",
	"Player_Id": "player_id",
	"Event": "event_type",
	"X_Coordinate": "x_coordinate",
	"Y_Coordinate": "y_coordinate",
	"Detail_1": "detail_1",
	"Detail_2": "detail_2",
	"Detail_3": "detail_3",
	"Detail_4": "detail_4",
	"Player_Id_2": "secondary_player_id",
	"X_Coordinate_2": "secondary_x_coordinate",
	"Y_Coordinate_2": "secondary_y_coordinate",
}

SHIFT_SCHEMA_MAP: Final[dict[str, str]] = {
	"Date": "date",
	"Home_Team": "home_team",
	"Away_Team": "away_team",
	"Team": "team",
	"Player_Id": "player_id",
	"shift_number": "shift_number",
	"period": "period",
	"start_clock": "start_clock",
	"end_clock": "end_clock",
	"shift_length": "shift_length",
}

CAMERA_ORIENTATION_SCHEMA_MAP: Final[dict[str, str]] = {
	"Game": "game",
	"GoalieTeamOnRightSideOfRink1stPeriod": "goalie_team_on_right_side_of_rink_1st_period",
}

_IMAGE_ID_GAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<game>.+)_\d+$")
GOALIE_MIN_PLAYER_ROWS_PER_TEAM_FRAME: Final[int] = 5
GOALIE_MIN_ABS_X_FEET: Final[float] = 80.0
GOALIE_MAX_ABS_Y_FEET: Final[float] = 15.0


def _align_dataframe_schema(
	frame: pd.DataFrame,
	schema_map: dict[str, str],
	dataset_name: str,
) -> pd.DataFrame:
	missing_columns = [column for column in schema_map if column not in frame.columns]
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"{dataset_name} is missing required columns: {missing}")

	aligned = frame.rename(columns=schema_map).copy()
	return aligned


def extract_game_from_image_id(image_id: str) -> str:
	"""Apply the explicit Image Id -> Game join rule used in this project.

	Example: "2025-10-28 Team C @ Team A_000158" -> "2025-10-28 Team C @ Team A"
	"""

	if pd.isna(image_id):
		raise ValueError("image_id cannot be null when deriving game")

	match = _IMAGE_ID_GAME_PATTERN.match(str(image_id).strip())
	if not match:
		raise ValueError(f"image_id does not match the expected game join rule: {image_id}")
	return match.group("game")


def align_tracking_schema(frame: pd.DataFrame) -> pd.DataFrame:
	aligned = _align_dataframe_schema(frame, TRACKING_SCHEMA_MAP, "tracking data")
	aligned["game"] = aligned["image_id"].map(extract_game_from_image_id)
	return aligned


def align_event_schema(frame: pd.DataFrame) -> pd.DataFrame:
	return _align_dataframe_schema(frame, EVENT_SCHEMA_MAP, "event data")


def align_shift_schema(frame: pd.DataFrame) -> pd.DataFrame:
	return _align_dataframe_schema(frame, SHIFT_SCHEMA_MAP, "shift data")


def align_camera_orientation_schema(frame: pd.DataFrame) -> pd.DataFrame:
	return _align_dataframe_schema(frame, CAMERA_ORIENTATION_SCHEMA_MAP, "camera orientation data")


def goalie_heuristic_conditions() -> dict[str, float | int | str]:
	return {
		"entity_type": "player",
		"min_player_rows_per_team_frame": GOALIE_MIN_PLAYER_ROWS_PER_TEAM_FRAME,
		"min_abs_x_feet": GOALIE_MIN_ABS_X_FEET,
		"max_abs_y_feet": GOALIE_MAX_ABS_Y_FEET,
		"selection_rule": "pick the deepest eligible player per image_id/team by abs_x, then lower abs_y, then player_id",
	}


def identify_goalie_rows(frame: pd.DataFrame) -> pd.Series:
	"""Identify conservative frame-level goalie candidates.

	Observed tracking data has many sparse image/team groups, so this heuristic only
	considers frames with enough tracked players to make a depth comparison useful.
	Within those groups it requires a goalie-like location near the end boards and
	relatively central laterally, then picks the deepest eligible player.
	"""

	required_columns = {"image_id", "team", "entity_type", "x_feet", "y_feet"}
	missing_columns = sorted(required_columns.difference(frame.columns))
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"tracking frame is missing columns required for goalie identification: {missing}")

	is_goalie = pd.Series(False, index=frame.index, dtype=bool)
	player_rows = frame[frame["entity_type"].astype(str).str.casefold() == "player"].copy()
	if player_rows.empty:
		return is_goalie

	player_rows["player_count_in_team_frame"] = player_rows.groupby(["image_id", "team"])["image_id"].transform("size")
	player_rows["player_id"] = player_rows.get("player_id")
	player_rows["abs_x_feet"] = player_rows["x_feet"].abs()
	player_rows["abs_y_feet"] = player_rows["y_feet"].abs()
	player_rows = player_rows[
		(player_rows["player_count_in_team_frame"] >= GOALIE_MIN_PLAYER_ROWS_PER_TEAM_FRAME)
		& (player_rows["abs_x_feet"] >= GOALIE_MIN_ABS_X_FEET)
		& (player_rows["abs_y_feet"] <= GOALIE_MAX_ABS_Y_FEET)
	].copy()
	if player_rows.empty:
		return is_goalie

	player_rows = player_rows.sort_values(
		by=["image_id", "team", "abs_x_feet", "abs_y_feet", "player_id"],
		ascending=[True, True, False, True, True],
		kind="stable",
	)

	goalie_indexes = player_rows.groupby(["image_id", "team"], sort=False).head(1).index
	is_goalie.loc[goalie_indexes] = True
	return is_goalie


def summarize_goalie_candidates(frame: pd.DataFrame, goalie_mask: pd.Series) -> dict[str, object]:
	goalie_rows = frame.loc[goalie_mask].copy()
	team_distribution: dict[str, int] = {}
	if "team" in goalie_rows.columns:
		team_distribution = {
			str(team): int(count)
			for team, count in goalie_rows["team"].value_counts(dropna=False).items()
		}

	unique_player_id_count: int | str = "unavailable"
	if "player_id" in goalie_rows.columns:
		usable_player_ids = goalie_rows["player_id"].dropna()
		unique_player_id_count = int(usable_player_ids.nunique()) if not usable_player_ids.empty else 0

	return {
		"frame_level_goalie_candidate_rows": int(goalie_mask.sum()),
		"unique_goalie_candidate_player_ids": unique_player_id_count,
		"goalie_candidate_team_distribution": team_distribution,
		"heuristic_conditions": goalie_heuristic_conditions(),
	}
