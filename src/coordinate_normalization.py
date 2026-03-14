from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema_alignment import align_camera_orientation_schema


def load_camera_orientations(path: str | Path) -> pd.DataFrame:
	"""Load and canonically rename the camera orientation lookup file."""

	frame = pd.read_csv(path)
	return align_camera_orientation_schema(frame)


def attach_camera_orientation(
	tracking_frame: pd.DataFrame,
	camera_orientations: pd.DataFrame,
) -> pd.DataFrame:
	"""Attach per-game camera metadata using the explicit game join key."""

	if "game" not in tracking_frame.columns:
		raise ValueError("tracking_frame must contain a canonical 'game' column before orientation join")
	if "game" not in camera_orientations.columns:
		raise ValueError("camera_orientations must contain a canonical 'game' column")

	orientation_lookup = camera_orientations.drop_duplicates(subset=["game"])
	merged = tracking_frame.merge(orientation_lookup, on="game", how="left", validate="m:1")
	return merged


def normalize_tracking_coordinates(
	tracking_frame: pd.DataFrame,
	camera_orientations: pd.DataFrame | None = None,
) -> pd.DataFrame:
	"""Attach normalization metadata without applying hard-coded transforms yet.

	The current project constraint is to keep normalization explicit and staged.
	We use camera orientation data as metadata, but we do not flip or rotate rink
	coordinates until the transformation rules are finalized.
	"""

	normalized = tracking_frame.copy()
	if camera_orientations is not None:
		normalized = attach_camera_orientation(normalized, camera_orientations)
	elif "goalie_team_on_right_side_of_rink_1st_period" not in normalized.columns:
		normalized["goalie_team_on_right_side_of_rink_1st_period"] = pd.NA

	normalized["normalized_x_feet"] = normalized["x_feet"]
	normalized["normalized_y_feet"] = normalized["y_feet"]
	normalized["normalized_z_feet"] = normalized.get("z_feet", pd.Series(pd.NA, index=normalized.index))
	normalized["normalization_source"] = "camera_orientations_placeholder"
	normalized["coordinates_flipped"] = False
	normalized["normalization_ready"] = normalized[
		"goalie_team_on_right_side_of_rink_1st_period"
	].notna()
	return normalized
