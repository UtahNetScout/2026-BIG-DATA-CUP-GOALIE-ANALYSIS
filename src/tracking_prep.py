from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .coordinate_normalization import load_camera_orientations, normalize_tracking_coordinates
from .schema_alignment import align_tracking_schema, identify_goalie_rows


def load_tracking_file(path: str | Path) -> pd.DataFrame:
	frame = pd.read_csv(path)
	aligned = align_tracking_schema(frame)
	aligned["source_file"] = Path(path).name
	return aligned


def load_tracking_files(paths: Iterable[str | Path]) -> pd.DataFrame:
	frames = [load_tracking_file(path) for path in paths]
	if not frames:
		return pd.DataFrame()
	return pd.concat(frames, ignore_index=True)


def load_tracking_directory(directory: str | Path, pattern: str = "*Tracking_*.csv") -> pd.DataFrame:
	tracking_paths = sorted(Path(directory).glob(pattern))
	return load_tracking_files(tracking_paths)


def split_tracking_by_period(tracking_frame: pd.DataFrame) -> dict[object, pd.DataFrame]:
	if "period" not in tracking_frame.columns:
		raise ValueError("tracking_frame must contain a 'period' column")

	period_splits: dict[object, pd.DataFrame] = {}
	for period_value, period_frame in tracking_frame.groupby("period", sort=True):
		period_splits[period_value] = period_frame.copy()
	return period_splits


def split_tracking_by_game(tracking_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
	if "game" not in tracking_frame.columns:
		raise ValueError("tracking_frame must contain a 'game' column")

	return {game: game_frame.copy() for game, game_frame in tracking_frame.groupby("game", sort=True)}


def prepare_tracking_data(
	tracking_paths: Iterable[str | Path],
	camera_orientations_path: str | Path | None = None,
	include_goalie_flags: bool = True,
) -> pd.DataFrame:
	tracking_frame = load_tracking_files(tracking_paths)
	if tracking_frame.empty:
		return tracking_frame

	camera_orientations = None
	if camera_orientations_path is not None:
		camera_orientations = load_camera_orientations(camera_orientations_path)

	prepared = normalize_tracking_coordinates(tracking_frame, camera_orientations)
	if include_goalie_flags:
		prepared["is_goalie"] = identify_goalie_rows(prepared)
	return prepared
