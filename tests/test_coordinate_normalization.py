import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.coordinate_normalization import (
	attach_camera_orientation,
	load_camera_orientations,
	normalize_tracking_coordinates,
)


class LoadCameraOrientationsTests(unittest.TestCase):
	def test_load_camera_orientations_aligns_expected_columns(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			csv_path = Path(temp_dir) / "camera_orientations.csv"
			pd.DataFrame(
				[
					{
						"Game": "2025-10-24 Team E @ Team D",
						"GoalieTeamOnRightSideOfRink1stPeriod": "Home",
					}
				]
			).to_csv(csv_path, index=False)

			loaded = load_camera_orientations(csv_path)

		self.assertEqual(loaded.columns.tolist(), ["game", "goalie_team_on_right_side_of_rink_1st_period"])
		self.assertEqual(loaded.loc[0, "game"], "2025-10-24 Team E @ Team D")
		self.assertEqual(loaded.loc[0, "goalie_team_on_right_side_of_rink_1st_period"], "Home")


class AttachCameraOrientationTests(unittest.TestCase):
	def test_attach_camera_orientation_merges_by_game(self) -> None:
		tracking_frame = pd.DataFrame(
			[
				{"game": "2025-10-24 Team E @ Team D", "image_id": "g_1", "x_feet": 10.0, "y_feet": 2.0},
				{"game": "2025-10-28 Team C @ Team A", "image_id": "g_2", "x_feet": -5.0, "y_feet": 1.0},
			]
		)
		camera_orientations = pd.DataFrame(
			[
				{
					"game": "2025-10-24 Team E @ Team D",
					"goalie_team_on_right_side_of_rink_1st_period": "Home",
				},
				{
					"game": "2025-10-28 Team C @ Team A",
					"goalie_team_on_right_side_of_rink_1st_period": "Away",
				},
			]
		)

		merged = attach_camera_orientation(tracking_frame, camera_orientations)

		self.assertEqual(merged["goalie_team_on_right_side_of_rink_1st_period"].tolist(), ["Home", "Away"])

	def test_attach_camera_orientation_requires_game_column(self) -> None:
		tracking_frame = pd.DataFrame([{"image_id": "g_1", "x_feet": 10.0, "y_feet": 2.0}])
		camera_orientations = pd.DataFrame(
			[
				{
					"game": "2025-10-24 Team E @ Team D",
					"goalie_team_on_right_side_of_rink_1st_period": "Home",
				}
			]
		)

		with self.assertRaisesRegex(ValueError, "tracking_frame must contain a canonical 'game' column"):
			attach_camera_orientation(tracking_frame, camera_orientations)


class NormalizeTrackingCoordinatesTests(unittest.TestCase):
	def test_normalize_tracking_coordinates_preserves_coordinates_and_sets_ready_flag_with_metadata(self) -> None:
		tracking_frame = pd.DataFrame(
			[
				{
					"game": "2025-10-24 Team E @ Team D",
					"image_id": "g_1",
					"x_feet": 10.0,
					"y_feet": -2.5,
					"z_feet": 1.0,
				}
			]
		)
		camera_orientations = pd.DataFrame(
			[
				{
					"game": "2025-10-24 Team E @ Team D",
					"goalie_team_on_right_side_of_rink_1st_period": "Home",
				}
			]
		)

		normalized = normalize_tracking_coordinates(tracking_frame, camera_orientations)

		self.assertEqual(normalized.loc[0, "normalized_x_feet"], 10.0)
		self.assertEqual(normalized.loc[0, "normalized_y_feet"], -2.5)
		self.assertEqual(normalized.loc[0, "normalized_z_feet"], 1.0)
		self.assertEqual(normalized.loc[0, "normalization_source"], "camera_orientations_placeholder")
		self.assertFalse(bool(normalized.loc[0, "coordinates_flipped"]))
		self.assertTrue(bool(normalized.loc[0, "normalization_ready"]))

	def test_normalize_tracking_coordinates_marks_not_ready_without_metadata(self) -> None:
		tracking_frame = pd.DataFrame(
			[
				{"game": "2025-10-24 Team E @ Team D", "image_id": "g_1", "x_feet": 10.0, "y_feet": -2.5}
			]
		)

		normalized = normalize_tracking_coordinates(tracking_frame)

		self.assertTrue(pd.isna(normalized.loc[0, "goalie_team_on_right_side_of_rink_1st_period"]))
		self.assertEqual(normalized.loc[0, "normalized_x_feet"], 10.0)
		self.assertEqual(normalized.loc[0, "normalized_y_feet"], -2.5)
		self.assertTrue(pd.isna(normalized.loc[0, "normalized_z_feet"]))
		self.assertFalse(bool(normalized.loc[0, "normalization_ready"]))
