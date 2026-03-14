import unittest

import pandas as pd

from src.schema_alignment import align_tracking_schema, identify_goalie_rows


def _tracking_schema_row(image_id: str = "2025-10-28 Team C @ Team A_000158") -> dict[str, object]:
	return {
		"Image Id": image_id,
		"Period": 1,
		"Game Clock": "19:58",
		"Player or Puck": "player",
		"Team": "Home",
		"Player Id": 30,
		"Player Jersey Number": 30,
		"Rink Location X (Feet)": 88.0,
		"Rink Location Y (Feet)": 4.0,
		"Rink Location Z (Feet)": 0.0,
		"Goal Score": 0,
	}


class AlignTrackingSchemaTests(unittest.TestCase):
	def test_align_tracking_schema_renames_columns_and_derives_game(self) -> None:
		frame = pd.DataFrame([_tracking_schema_row()])

		aligned = align_tracking_schema(frame)

		self.assertIn("image_id", aligned.columns)
		self.assertIn("game", aligned.columns)
		self.assertEqual(aligned.loc[0, "image_id"], "2025-10-28 Team C @ Team A_000158")
		self.assertEqual(aligned.loc[0, "game"], "2025-10-28 Team C @ Team A")

	def test_align_tracking_schema_raises_when_required_columns_are_missing(self) -> None:
		frame = pd.DataFrame([_tracking_schema_row()]).drop(columns=["Team"])

		with self.assertRaisesRegex(ValueError, "missing required columns: Team"):
			align_tracking_schema(frame)


class IdentifyGoalieRowsTests(unittest.TestCase):
	def test_identify_goalie_rows_selects_deepest_eligible_player_per_team_frame(self) -> None:
		frame = pd.DataFrame(
			[
				{"image_id": "game_000001", "team": "Home", "entity_type": "player", "player_id": 10, "x_feet": 82.0, "y_feet": 6.0},
				{"image_id": "game_000001", "team": "Home", "entity_type": "player", "player_id": 11, "x_feet": 84.0, "y_feet": 5.0},
				{"image_id": "game_000001", "team": "Home", "entity_type": "player", "player_id": 12, "x_feet": 88.0, "y_feet": 4.0},
				{"image_id": "game_000001", "team": "Home", "entity_type": "player", "player_id": 13, "x_feet": 90.0, "y_feet": 3.0},
				{"image_id": "game_000001", "team": "Home", "entity_type": "player", "player_id": 14, "x_feet": 78.0, "y_feet": 2.0},
				{"image_id": "game_000001", "team": "Home", "entity_type": "puck", "player_id": None, "x_feet": 0.0, "y_feet": 0.0},
			]
		)

		goalie_mask = identify_goalie_rows(frame)

		self.assertEqual(goalie_mask.sum(), 1)
		self.assertTrue(goalie_mask.loc[3])
