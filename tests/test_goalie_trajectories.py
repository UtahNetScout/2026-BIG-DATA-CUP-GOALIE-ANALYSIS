import unittest

import pandas as pd

from src.goalie_trajectories import extract_goalie_trajectories, infer_stable_goalie_identities, select_frame_level_goalie_rows, summarize_goalie_trajectories


def _team_frame_rows(*, image_id: str, game_clock: str, goalie_player_id: int, goalie_x: float, goalie_y: float) -> list[dict[str, object]]:
	rows = [
		{
			"game": "2025-10-24 Team E @ Team D",
			"image_id": image_id,
			"team": "Home",
			"period": 1,
			"game_clock": game_clock,
			"entity_type": "player",
			"player_id": goalie_player_id,
			"x_feet": goalie_x,
			"y_feet": goalie_y,
			"z_feet": 0.0,
			"normalized_x_feet": -goalie_x,
			"normalized_y_feet": goalie_y + 1.0,
			"normalized_z_feet": 0.0,
		},
	]
	for offset, player_id in enumerate([11, 12, 13, 14], start=1):
		rows.append(
			{
				"game": "2025-10-24 Team E @ Team D",
				"image_id": image_id,
				"team": "Home",
				"period": 1,
				"game_clock": game_clock,
				"entity_type": "player",
				"player_id": player_id,
				"x_feet": 75.0 + offset,
				"y_feet": 20.0 + offset,
				"z_feet": 0.0,
				"normalized_x_feet": -(75.0 + offset),
				"normalized_y_feet": 20.0 + offset,
				"normalized_z_feet": 0.0,
			}
		)
	return rows


class GoalieTrajectoryTests(unittest.TestCase):
	def test_select_frame_level_goalie_rows_returns_one_row_with_metadata(self) -> None:
		tracking_frame = pd.DataFrame(
			_team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000002",
				game_clock="19:59",
				goalie_player_id=30,
				goalie_x=90.0,
				goalie_y=4.0,
			)
		)

		goalie_rows = select_frame_level_goalie_rows(tracking_frame)

		self.assertEqual(len(goalie_rows), 1)
		self.assertEqual(int(goalie_rows.iloc[0]["player_id"]), 30)
		self.assertEqual(goalie_rows.iloc[0]["goalie_selection_level"], "frame")
		self.assertEqual(goalie_rows.iloc[0]["goalie_selector"], "heuristic")

	def test_extract_goalie_trajectories_prefers_normalized_coordinates_and_sorts_frames(self) -> None:
		tracking_frame = pd.DataFrame(
			_team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000003",
				game_clock="19:58",
				goalie_player_id=31,
				goalie_x=91.0,
				goalie_y=5.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000002",
				game_clock="19:59",
				goalie_player_id=30,
				goalie_x=90.0,
				goalie_y=4.0,
			)
		)

		goalie_trajectories = extract_goalie_trajectories(tracking_frame)

		self.assertEqual(goalie_trajectories["image_id"].tolist(), [
			"2025-10-24 Team E @ Team D_000002",
			"2025-10-24 Team E @ Team D_000003",
		])
		self.assertEqual(goalie_trajectories["trajectory_coordinate_source"].tolist(), ["normalized_xy_feet", "normalized_xy_feet"])
		self.assertEqual(goalie_trajectories["trajectory_x_feet"].tolist(), [-90.0, -91.0])
		self.assertEqual(goalie_trajectories["goalie_identity_inference_level"].tolist(), ["unresolved", "unresolved"])
		self.assertEqual(goalie_trajectories["is_stabilized_goalie_frame"].tolist(), [False, False])

	def test_infer_stable_goalie_identities_requires_clear_modal_player_per_team(self) -> None:
		tracking_frame = pd.DataFrame(
			_team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000001",
				game_clock="20:00",
				goalie_player_id=30,
				goalie_x=89.0,
				goalie_y=4.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000002",
				game_clock="19:59",
				goalie_player_id=30,
				goalie_x=90.0,
				goalie_y=4.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000003",
				game_clock="19:58",
				goalie_player_id=30,
				goalie_x=91.0,
				goalie_y=5.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000004",
				game_clock="19:57",
				goalie_player_id=31,
				goalie_x=92.0,
				goalie_y=5.0,
			)
		)

		goalie_rows = select_frame_level_goalie_rows(tracking_frame)
		stability = infer_stable_goalie_identities(goalie_rows)
		goalie_trajectories = extract_goalie_trajectories(tracking_frame)

		self.assertEqual(int(stability.loc[0, "frame_level_candidate_rows"]), 4)
		self.assertEqual(int(stability.loc[0, "frame_level_unique_candidate_player_ids"]), 2)
		self.assertEqual(int(stability.loc[0, "stabilized_goalie_player_id"]), 30)
		self.assertEqual(int(stability.loc[0, "stabilized_goalie_frame_count"]), 3)
		self.assertAlmostEqual(float(stability.loc[0, "stability_player_share"]), 0.75)
		self.assertEqual(int(stability.loc[0, "stability_margin_frames"]), 2)
		self.assertTrue(bool(stability.loc[0, "stability_rule_passed"]))
		self.assertEqual(goalie_trajectories["stabilized_goalie_player_id"].tolist(), [30, 30, 30, 30])
		self.assertEqual(goalie_trajectories["is_stabilized_goalie_frame"].tolist(), [True, True, True, False])

	def test_infer_stable_goalie_identities_leaves_ambiguous_split_unresolved(self) -> None:
		tracking_frame = pd.DataFrame(
			_team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000001",
				game_clock="20:00",
				goalie_player_id=30,
				goalie_x=89.0,
				goalie_y=4.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000002",
				game_clock="19:59",
				goalie_player_id=30,
				goalie_x=90.0,
				goalie_y=4.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000003",
				game_clock="19:58",
				goalie_player_id=31,
				goalie_x=91.0,
				goalie_y=5.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000004",
				game_clock="19:57",
				goalie_player_id=31,
				goalie_x=92.0,
				goalie_y=5.0,
			)
		)

		goalie_rows = select_frame_level_goalie_rows(tracking_frame)
		stability = infer_stable_goalie_identities(goalie_rows)
		goalie_trajectories = extract_goalie_trajectories(tracking_frame)

		self.assertFalse(bool(stability.loc[0, "stability_rule_passed"]))
		self.assertTrue(pd.isna(stability.loc[0, "stabilized_goalie_player_id"]))
		self.assertEqual(goalie_trajectories["goalie_identity_inference_level"].tolist(), ["unresolved", "unresolved", "unresolved", "unresolved"])
		self.assertEqual(goalie_trajectories["is_stabilized_goalie_frame"].tolist(), [False, False, False, False])

	def test_summarize_goalie_trajectories_reports_counts_and_coordinate_source(self) -> None:
		tracking_frame = pd.DataFrame(
			_team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000001",
				game_clock="20:00",
				goalie_player_id=30,
				goalie_x=89.0,
				goalie_y=4.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000002",
				game_clock="19:59",
				goalie_player_id=30,
				goalie_x=90.0,
				goalie_y=4.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000003",
				game_clock="19:58",
				goalie_player_id=30,
				goalie_x=91.0,
				goalie_y=5.0,
			)
			+ _team_frame_rows(
				image_id="2025-10-24 Team E @ Team D_000004",
				game_clock="19:57",
				goalie_player_id=31,
				goalie_x=92.0,
				goalie_y=5.0,
			)
		)
		goalie_trajectories = extract_goalie_trajectories(tracking_frame)

		summary = summarize_goalie_trajectories(goalie_trajectories)

		self.assertEqual(summary["frame_level_goalie_rows"], 4)
		self.assertEqual(summary["unique_goalie_candidate_player_ids"], 2)
		self.assertEqual(summary["stabilized_goalie_identity_groups"], 1)
		self.assertEqual(summary["stabilized_goalie_frame_rows"], 3)
		self.assertEqual(summary["trajectory_coordinate_source"], "normalized_xy_feet")
		self.assertEqual(summary["team_distribution"], {"Home": 4})
