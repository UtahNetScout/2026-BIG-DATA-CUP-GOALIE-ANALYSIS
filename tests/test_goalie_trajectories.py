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


def _team_frame_rows_with_jersey(
	*,
	image_id: str,
	game_clock: str,
	goalie_player_id: int,
	goalie_x: float,
	goalie_y: float,
	goalie_jersey: str,
) -> list[dict[str, object]]:
	"""Like _team_frame_rows but includes player_jersey_number on each row."""
	rows = [
		{
			"game": "2025-10-24 Team E @ Team D",
			"image_id": image_id,
			"team": "Home",
			"period": 1,
			"game_clock": game_clock,
			"entity_type": "player",
			"player_id": goalie_player_id,
			"player_jersey_number": goalie_jersey,
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
				"player_jersey_number": str(player_id),
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

	def test_infer_stable_goalie_identities_stabilizes_by_jersey_when_player_ids_are_fragmented(self) -> None:
		"""Jersey-number fallback resolves identity when many player_ids map to one dominant jersey."""
		# Simulate a common real-data scenario: the tracking system assigns a fresh player_id
		# each tracking segment, so the same physical goalie appears under 8 different IDs.
		# Each player_id appears only once (12.5% share < 60%), so the player_id rule fails.
		# All rows carry jersey "Go", which covers 100% of frames, so the jersey rule passes.
		all_rows: list[dict[str, object]] = []
		for i in range(8):
			all_rows.extend(
				_team_frame_rows_with_jersey(
					image_id=f"2025-10-24 Team E @ Team D_{(i + 1):06d}",
					game_clock=f"19:{52 - i:02d}",
					goalie_player_id=200 + i,
					goalie_x=90.0,
					goalie_y=4.0,
					goalie_jersey="Go",
				)
			)
		tracking_frame = pd.DataFrame(all_rows)

		goalie_rows = select_frame_level_goalie_rows(tracking_frame)
		stability = infer_stable_goalie_identities(goalie_rows)
		goalie_trajectories = extract_goalie_trajectories(tracking_frame)

		# Player-id rule must fail (each player_id appears once → ~12.5% share < 60%).
		self.assertFalse(bool(stability.loc[0, "stability_rule_passed_by_player_id"]))
		# Jersey rule must fire as the fallback.
		self.assertTrue(bool(stability.loc[0, "stability_by_jersey_number_rule_passed"]))
		self.assertTrue(bool(stability.loc[0, "stability_rule_passed"]))
		self.assertEqual(str(stability.loc[0, "stabilized_goalie_jersey_number"]), "Go")
		self.assertTrue(pd.isna(stability.loc[0, "stabilized_goalie_player_id"]))
		self.assertEqual(stability.loc[0, "goalie_identity_inference_level"], "stabilized_by_jersey_number")
		self.assertEqual(int(stability.loc[0, "stabilized_goalie_frame_count"]), 8)
		# Every goalie-candidate frame carries jersey "Go", so all 8 must be stabilized.
		self.assertEqual(len(goalie_trajectories), 8)
		self.assertTrue(all(goalie_trajectories["is_stabilized_goalie_frame"]))

	def test_infer_stable_goalie_identities_jersey_fallback_requires_dominant_jersey(self) -> None:
		"""Jersey-based fallback stays unresolved when no single jersey clearly dominates."""
		# 4 frames with jersey "Go" and 4 with jersey "31" → 50/50 split; neither dominates.
		all_rows: list[dict[str, object]] = []
		for i in range(4):
			all_rows.extend(
				_team_frame_rows_with_jersey(
					image_id=f"2025-10-24 Team E @ Team D_{(i + 1):06d}",
					game_clock=f"19:{52 - i:02d}",
					goalie_player_id=200 + i,
					goalie_x=90.0,
					goalie_y=4.0,
					goalie_jersey="Go",
				)
			)
		for i in range(4):
			all_rows.extend(
				_team_frame_rows_with_jersey(
					image_id=f"2025-10-24 Team E @ Team D_{(i + 5):06d}",
					game_clock=f"19:{48 - i:02d}",
					goalie_player_id=300 + i,
					goalie_x=90.0,
					goalie_y=4.0,
					goalie_jersey="31",
				)
			)
		tracking_frame = pd.DataFrame(all_rows)

		goalie_rows = select_frame_level_goalie_rows(tracking_frame)
		stability = infer_stable_goalie_identities(goalie_rows)
		goalie_trajectories = extract_goalie_trajectories(tracking_frame)

		# Both player_id and jersey rules should fail (50/50 split, 50% share < 60%).
		self.assertFalse(bool(stability.loc[0, "stability_rule_passed_by_player_id"]))
		self.assertFalse(bool(stability.loc[0, "stability_by_jersey_number_rule_passed"]))
		self.assertFalse(bool(stability.loc[0, "stability_rule_passed"]))
		self.assertEqual(stability.loc[0, "goalie_identity_inference_level"], "unresolved")
		self.assertTrue(all(~goalie_trajectories["is_stabilized_goalie_frame"]))
