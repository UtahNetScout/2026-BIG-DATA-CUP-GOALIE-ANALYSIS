import unittest

import pandas as pd

from src.event_alignment import align_goalie_trajectories_to_event_windows, derive_event_game_key, summarize_aligned_goalie_motion_by_event_type


class DeriveEventGameKeyTests(unittest.TestCase):
	def test_derive_event_game_key_matches_tracking_style_game_name(self) -> None:
		events = pd.DataFrame(
			[
				{"date": "2025-10-28", "home_team": "Team A", "away_team": "Team C"},
			]
		)

		game_key = derive_event_game_key(events)

		self.assertEqual(game_key.tolist(), ["2025-10-28 Team C @ Team A"])


class AlignGoalieTrajectoriesToEventWindowsTests(unittest.TestCase):
	def test_aligns_frame_rows_to_same_second_event_window(self) -> None:
		goalie_rows = pd.DataFrame(
			[
				{"game": "2025-10-28 Team C @ Team A", "period": 1, "game_clock": "19:58", "image_id": "2025-10-28 Team C @ Team A_000158", "team": "Home", "player_id": 30},
				{"game": "2025-10-28 Team C @ Team A", "period": 1, "game_clock": "19:57", "image_id": "2025-10-28 Team C @ Team A_000159", "team": "Home", "player_id": 30},
			]
		)
		events = pd.DataFrame(
			[
				{"date": "2025-10-28", "home_team": "Team A", "away_team": "Team C", "period": 1, "clock": "19:58", "team": "Team A", "event_type": "Play"},
			]
		)

		aligned = align_goalie_trajectories_to_event_windows(goalie_rows, events)

		self.assertEqual(len(aligned), 1)
		self.assertEqual(aligned.loc[0, "image_id"], "2025-10-28 Team C @ Team A_000158")
		self.assertEqual(aligned.loc[0, "event_type"], "Play")

	def test_supports_pre_and_post_event_second_windows(self) -> None:
		goalie_rows = pd.DataFrame(
			[
				{"game": "2025-10-28 Team C @ Team A", "period": 1, "game_clock": "19:59", "image_id": "2025-10-28 Team C @ Team A_000157", "team": "Home", "player_id": 30},
				{"game": "2025-10-28 Team C @ Team A", "period": 1, "game_clock": "19:58", "image_id": "2025-10-28 Team C @ Team A_000158", "team": "Home", "player_id": 30},
				{"game": "2025-10-28 Team C @ Team A", "period": 1, "game_clock": "19:57", "image_id": "2025-10-28 Team C @ Team A_000159", "team": "Home", "player_id": 30},
			]
		)
		events = pd.DataFrame(
			[
				{"date": "2025-10-28", "home_team": "Team A", "away_team": "Team C", "period": 1, "clock": "19:58", "team": "Team A", "event_type": "Play"},
			]
		)

		aligned = align_goalie_trajectories_to_event_windows(
			goalie_rows,
			events,
			pre_event_seconds=1,
			post_event_seconds=1,
		)

		self.assertEqual(aligned["image_id"].tolist(), [
			"2025-10-28 Team C @ Team A_000157",
			"2025-10-28 Team C @ Team A_000158",
			"2025-10-28 Team C @ Team A_000159",
		])

	def test_requires_current_event_alignment_inputs(self) -> None:
		goalie_rows = pd.DataFrame(
			[
				{"game": "2025-10-28 Team C @ Team A", "period": 1, "image_id": "2025-10-28 Team C @ Team A_000158", "team": "Home"},
			]
		)
		events = pd.DataFrame(
			[
				{"date": "2025-10-28", "home_team": "Team A", "away_team": "Team C", "period": 1, "clock": "19:58", "team": "Team A", "event_type": "Play"},
			]
		)

		with self.assertRaisesRegex(ValueError, "goalie_trajectories is missing columns required for event alignment: game_clock"):
			align_goalie_trajectories_to_event_windows(goalie_rows, events)


class SummarizeAlignedGoalieMotionByEventTypeTests(unittest.TestCase):
	def test_summarizes_aligned_rows_by_event_type(self) -> None:
		aligned = pd.DataFrame(
			[
				{
					"game": "2025-10-28 Team C @ Team A",
					"period": 1,
					"game_clock": "19:58",
					"image_id": "2025-10-28 Team C @ Team A_000158",
					"event_index": 1,
					"event_type": "Play",
					"team_goalie": "Home",
					"player_id_goalie": 30,
					"trajectory_x_feet": 0.0,
					"trajectory_y_feet": 0.0,
				},
				{
					"game": "2025-10-28 Team C @ Team A",
					"period": 1,
					"game_clock": "19:57",
					"image_id": "2025-10-28 Team C @ Team A_000159",
					"event_index": 1,
					"event_type": "Play",
					"team_goalie": "Home",
					"player_id_goalie": 30,
					"trajectory_x_feet": 3.0,
					"trajectory_y_feet": 4.0,
				},
				{
					"game": "2025-10-28 Team C @ Team A",
					"period": 1,
					"game_clock": "19:58",
					"image_id": "2025-10-28 Team C @ Team A_000160",
					"event_index": 2,
					"event_type": "Shot",
					"team_goalie": "Away",
					"player_id_goalie": 40,
					"trajectory_x_feet": 1.0,
					"trajectory_y_feet": 1.0,
				},
				{
					"game": "2025-10-28 Team C @ Team A",
					"period": 1,
					"game_clock": "19:57",
					"image_id": "2025-10-28 Team C @ Team A_000161",
					"event_index": 2,
					"event_type": "Shot",
					"team_goalie": "Away",
					"player_id_goalie": 40,
					"trajectory_x_feet": 1.0,
					"trajectory_y_feet": 4.0,
				},
			]
		)

		summary = summarize_aligned_goalie_motion_by_event_type(aligned)

		self.assertEqual(summary["event_type"].tolist(), ["Play", "Shot"])
		self.assertEqual(summary["aligned_row_count"].tolist(), [2, 2])
		self.assertEqual(summary["distinct_event_count"].tolist(), [1, 1])
		self.assertEqual(summary["motion_summary_group_count"].tolist(), [1, 1])
		self.assertEqual(summary["median_frame_count"].tolist(), [2.0, 2.0])
		self.assertAlmostEqual(float(summary.loc[0, "median_path_length_feet"]), 5.0)
		self.assertAlmostEqual(float(summary.loc[1, "median_path_length_feet"]), 3.0)

	def test_optionally_keeps_team_breakout(self) -> None:
		aligned = pd.DataFrame(
			[
				{
					"game": "2025-10-28 Team C @ Team A",
					"period": 1,
					"game_clock": "19:58",
					"image_id": "2025-10-28 Team C @ Team A_000158",
					"event_index": 1,
					"event_type": "Play",
					"team_goalie": "Home",
					"player_id_goalie": 30,
					"trajectory_x_feet": 0.0,
					"trajectory_y_feet": 0.0,
				},
				{
					"game": "2025-10-28 Team C @ Team A",
					"period": 1,
					"game_clock": "19:58",
					"image_id": "2025-10-28 Team C @ Team A_000160",
					"event_index": 2,
					"event_type": "Play",
					"team_goalie": "Away",
					"player_id_goalie": 40,
					"trajectory_x_feet": 1.0,
					"trajectory_y_feet": 1.0,
				},
			]
		)

		summary = summarize_aligned_goalie_motion_by_event_type(aligned, include_team=True)

		self.assertEqual(summary[["event_type", "team_goalie"]].values.tolist(), [["Play", "Home"], ["Play", "Away"]])
		self.assertEqual(summary["aligned_row_count"].tolist(), [1, 1])