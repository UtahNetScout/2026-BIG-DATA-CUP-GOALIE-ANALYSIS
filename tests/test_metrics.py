import unittest

import pandas as pd

from src.metrics import build_tracking_frame_sequence, categorize_rebound_control, compute_motion_efficiency


class BuildTrackingFrameSequenceTests(unittest.TestCase):
	def test_build_tracking_frame_sequence_orders_by_clock_then_suffix(self) -> None:
		frame = pd.DataFrame(
			[
				{"image_id": "game_000002", "period": 1, "game_clock": "19:59"},
				{"image_id": "game_000003", "period": 1, "game_clock": "19:59"},
				{"image_id": "game_000001", "period": 1, "game_clock": "19:58"},
			]
		)

		sequence = build_tracking_frame_sequence(frame)

		self.assertLess(sequence.iloc[0], sequence.iloc[1])
		self.assertLess(sequence.iloc[1], sequence.iloc[2])

	def test_build_tracking_frame_sequence_ot_period_follows_regulation(self) -> None:
		"""'OT' period label should produce frame sequence values greater than period 3."""
		frame = pd.DataFrame(
			[
				{"image_id": "game_000001", "period": "OT", "game_clock": "4:59"},
				{"image_id": "game_000002", "period": 3, "game_clock": "0:01"},
			]
		)

		sequence = build_tracking_frame_sequence(frame)

		# OT (mapped to period 4) must sort after regulation period 3
		ot_seq = sequence.iloc[0]
		p3_seq = sequence.iloc[1]
		self.assertGreater(ot_seq, p3_seq)

	def test_build_tracking_frame_sequence_ot_is_monotonic_within_period(self) -> None:
		"""Multiple OT frames should be ordered by clock and image suffix, not crash."""
		frame = pd.DataFrame(
			[
				{"image_id": "game_000001", "period": "OT", "game_clock": "5:00"},
				{"image_id": "game_000002", "period": "OT", "game_clock": "4:30"},
				{"image_id": "game_000003", "period": "OT", "game_clock": "4:30"},
			]
		)

		sequence = build_tracking_frame_sequence(frame)

		# First frame has more time remaining → lower elapsed → lower sequence value
		self.assertLess(sequence.iloc[0], sequence.iloc[1])
		# Same clock, higher suffix → higher sequence value
		self.assertLess(sequence.iloc[1], sequence.iloc[2])


class ComputeMotionEfficiencyTests(unittest.TestCase):
	def test_compute_motion_efficiency_returns_one_for_straight_line_path(self) -> None:
		tracking_frame = pd.DataFrame(
			[
				{"game": "g1", "team": "Home", "player_id": 30, "image_id": "g1_000001", "period": 1, "game_clock": "19:59", "normalized_x_feet": 0.0, "normalized_y_feet": 0.0},
				{"game": "g1", "team": "Home", "player_id": 30, "image_id": "g1_000002", "period": 1, "game_clock": "19:58", "normalized_x_feet": 3.0, "normalized_y_feet": 4.0},
				{"game": "g1", "team": "Home", "player_id": 30, "image_id": "g1_000003", "period": 1, "game_clock": "19:57", "normalized_x_feet": 6.0, "normalized_y_feet": 8.0},
			]
		)

		summary = compute_motion_efficiency(tracking_frame)

		self.assertEqual(int(summary.loc[0, "sample_count"]), 3)
		self.assertEqual(int(summary.loc[0, "frame_count"]), 3)
		self.assertAlmostEqual(float(summary.loc[0, "path_length_feet"]), 10.0)
		self.assertAlmostEqual(float(summary.loc[0, "net_displacement_feet"]), 10.0)
		self.assertAlmostEqual(float(summary.loc[0, "straight_line_distance_feet"]), 10.0)
		self.assertAlmostEqual(float(summary.loc[0, "start_x_feet"]), 0.0)
		self.assertAlmostEqual(float(summary.loc[0, "start_y_feet"]), 0.0)
		self.assertAlmostEqual(float(summary.loc[0, "end_x_feet"]), 6.0)
		self.assertAlmostEqual(float(summary.loc[0, "end_y_feet"]), 8.0)
		self.assertAlmostEqual(float(summary.loc[0, "motion_efficiency"]), 1.0)

	def test_compute_motion_efficiency_drops_below_one_for_non_linear_path(self) -> None:
		tracking_frame = pd.DataFrame(
			[
				{"game": "g1", "team": "Home", "player_id": 30, "image_id": "g1_000001", "period": 1, "game_clock": "19:59", "normalized_x_feet": 0.0, "normalized_y_feet": 0.0},
				{"game": "g1", "team": "Home", "player_id": 30, "image_id": "g1_000002", "period": 1, "game_clock": "19:58", "normalized_x_feet": 3.0, "normalized_y_feet": 4.0},
				{"game": "g1", "team": "Home", "player_id": 30, "image_id": "g1_000003", "period": 1, "game_clock": "19:57", "normalized_x_feet": 3.0, "normalized_y_feet": 0.0},
			]
		)

		summary = compute_motion_efficiency(tracking_frame)

		self.assertEqual(int(summary.loc[0, "frame_count"]), 3)
		self.assertAlmostEqual(float(summary.loc[0, "path_length_feet"]), 9.0)
		self.assertAlmostEqual(float(summary.loc[0, "net_displacement_feet"]), 3.0)
		self.assertAlmostEqual(float(summary.loc[0, "straight_line_distance_feet"]), 3.0)
		self.assertAlmostEqual(float(summary.loc[0, "start_x_feet"]), 0.0)
		self.assertAlmostEqual(float(summary.loc[0, "start_y_feet"]), 0.0)
		self.assertAlmostEqual(float(summary.loc[0, "end_x_feet"]), 3.0)
		self.assertAlmostEqual(float(summary.loc[0, "end_y_feet"]), 0.0)
		self.assertAlmostEqual(float(summary.loc[0, "motion_efficiency"]), 3.0 / 9.0)


class CategorizeReboundControlTests(unittest.TestCase):
	def test_goalie_control_takes_precedence(self) -> None:
		category = categorize_rebound_control(
			controlled_by_goalie=True,
			controlled_by_defending_team=True,
			immediate_follow_up_shot=True,
		)

		self.assertEqual(category, "freeze")
