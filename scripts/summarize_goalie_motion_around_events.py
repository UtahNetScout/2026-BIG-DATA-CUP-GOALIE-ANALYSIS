from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
	sys.path.insert(0, str(WORKSPACE_ROOT))

from src.event_alignment import EVENT_ALIGNMENT_NOTE, align_goalie_trajectories_to_event_windows, summarize_aligned_goalie_motion_by_event_type
from src.goalie_trajectories import GOALIE_TRAJECTORY_NOTE, extract_goalie_trajectories
from src.metrics import compute_motion_efficiency
from src.schema_alignment import align_event_schema, align_tracking_schema


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Summarize frame-level goalie motion around aligned event windows for one event file and one tracking file."
	)
	parser.add_argument("event_file", help="Path to an event CSV file.")
	parser.add_argument("tracking_file", help="Path to a tracking CSV file.")
	parser.add_argument("--pre-event-seconds", type=int, default=0, help="Seconds to include before each event clock time.")
	parser.add_argument("--post-event-seconds", type=int, default=0, help="Seconds to include after each event clock time.")
	parser.add_argument("--preview-rows", type=int, default=10, help="Number of aligned summary rows to preview.")
	return parser.parse_args()


def _resolve_path(raw_path: str, label: str) -> Path:
	path = Path(raw_path).expanduser().resolve()
	if not path.exists():
		raise FileNotFoundError(f"{label} not found: {path}")
	return path


def main() -> None:
	args = parse_args()
	event_path = _resolve_path(args.event_file, "Event file")
	tracking_path = _resolve_path(args.tracking_file, "Tracking file")

	raw_events = pd.read_csv(event_path)
	raw_tracking = pd.read_csv(tracking_path)
	canonical_events = align_event_schema(raw_events)
	canonical_tracking = align_tracking_schema(raw_tracking)
	goalie_trajectories = extract_goalie_trajectories(canonical_tracking)
	aligned = align_goalie_trajectories_to_event_windows(
		goalie_trajectories,
		canonical_events,
		pre_event_seconds=args.pre_event_seconds,
		post_event_seconds=args.post_event_seconds,
	)

	motion_summary = compute_motion_efficiency(
		aligned,
		group_columns=("game", "event_index", "event_type", "team_goalie", "player_id_goalie"),
		x_column="trajectory_x_feet",
		y_column="trajectory_y_feet",
	)

	event_type_summary = summarize_aligned_goalie_motion_by_event_type(aligned)

	preview_columns = [
		column
		for column in [
			"game",
			"event_index",
			"event_type",
			"team_goalie",
			"player_id_goalie",
			"frame_count",
			"path_length_feet",
			"straight_line_distance_feet",
			"motion_efficiency",
			"start_x_feet",
			"start_y_feet",
			"end_x_feet",
			"end_y_feet",
		]
		if column in motion_summary.columns
	]

	matched_events = int(aligned["event_index"].nunique()) if "event_index" in aligned.columns and not aligned.empty else 0

	print(f"event_file: {event_path}")
	print(f"tracking_file: {tracking_path}")
	print(f"total_aligned_goalie_event_rows: {len(aligned)}")
	print(f"matched_events: {matched_events}")
	print(f"aligned_motion_summary_groups: {len(motion_summary)}")
	print("event_type_aggregate_summary:")
	if event_type_summary.empty:
		print("<no event-type summary rows>")
	else:
		print(event_type_summary.to_string(index=False))
	print("aligned_motion_summary_preview:")
	if motion_summary.empty:
		print("<no aligned motion summary rows>")
	else:
		print(motion_summary.loc[:, preview_columns].head(args.preview_rows).to_string(index=False))
	print(f"goalie_caveat: {GOALIE_TRAJECTORY_NOTE}")
	print(f"alignment_caveat: {EVENT_ALIGNMENT_NOTE}")
	print(
		"analysis_note: aligned motion summaries are second-level event-window joins on heuristic frame-level goalie candidates. "
		"They are useful for narrow audit slices, not finalized goalie identity or frame-exact event causality."
	)


if __name__ == "__main__":
	main()