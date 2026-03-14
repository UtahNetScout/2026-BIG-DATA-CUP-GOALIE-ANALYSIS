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
from src.schema_alignment import align_event_schema, align_tracking_schema


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compare frame-level goalie movement aggregates across event types for one event file and one tracking file."
	)
	parser.add_argument("event_file", help="Path to an event CSV file.")
	parser.add_argument("tracking_file", help="Path to a tracking CSV file.")
	parser.add_argument("--pre-event-seconds", type=int, default=0, help="Seconds to include before each event clock time.")
	parser.add_argument("--post-event-seconds", type=int, default=0, help="Seconds to include after each event clock time.")
	return parser.parse_args()


def _resolve_path(raw_path: str, label: str) -> Path:
	path = Path(raw_path).expanduser().resolve()
	if not path.exists():
		raise FileNotFoundError(f"{label} not found: {path}")
	return path


def _build_interpretation(event_type_summary: pd.DataFrame) -> list[str]:
	if event_type_summary.empty:
		return ["No aligned event-type rows were found for the selected files and window settings."]

	working = event_type_summary.dropna(subset=["event_type"]).copy()
	if working.empty:
		return ["Aligned rows were found, but no usable event_type values were available for comparison."]

	lines: list[str] = []
	most_rows = working.sort_values(["aligned_row_count", "event_type"], ascending=[False, True], kind="stable").iloc[0]
	most_groups = working.sort_values(["motion_summary_group_count", "event_type"], ascending=[False, True], kind="stable").iloc[0]
	highest_path = working.sort_values(["median_path_length_feet", "event_type"], ascending=[False, True], kind="stable").iloc[0]
	highest_efficiency = working.sort_values(["median_motion_efficiency", "event_type"], ascending=[False, True], kind="stable").iloc[0]
	lowest_efficiency = working.sort_values(["median_motion_efficiency", "event_type"], ascending=[True, True], kind="stable").iloc[0]

	lines.append(
		f"In this file and event-window setting, {most_rows['event_type']} has the largest aligned sample "
		f"with {int(most_rows['aligned_row_count'])} aligned goalie rows across {int(most_rows['distinct_event_count'])} events."
	)
	lines.append(
		f"{most_groups['event_type']} has the most aligned motion-summary groups at {int(most_groups['motion_summary_group_count'])}, "
		"so its medians are being computed over the broadest set of grouped movement slices here."
	)
	lines.append(
		f"By median path length, {highest_path['event_type']} is highest at {float(highest_path['median_path_length_feet']):.3f} feet "
		f"per aligned motion group in this descriptive comparison."
	)
	lines.append(
		f"By median motion efficiency, {highest_efficiency['event_type']} is highest at {float(highest_efficiency['median_motion_efficiency']):.3f}, "
		f"while {lowest_efficiency['event_type']} is lowest at {float(lowest_efficiency['median_motion_efficiency']):.3f}."
	)
	lines.append(
		"These are descriptive differences in second-level aligned heuristic goalie-candidate windows only; "
		"they do not establish causality or a finalized goalie identity."
	)
	return lines


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
	event_type_summary = summarize_aligned_goalie_motion_by_event_type(aligned)
	interpretation_lines = _build_interpretation(event_type_summary)

	print(f"event_file: {event_path}")
	print(f"tracking_file: {tracking_path}")
	print("event_type_aggregate_summary:")
	if event_type_summary.empty:
		print("<no event-type summary rows>")
	else:
		print(event_type_summary.to_string(index=False))
	print("interpretation:")
	for line in interpretation_lines:
		print(f"- {line}")
	print(f"goalie_caveat: {GOALIE_TRAJECTORY_NOTE}")
	print(f"alignment_caveat: {EVENT_ALIGNMENT_NOTE}")


if __name__ == "__main__":
	main()