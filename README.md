# Goalie Analytics Progress

This repository currently supports a conservative, auditable goalie analytics pipeline built around frame-level heuristic goalie candidate selection from tracking data. It is useful for trajectory and movement summaries, but it does not yet provide a finalized goalie identity model.

## Pipeline Modules

- `src/schema_alignment.py`: canonical column alignment for tracking, event, shift, and camera-orientation files; explicit `image_id -> game` join rule; frame-level goalie candidate heuristic.
- `src/coordinate_normalization.py`: staged coordinate-normalization placeholder that attaches camera metadata and preserves coordinates until transformation rules are finalized.
- `src/event_alignment.py`: first-pass event-window alignment utilities using derived game keys, period, and second-level clock timing, plus compact event-type aggregate summaries.
- `src/goalie_trajectories.py`: frame-level goalie candidate extraction, trajectory shaping, sorting, and conservative cross-frame identity audit layer.
- `src/metrics.py`: frame ordering, motion summaries, rebound-control placeholder logic, and other simple movement metrics.
- `src/tracking_prep.py`: tracking-file loading and preparation helpers for directory-level workflows.

## Scripts

- Validate one real tracking file through the current pipeline:
  - `python scripts/validate_tracking_pipeline.py --tracking-file "2025-10-24.Team.E.@.Team.D.Tracking_P1.csv"`
- Run richer goalie motion summaries from frame-level goalie trajectories:
  - `python scripts/run_goalie_motion_summary.py "2025-10-24.Team.E.@.Team.D.Tracking_P1.csv"`
- Inspect why conservative stabilized goalie identity remains unresolved:
  - `python scripts/inspect_goalie_identity_stability.py "2025-10-24.Team.E.@.Team.D.Tracking_P1.csv" --top-n 8`
- Summarize motion strictly at the frame-level goalie candidate layer:
  - `python scripts/summarize_frame_level_goalie_motion.py "2025-10-24.Team.E.@.Team.D.Tracking_P1.csv" --preview-rows 5`
- Summarize frame-level goalie motion around second-level event windows:
  - `python scripts/summarize_goalie_motion_around_events.py "2025-10-28.Team.C.@.Team.A.Events.csv" "2025-10-28.Team.C.@.Team.A.Tracking_P1.csv" --pre-event-seconds 1 --post-event-seconds 1 --preview-rows 5`
- Compare descriptive goalie-movement aggregates across event types for one event and tracking pair:
  - `python scripts/compare_goalie_movement_by_event_type.py "2025-10-28.Team.C.@.Team.A.Events.csv" "2025-10-28.Team.C.@.Team.A.Tracking_P1.csv" --pre-event-seconds 1 --post-event-seconds 1`

## Event-Window Analysis

- The repository now supports a narrow event-window alignment layer between aligned event rows and frame-level goalie trajectories.
- The join uses derived `game`, shared `period`, and second-level clock timing with optional pre/post windows.
- Current event-window outputs are intended for audit and descriptive comparison only.
- These outputs can summarize goalie-candidate movement around event windows and compare aggregate movement differences across event types.

What these outputs do mean:
- Which frame-level goalie candidate rows fell inside the configured event windows.
- How movement summaries differ descriptively across the aligned event windows in one file pair.

What these outputs do not mean:
- Frame-exact event causality.
- Finalized goalie identity.
- Inferential or causal conclusions about event effects.

## Current Caveat

Supported today:
- Frame-level heuristic goalie candidate selection.
- Frame-level trajectory extraction and simple movement summaries.
- Second-level event-window alignment and descriptive event-type aggregate summaries.

Not supported today:
- A trustworthy finalized goalie identity model.
- Frame-exact event linkage.
- Causal claims from event-window summaries.

On the current real sample, conservative cross-frame stabilization remains unresolved. The evidence so far points more toward unstable candidate identifiers than toward a clean non-ML tie-break that can be justified confidently.
Event-window outputs should be read as second-level joins on heuristic frame-level goalie candidates, not as exact event-state reconstructions.

## Tests

- Unit tests use small synthetic pandas DataFrames only.
- Current coverage includes schema alignment, coordinate normalization, event alignment, motion metrics, and goalie trajectory or stability behavior.
- Verified command:
  - `python -m unittest discover -s tests -v`
- Latest verified result:
  - `Ran 23 tests ... OK`

## CI

- GitHub Actions workflow: `.github/workflows/tests.yml`
- Triggers: `push`, `pull_request`
- Environment: `ubuntu-latest` with Python `3.11`
- Commands:
  - `python -m pip install --upgrade pip`
  - `pip install -r requirements.txt`
  - `python -m unittest discover -s tests -v`

## Recommended Next Steps

1. Audit whether jersey-number fields can support a cautious identity-quality diagnostic without being promoted into a goalie identity rule prematurely.
2. Expand real-file diagnostics across additional games to see whether unresolved stabilization is consistent or sample-specific.
3. Add a small amount of script-level documentation or examples if the workflow will be handed off to someone else.