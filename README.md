# Goalie Analytics Progress

This repository supports a conservative, auditable goalie analytics pipeline built around frame-level heuristic goalie candidate selection from tracking data. It is useful for trajectory and movement summaries, and it now produces stabilized goalie identities on real tracking files using a two-level heuristic identity resolver.

## Pipeline Modules

- `src/schema_alignment.py`: canonical column alignment for tracking, event, shift, and camera-orientation files; explicit `image_id -> game` join rule; frame-level goalie candidate heuristic.
- `src/coordinate_normalization.py`: staged coordinate-normalization placeholder that attaches camera metadata and preserves coordinates until transformation rules are finalized.
- `src/event_alignment.py`: first-pass event-window alignment utilities using derived game keys, period, and second-level clock timing, plus compact event-type aggregate summaries.
- `src/goalie_trajectories.py`: frame-level goalie candidate extraction, trajectory shaping, sorting, and conservative two-level cross-frame identity resolver (player_id primary, jersey-number fallback).
- `src/metrics.py`: frame ordering, motion summaries, rebound-control placeholder logic, and other simple movement metrics.
- `src/tracking_prep.py`: tracking-file loading and preparation helpers for directory-level workflows.

## Scripts

- Validate one real tracking file through the current pipeline:
  - `python scripts/validate_tracking_pipeline.py --tracking-file "2025-10-24.Team.E.@.Team.D.Tracking_P1.csv"`
- Run richer goalie motion summaries from frame-level goalie trajectories (includes per-game/team stabilization diagnostics):
  - `python scripts/run_goalie_motion_summary.py "2025-10-24.Team.E.@.Team.D.Tracking_P1.csv"`
- Inspect granular goalie identity stability signals:
  - `python scripts/inspect_goalie_identity_stability.py "2025-10-24.Team.E.@.Team.D.Tracking_P1.csv" --top-n 8`
- Summarize motion strictly at the frame-level goalie candidate layer:
  - `python scripts/summarize_frame_level_goalie_motion.py "2025-10-24.Team.E.@.Team.D.Tracking_P1.csv" --preview-rows 5`
- Summarize frame-level goalie motion around second-level event windows:
  - `python scripts/summarize_goalie_motion_around_events.py "2025-10-28.Team.C.@.Team.A.Events.csv" "2025-10-28.Team.C.@.Team.A.Tracking_P1.csv" --pre-event-seconds 1 --post-event-seconds 1 --preview-rows 5`
- Compare descriptive goalie-movement aggregates across event types for one event and tracking pair:
  - `python scripts/compare_goalie_movement_by_event_type.py "2025-10-28.Team.C.@.Team.A.Events.csv" "2025-10-28.Team.C.@.Team.A.Tracking_P1.csv" --pre-event-seconds 1 --post-event-seconds 1`

## Goalie Identity Stabilization Heuristic

Cross-frame goalie identity is resolved by `infer_stable_goalie_identities()` using a conservative two-level rule:

1. **Primary path – player_id**: Count candidate frames per player_id within each game/team. A single player_id is stabilized when it clears all three thresholds: at least 3 candidate frames, at least 60% share of all candidate frames for that game/team, and at least 2 frames more than the runner-up.

2. **Fallback path – jersey number**: Many tracking systems assign a new ephemeral player_id each time a player re-enters the tracking window. When the primary path fails because player_id counts are too fragmented, the same three thresholds are applied to player_jersey_number instead. In practice, jersey number "Go" (used consistently for goalies in the observed data) carries 70–76% of goalie-zone candidate frames per period and easily clears the thresholds.

The jersey-number fallback fires **only when the player_id path failed**. A group is marked `stability_rule_passed = True` and `goalie_identity_inference_level = "stabilized_by_jersey_number"`.  Frames whose `player_jersey_number` matches the stabilized jersey are marked `is_stabilized_goalie_frame = True`.

`run_goalie_motion_summary.py` prints a per-game/team diagnostic block showing:
- candidate row counts
- player_id share, margin, and pass/fail
- jersey-number share, margin, and pass/fail (when jersey data is present)
- the final outcome and stabilized identifier

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
- Cross-frame goalie identity stabilization via two-level heuristic (player_id primary, jersey-number fallback). Observed real-data files now yield nonzero stabilized goalie identity groups and stabilized frame rows.
- Second-level event-window alignment and descriptive event-type aggregate summaries.

Caveats that remain:
- Stabilization is still heuristic and conservative: it resolves when one identifier clearly dominates and intentionally leaves ambiguous cases unresolved.
- `is_stabilized_goalie_frame = True` means the frame's jersey number (or player_id) matched the stabilized goalie identifier; it is not a guarantee that the tracked entity is the goalie.
- Frame-exact event linkage and causal event-effect claims are not supported.

Event-window outputs should be read as second-level joins on heuristic frame-level goalie candidates, not as exact event-state reconstructions.

## Tests

- Unit tests use small synthetic pandas DataFrames only.
- Current coverage includes schema alignment, coordinate normalization, event alignment, motion metrics, and goalie trajectory or stability behavior (including noisy candidate-switching scenarios and the jersey-number fallback path).
- Verified command:
  - `python -m unittest discover -s tests -v`
- Latest verified result:
  - `Ran 25 tests ... OK`

## CI

- GitHub Actions workflow: `.github/workflows/tests.yml`
- Triggers: `push`, `pull_request`
- Environment: `ubuntu-latest` with Python `3.11`
- Commands:
  - `python -m pip install --upgrade pip`
  - `pip install -r requirements.txt`
  - `python -m unittest discover -s tests -v`

## Recommended Next Steps

1. Audit whether jersey-number-based identity can be further validated against shift data (player on-ice periods vs. stabilized identity windows).
2. Expand real-file diagnostics across additional games to confirm the jersey fallback generalizes beyond the observed sample.
3. Add a small amount of script-level documentation or examples if the workflow will be handed off to someone else.