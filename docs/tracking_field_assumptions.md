# Tracking Field Assumptions

This note documents the tracking fields currently used by the pipeline, how they are interpreted in code today, and where uncertainty remains. It is intentionally limited to observed and implemented behavior.

## Scope

Current code paths using these fields are primarily in `src/schema_alignment.py`, `src/coordinate_normalization.py`, `src/goalie_trajectories.py`, and `src/metrics.py`.

## image_id

### Observed or Implemented Behavior
- Canonical tracking schema renames `Image Id` to `image_id`.
- `image_id` is treated as the frame identifier.
- Goalie candidate selection groups by `image_id` and `team` to choose at most one frame-level candidate row per team per frame.
- Frame ordering logic also uses the numeric suffix at the end of `image_id`.

### Assumptions
- `image_id` values follow the implemented pattern `"<game text>_<numeric frame suffix>"`.
- The suffix increases with frame progression within a period when clocks are otherwise equal.

### Unresolved Uncertainties
- The code assumes this naming pattern is valid for all tracking files in scope.
- No fallback parsing exists for alternate `image_id` formats.

## game extraction from image_id

### Observed or Implemented Behavior
- The pipeline derives `game` directly from `image_id` using the regex rule `^(?P<game>.+)_\d+$`.
- Everything before the final underscore-plus-digits is treated as the canonical game key.

### Assumptions
- The explicit game join rule is trustworthy enough for current tracking workflows.
- The derived `game` field is stable enough to join tracking rows to camera orientation metadata.

### Unresolved Uncertainties
- The project has not yet validated whether every historical or future tracking file follows this exact join convention.

## team

### Observed or Implemented Behavior
- Canonical tracking schema renames `Team` to `team`.
- `team` is used as a grouping key for frame-level goalie candidate selection, trajectory sorting, and motion summaries.
- Current real-sample outputs show `Home` and `Away` values.

### Assumptions
- `team` correctly distinguishes the two sides for frame-level grouping.

### Unresolved Uncertainties
- The code does not currently validate whether any other team labels appear in the raw data.

## player_id

### Observed or Implemented Behavior
- Canonical tracking schema renames `Player Id` to `player_id`.
- `player_id` is used as a sort tiebreaker in frame-level goalie candidate selection after `abs_x` and `abs_y`.
- Motion summaries currently group frame-level goalie candidate rows by `game`, `team`, and `player_id`.
- Conservative cross-frame identity inference also uses `player_id`, but only as an audit layer and only when one value clearly dominates candidate rows.

### Assumptions
- `player_id` is usable as a row-level identifier for grouping and tie-breaking.

### Unresolved Uncertainties
- On the inspected real sample, `player_id` appears unstable and should not yet be treated as a finalized goalie identity key.
- The real sample produced 233 unique frame-level goalie candidate `player_id` values across 21000 selected rows, which is inconsistent with a stable single-goalie identity interpretation.

## player_jersey_number

### Observed or Implemented Behavior
- Canonical tracking schema renames `Player Jersey Number` to `player_jersey_number`.
- The main pipeline does not currently use `player_jersey_number` to assign goalie identity.
- Diagnostic work shows far fewer unique jersey values than unique `player_id` values on the real sample.

### Assumptions
- `player_jersey_number` may be useful for diagnostics or audit views.

### Unresolved Uncertainties
- The field is not yet promoted into any goalie identity rule.
- The real sample showed substantial jersey-to-player_id churn, including a `Go` label mapping to many different `player_id` values, so this field is also not yet a trustworthy finalized identity key by itself.

## x_feet and y_feet

### Observed or Implemented Behavior
- Canonical tracking schema renames rink coordinates to `x_feet` and `y_feet`.
- Frame-level goalie candidate selection uses derived `abs_x_feet = abs(x_feet)` and `abs_y_feet = abs(y_feet)`.
- Current heuristic filters candidate rows to `abs_x_feet >= 80.0` and `abs_y_feet <= 15.0` before selecting the deepest eligible player per `image_id/team`.
- Trajectory extraction and motion summaries use either normalized coordinates or raw coordinates depending on what is available.

### Assumptions
- Larger `abs_x_feet` corresponds to being deeper toward an end of the rink.
- Smaller `abs_y_feet` corresponds to being more central laterally.
- These interpretations are strong enough for conservative frame-level candidate selection.

### Unresolved Uncertainties
- Coordinate normalization is still staged and does not yet apply finalized flips or transforms.
- Some selected candidate rows in the real sample sit near the heuristic cutoffs, so geometry may contribute noise even though it does not appear to be the main source of identity churn.

## abs_x_feet and abs_y_feet

### Observed or Implemented Behavior
- These are derived fields, not raw columns.
- They are used only inside the current goalie candidate heuristic.
- Selection order is: highest `abs_x_feet`, then lowest `abs_y_feet`, then lowest `player_id`.

### Assumptions
- Absolute rink position is sufficient for a conservative end-board and central-lane goalie proxy.

### Unresolved Uncertainties
- This heuristic is useful for frame-level candidate extraction, but not sufficient on its own to infer stable goalie identity across frames.

## Frame Ordering Inputs

### Observed or Implemented Behavior
- Frame ordering uses `period`, `game_clock`, and the numeric suffix from `image_id`.
- The current sequence logic converts `game_clock` to remaining seconds, then computes an increasing frame sequence within and across periods.

### Assumptions
- `game_clock` is formatted as `MM:SS`.
- `period` is numeric or numerically castable.
- The `image_id` numeric suffix is a valid sub-second or tie-break ordering input.

### Unresolved Uncertainties
- The current ordering logic does not validate whether every file uses the same clock semantics or suffix behavior.

## Current Practical Interpretation

- Supported today: frame-level heuristic goalie candidate selection, trajectory shaping, and movement summaries.
- Not supported today: a trustworthy finalized goalie identity key derived from `player_id` or `player_jersey_number`.
- Preferred current stance: keep cross-frame goalie identity unresolved unless the data supports a clearly defensible rule.