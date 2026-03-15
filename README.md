# Goalie Vulnerability System

## Abstract

This repository presents a research-style goalie vulnerability system built from hockey tracking data. The goal is to produce an interpretable representation of defensive exposure that begins with goalie state and then conditions that state on live threat context. The resulting system is designed to answer a more useful question than raw movement alone: not only where the goalie is exposed, but how that exposure changes as puck location and attacking skater geometry shape the developing play.

The final submission package includes a static executive compass, a threat-conditioned dynamic field, a rolling animated compass, a synchronized combined presentation, a methodology overview, a validation panel, and a proof-of-value panel. Together these artifacts show model structure, visual behavior, and limited but defensible event-linked evidence.

## Problem Motivation

Goalie movement by itself is an incomplete description of danger. A deep or laterally displaced goalie is not equally vulnerable in every context, and the same body position can carry very different meaning depending on puck location, slot pressure, and attacking support. This project therefore separates the problem into two layers:

- a base goalie-state vulnerability model
- a threat-conditioning layer that shifts vulnerability toward the live attacking lane

This framing preserves interpretability while moving the system closer to real hockey context.

## Method Summary

The system is organized as a staged pipeline.

1. Tracking preparation and goalie stabilization
   Tracking files are aligned into a canonical schema, stabilized goalie rows are extracted, and frame order is preserved for downstream modeling.

2. Base goalie-state vulnerability modeling
   Frame-level features summarize lateral exposure, depth exposure, recovery burden, instability, and crease control. These components drive both a local field representation and the compass axes.

3. Threat-conditioning
   Puck location is used when available, nearest-attacker geometry acts as a practical shooter proxy, and broader attacking skater support contributes slot pressure, side bias, and lane pressure. The field is then re-weighted toward the active threat source.

4. Rolling summary logic
   A centered rolling window produces a calmer summary of the same sequence shown by the dynamic field. This becomes the animated companion compass and the compact inset used in the combined presentation.

## Submission Artifacts

The main packaging script is [scripts/build_vulnerability_compass.py](scripts/build_vulnerability_compass.py). It exports the final submission set.

Core artifacts:

- [vulnerability_compass_signature_final.png](vulnerability_compass_signature_final.png)
- [vulnerability_field_animation_threat_conditioned.gif](vulnerability_field_animation_threat_conditioned.gif)
- [vulnerability_compass_animation_threat_conditioned.gif](vulnerability_compass_animation_threat_conditioned.gif)
- [vulnerability_field_compass_combo_threat_conditioned.gif](vulnerability_field_compass_combo_threat_conditioned.gif)

Supporting artifacts:

- [vulnerability_method_overview.png](vulnerability_method_overview.png)
- [vulnerability_validation_panel.png](vulnerability_validation_panel.png)
- [vulnerability_field_compass_combo_threat_conditioned_storyboard.png](vulnerability_field_compass_combo_threat_conditioned_storyboard.png)
- [vulnerability_value_panel.png](vulnerability_value_panel.png)

## Main Findings

The most important model finding is that threat-conditioning materially changes the output rather than acting as a cosmetic overlay. In the showcase sequence, threat context is available on 99.21% of goalie frames, the mean composite shift from threat-conditioning is 0.1392, and 94.49% of frames move by at least 0.05.

The proof-of-value panel adds a second layer of evidence. For the full 2025-10-28 Team C at Team A game, the Home goalie's threat-conditioned composite vulnerability averages about 0.317 across all conditioned frames and about 0.364 in 2-second windows leading into opposing on-net shots and goals. That directional lift is paired with a case-study sequence in Period 1 leading into a Team C shot on net at 11:50.

## Evidence Standard

This repository supports two levels of evidence.

- Frame-level visual interpretation through the field, compass, combined presentation, and case-study assets.
- Narrow second-level event-window linkage through the value panel.

The second level should be interpreted conservatively. Event alignment here is a second-level timing join, not a frame-exact possession or causality model. It is strong enough to support a directional proof-of-value claim, but not strong enough to claim predictive or causal validation.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the final package:

```bash
python scripts/build_vulnerability_compass.py
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Limitations

The project is submission-ready but not yet production-grade.

- Goalie identity stabilization is still heuristic and conservative.
- Shooter identity is proxied from puck-plus-attacker geometry rather than directly resolved from labeled possession data.
- Event linkage is limited to second-level windows rather than frame-synchronized threat-state reconstruction.
- Current compass timing is frame-sequence based rather than fully calibrated to exact elapsed time.
- MP4 export remains unavailable in this environment because the required video backend is not installed.

## Future Work

1. Add frame-synchronized event context so shot, pass, rush, and rebound states can directly condition the field.
2. Replace the shooter proxy with stronger possession and attacker-resolution logic.
3. Calibrate motion features to exact elapsed time and camera confidence rather than frame-sequence proxies.
4. Expand validation across more games and more teams.
5. Add production video export and a lightweight presentation runner.

## Repository Contents

Supporting modules in [src](src) handle schema alignment, tracking preparation, coordinate normalization, event alignment, goalie trajectory extraction, and descriptive motion utilities. Additional scripts in [scripts](scripts) support validation, motion summaries, and final artifact generation.

## Repository Notes

Supporting modules in [src](src) handle schema alignment, tracking preparation, coordinate normalization, goalie trajectory extraction, and descriptive motion utilities. Additional scripts in [scripts](scripts) support validation, inspection, and motion summaries used during development of the final vulnerability package.