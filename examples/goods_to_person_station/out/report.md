# Goods-to-person pick station — workstation diagnostic

Distribution-centre pick station: a mobile shelf pod docks into a black portal; the operator picks an item from a pod cell and drops it into a tote on the left. Rebuilt from three phone photos.

**Verdict (reach only): Reach works on the nominal geometry but depends on dimensions the photos do not pin down: measure before deciding.**

> Grasp reliability is not assessed by this diagnostic. It checks that the robot can reach, carry and drop — not that it can grip a soft or cluttered item.

## Robots

| Robot | Targets reached (drop included) | Best base (x, y, pedestal) | Base positions where the drop is reachable |
|---|---|---|---|
| UR5e | 7 / 32 | -0.90, -0.60, 1.30 m | 2 / 60 |
| UR10e | 30 / 32 | -0.60, -0.60, 1.00 m | 20 / 60 |
| CRX-5iA | 15 / 32 | -0.60, -0.40, 1.30 m | 9 / 60 |

## Cycle

- Full collision-free cycle for **25 of 32** targets with UR10e.
- Cycle time at joint speed limits: mean 6.2 s (min 5.6, max 7.0). This is a ceiling, not a production rate.

## How sure are we

The scene was re-simulated on 40 plausible variants drawn from the uncertainty on each dimension.

| Layout | Drop unreachable | ≥ 80 % of targets | ≥ 90 % of targets |
|---|---|---|---|
| Fixed pedestal | 13 / 40 | 27 | 23 |
| Vertical axis | 0 / 40 | 40 | 31 |

**Measure on site first** (dimensions that move the result most):

- `tote_rim` — tote rim height (currently 1.12 ± 0.07 m, correlation -0.63)
- `bin_z0` — bottom of the lowest cell (currently 0.2 ± 0.05 m, correlation +0.37)
- `tote_dx` — portal column to tote centre, to the left (currently 0.55 ± 0.1 m, correlation -0.30)
- `pod_gap` — portal plane to pod face (currently 0.1 ± 0.05 m, correlation -0.29)

## Models and assets used

| Robot | Model | Licence | Reach, payload, joint speeds |
|---|---|---|---|
| UR5e | MuJoCo Menagerie `universal_robots_ur5e` at `98a70c5` | BSD-3-Clause | datasheet values kept in `ari_sim/robots.py` |
| UR10e | MuJoCo Menagerie `universal_robots_ur10e` at `98a70c5` | BSD-3-Clause | datasheet values kept in `ari_sim/robots.py` |
| CRX-5iA | MuJoCo Menagerie `fanuc_crx5ia` at `98a70c5` | Apache-2.0 | datasheet values kept in `ari_sim/robots.py` |

## Scale references used

- Black and red totes: 600 x 400 mm Euro footprint
- A4 instruction sheets on the black panel (210 x 297 mm)
- 22-inch touch screen
- Two-step stool risers
- Standard 1 m pod footprint

## Assumptions

- No pod was docked in the photos: pod geometry (4 columns, 250 mm cell pitch, 2.3 m high) is assumed from the pods visible in the background.
- Screen and sensor mast positions are approximate.
- The step stool is removed for the robot layout.
