# Goods-to-person pick station — workstation diagnostic

Distribution-centre pick station: a mobile shelf pod docks into a black portal; the operator picks an item from a pod cell and drops it into a tote on the left. Rebuilt from three phone photos.

**Verdict (reach only): Reach works on the nominal geometry but depends on dimensions the photos do not pin down: measure before deciding.**

> Grasp reliability is not assessed by this diagnostic. It checks that the robot can reach, carry and drop — not that it can grip a soft or cluttered item.

## Robots

| Robot | Targets reached (drop included) | Best base (x, y, pedestal) | Base positions where the drop is reachable |
|---|---|---|---|
| UR5e | 6 / 32 | -0.75, -0.40, 1.30 m | 3 / 60 |
| UR10e | 27 / 32 | -0.60, -0.80, 1.30 m | 19 / 60 |
| CRX-5iA | 13 / 32 | -0.60, -0.60, 1.30 m | 8 / 60 |

## Cycle

- Full collision-free cycle for **22 of 32** targets with UR10e.
- Cycle time under joint speed, acceleration and tool speed limits: mean 15.5 s (min 10.1, max 23.3). This is a ceiling, not a production rate.

## How sure are we

The scene was re-simulated on 40 plausible variants drawn from the uncertainty on each dimension.

| Layout | Drop unreachable | ≥ 80 % of targets | ≥ 90 % of targets |
|---|---|---|---|
| Fixed pedestal | 0 / 40 | 35 | 2 |
| Vertical axis | 0 / 40 | 23 | 5 |

**Measure on site first** (dimensions that move the result most):

- `pod_gap` — portal plane to pod face (currently 0.1 ± 0.05 m, correlation -0.40)
- `tote_dx` — portal column to tote centre, to the left (currently 0.55 ± 0.1 m, correlation -0.38)
- `bin_z0` — bottom of the lowest cell (currently 0.2 ± 0.05 m, correlation +0.26)
- `pod_w` — pod face width (currently 0.95 ± 0.06 m, correlation -0.24)

## Tool

**suction_wand** (suction): long vacuum wand: reaches deep into cells and bins; needs a flat, sealed face.

Why this tool: Goods-to-person items are mostly boxed or bagged, with a flat face, and sit up to 0.4 m deep in narrow cells: a long vacuum wand reaches them without needing room for fingers.

Payload with the heaviest part (1.00 kg) and the tool (0.60 kg):

- UR5e: 1.60 of 5.0 kg
- UR10e: 1.60 of 12.5 kg
- CRX-5iA: 1.60 of 5.0 kg

## How this was computed

Simulation and planning run on Google DeepMind's robotics plugin (MuJoCo 4.0.1, plugin commit `3dde338`):

- Reach, collisions, paths and cycle times: `motion-planning` (multi-start IK with collision filtering, bidirectional RRT, Cartesian strokes, trajectories limited in joint speed and acceleration and to 1.0 m/s at the tool).
- Station model: MjSpec, written to the plugin's MuJoCo asset standards (`station model.xml`: visual group 2, collision group 3).

## Models and assets used

| Robot | Model | Licence | Reach, payload, joint speeds |
|---|---|---|---|
| UR5e | MuJoCo Menagerie `universal_robots_ur5e` at `98a70c5` | BSD-3-Clause | https://www.universal-robots.com/manuals/latest/en/datasheets/ur5e/ (wrist speeds kept at the conservative 180 deg/s) |
| UR10e | MuJoCo Menagerie `universal_robots_ur10e` at `98a70c5` | BSD-3-Clause | https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-usr-man/complianceUR10e/H_g5_sections/appendix_g5/tech_spec_sheet.htm |
| CRX-5iA | MuJoCo Menagerie `fanuc_crx5ia` at `98a70c5` | Apache-2.0 | https://crx.fanucamerica.com/crx-5ia (reach, payload); https://robotec.org/products/cobots/fanuc-crx-5ia.html (joint speeds) |

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
- Heaviest item about 1 kg, typical for goods-to-person picking; not measured.
