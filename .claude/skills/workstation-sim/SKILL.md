---
name: workstation-sim
description: Turn photos or a video of a workstation into a robot simulation and a reach diagnostic with this repo. Use when someone shares pictures of a workstation, a production cell, a pick station or a manual task and asks whether a robot could do it, where to put it, or wants "a sim of this station".
---

# Workstation diagnostic from photos

You do the reading: you look at the images and write `scene.json`. The code does the rest. Work in a new folder `runs/<short-name>/`.

## 0. Get the code

The steps below run from a checkout of the repo. If the current folder has no `ari_sim/` (for example when this skill came from the plugin), clone it, install it in a virtual environment with Python 3.11 or later (MuJoCo 4 needs it), and work from there:
`git clone https://github.com/ari-robotics/can-a-robot-do-it && cd can-a-robot-do-it && pip install -e ".[simready]" && ari-sim setup`

## 1. Look, then ask only what the images cannot tell you

Read every photo. For a video, pull frames first:
`python -c "from ari_sim.perceive import frames_from_video as f; print(f('clip.mp4', 8, 'runs/<name>/frames'))"`

You need one sentence for the task: what is picked, from where, and where it goes. If the person did not say and the images do not make it obvious, ask that one question. Do not ask for dimensions: estimating them, with their uncertainty, is the job.

## 1b. Decide how the robot holds the parts: the use case decides

The kind of tool follows from what is picked and where it sits, never from which tool scores better in simulation. Simulation comes after: it tells whether a robotic configuration with that kind of tool can do the job, and which one. Write the decision in `robot.tool` with a `why` that names the parts, and estimate the heaviest part's mass in `task.item_mass_kg`.

| What is picked | Kind of tool |
|---|---|
| Cartons, cases, boxed goods, panels, sheet metal, sealed bags: a flat, smooth, airtight face of at least 3–4 cm | Suction |
| Parts with no flat sealed face (porous, mesh, cloth, round, irregular), tools, bottles, machined parts, anything to orient or insert | Fingers |
| Ferrous parts, very soft or tangled items, food, anything else | The tool the trade uses for it (magnet, soft gripper, needle gripper...): look it up |
| Several kinds in the same flow | One kind per family of parts; say so, and simulate each family with its tool |

Then pick the model inside that kind, from everything the libraries hold; never stop at "not in the presets":

1. Presets (`suction_wand`, `suction_cup`, `gripper_2f85`) when they fit: stroke wider than the part's gripping side, payload above the part's mass, length enough to reach where the part sits.
2. Otherwise `ari-sim assets search <kind of gripper, maker>`: SimReady holds makers' end effectors (Robotiq 2F-140 and Hand-E, OnRobot RG2/RG6/2FG7/2FG14, more over time). Declare the one that fits as `{"type": "custom", "name", "kind", "source": "simready:<id>", "mass", "stroke", "payload", "datasheet": "<url>"}`: its real geometry is mounted on the flange. Stroke, payload and mass come from the maker's datasheet.
3. If no library has a model of the right tool, still do not stop: declare it as a custom tool with its `length` and `mass` from a datasheet (it is then simulated as a slender body of that length) and say in the report that its shape is approximate.

Ask the person only for facts about the use case that the photos cannot show and that change the kind of tool: what the parts are made of, whether they have a flat airtight face, how heavy the heaviest one is, whether the parts vary. One short question with the options laid out (in Claude Code, use the question tool). Do not ask which tool they want: that is your job.

## 2. Write `runs/<name>/scene.json`

Follow `SCENE_RULES` in `ari_sim/perceive.py` exactly (frame, scale references, params with sigma, component library, assumptions) and use `examples/goods_to_person_station/scene.json` as the format reference. The points that matter most:

- Every dimension comes from a scale reference you can actually see. List them. Never invent one.
- Every estimated dimension that affects reach is a param with a `sigma`. 10–20 % of the value is normal from one photo.
- Model what constrains a robot; leave out the rest. What a robot cell would remove goes in `assumptions`.
- `robot.base_search` is where a base could physically stand.

## 2b. Use real objects and real robots where SimReady has them

[SimReady Central](https://huggingface.co/spaces/nvidia/simready-central) indexes every SimReady dataset on Hugging Face: thousands of real objects (crates, bins, workbenches, conveyors, pallets, tools, parts) and robot arms and grippers published by their manufacturers (FANUC, Universal Robots, Franka, Comau, Robotiq, OnRobot...), with true dimensions and physics. Use it for what the photos show and the parametric components would only approximate. Never bulk-download: each step costs only what it fetches.

1. `ari-sim assets search <plain words>` for each object that matters (the drop container, a workbench, the items picked). The search understands descriptions ("plastic crate 600x400", "fanuc collaborative arm"). Each hit is an asset id with its download size.
2. `ari-sim assets show <id>` for the two or three best hits, then open the thumbnail and compare it with the photos: same kind of object, same proportions, same openings. Pick by eye, not by name.
3. `ari-sim assets get <id>` for the object you chose. Compare its `extent` with your estimate from the photos: if they disagree by more than your sigma, either the asset is the wrong one or your estimate is; say which. Read `warnings` and do not rely on a flagged mass.
4. Add it as an `asset` component with `"source": "simready:<id>"` (fixture, container or item). Its dimensions are now known; only its position stays a param with a sigma.

Robots. `ari-sim robots list` shows the presets (reach, payload, joint speeds from datasheets). To try another arm, find it with `ari-sim assets search`, check it with `ari-sim robots inspect <id>` (it must have six joints and a flange), then add it to `robot.candidates` as `{"name", "source": "simready:<id>", "reach", "payload", "vmax_deg": [six joint speeds in deg/s], "datasheet": "<url>"}`. The robot file carries joint limits but not speeds, reach or payload: take them from the manufacturer's datasheet and cite it. If you cannot find the speeds, do not guess; say the arm could not be timed.

Say in the report which SimReady assets you used, with their attribution line.

## 3. Check it against the photos

`ari-sim check runs/<name>/scene.json --out runs/<name>/out`, then open `out/scene.png` and compare it with the photos: left/right order, heights, obstacles near the work zone. Fix the scene and re-check until it matches as well as the photos allow. Two or three rounds is typical; stop when the remaining differences are inside your sigmas.

## 4. Run the diagnostic

`ari-sim diagnose runs/<name>/scene.json --out runs/<name>/out` (a few minutes). Read `out/report.md` and look at `out/diagnostic.png`.

## 4b. Iterate until a configuration works

The first run is rarely the answer. The use case is fixed (the station, the parts, the kind of tool); everything robotic is yours to change, and each run takes about a minute. If the verdict is not `reach_ok`, read which step failed in `report.md` and change the configuration, one thing at a time:

- Targets out of reach: longer arms from `ari-sim robots list`, then from the libraries (`ari-sim assets search <maker> arm`, `ari-sim robots inspect <id>`, datasheet speeds); a wider `base_search`; another pedestal range; a vertical axis.
- Drop unreachable: move the base towards the drop, or propose moving the drop container (a layout change the person must accept: say so).
- Path blocked or collisions: another model of the same kind of tool (longer, slimmer), another base side, an obstacle the robot cell would remove (move it to `assumptions`).
- Over payload: a stronger arm, a lighter tool of the same kind.

Keep each run's folder (`out_1`, `out_2`...). Stop when a configuration passes, or when arms, bases and tool models are exhausted; then say plainly what was tried and what blocks. Do not change the kind of tool to get a better score.

## 5. Report to the person

Lead with the verdict and the two or three dimensions to measure on site. Then: the kind of tool and why the parts call for it, the model chosen and its payload margin, the configurations tried and why the final one won, robot and base position, share of targets with a full collision-free cycle, cycle time as a ceiling, how the result holds across the variants. Always say what was assumed and that grasp reliability is not assessed. Send `diagnostic.png` and `cycle.mp4`.

If the verdict is `no_feasible_layout`, look at which step failed (drop unreachable? targets out of reach?) and say what layout change would fix it, for example moving the drop container in front of the robot, rather than only reporting the failure.

## Optional: physics grasping

Only when the tool is the 2-finger gripper, and only for shelf-type picks: `ari-sim grasp runs/<name>/scene.json --episodes 60 --out runs/<name>/out/episodes`. Report the outcome counts as they are; the scripted expert is a weak baseline by design.
