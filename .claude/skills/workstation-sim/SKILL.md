---
name: workstation-sim
description: Turn photos or a video of a workstation into a robot simulation and a feasibility diagnostic, running on Google DeepMind's robotics plugin (MuJoCo, motion planning, audit, Studio). Use when someone shares pictures of a workstation, a production cell, a pick station or a manual task and asks whether a robot could do it, which robot, where to put it, or wants "a sim of this station".
---

# Workstation diagnostic from photos, on Google's robotics tools

You do the reading: you look at the images and write `scene.json`. Everything computed after that runs on the skills of Google DeepMind's robotics plugin (`google-deepmind/robotics-antigravity-plugin`). `ari-sim` is the thin layer that calls them in the right order for one question: can a robot do this station, which one, standing where. Work in a new folder `runs/<short-name>/`.

Which Google tool does what:

| Step | Google skill | What it does here |
|---|---|---|
| 1. Read the photos | `rgb-object-localization` | metres from pixels with a scale reference (`ari-sim measure`); Gemini pointing at objects only if the person set `GEMINI_API_KEY` |
| 2. Rebuild the station | `mujoco-spec-editing`, rule `mujoco_asset_standards` | the station as a MuJoCo asset: visual group 2, collision group 3, hollow containers |
| 2. Missing object | `blender-creation` | model an object no library holds, from the photos |
| 3. Check the rebuild | `mujoco-audit`, `mujoco-showcase-shot`, `mujoco-studio-sidecar` + `mujoco-studio` | static and physics audit, turntable and collision view, live 3D view |
| 4. Search the cell | `motion-planning` | IK, collisions, paths and timed trajectories for every arm, tool and base |
| 5. Execute with physics | `iteration`, `rgb-object-localization`, `motion-planning`, `visual-servoing` | staged pick, part located by camera, success measured after release and settle |
| 6. Compare two cells | `sim2real-gauntlet`, `sim2real-iteration` | perturbation tiers, paired comparison, no claim from one seed |
| 7. Real arm | `mujoco-playground`, `so101-*`, `i2rt-yam-*`, `fr3-duo-control`, `mujoco-playground-rl` | same controller on a real SO-101, YAM or FR3 the person owns |
| 8. Share | `share-example` | bundle for GitHub Discussions |

Load the Google skill named in a step before doing that step by hand; `ari-sim` already calls its script where a command is given.

## 0. Get the code and the tools

The steps below run from a checkout of the repo. If the current folder has no `ari_sim/`, clone it and install it in a virtual environment with Python 3.11 or later:
`git clone https://github.com/ari-robotics/can-a-robot-do-it && cd can-a-robot-do-it && pip install -e ".[simready]" && ari-sim setup`

`ari-sim setup` fetches Google's plugin at a pinned commit if no agent has installed it (`npx skills add google-deepmind/robotics-antigravity-plugin` and `npx skills add google-deepmind/mujoco` install the same skills for you to load). `ari-sim doctor` lists every tool, what it is used for, and what is missing. Google's standalone scripts run through `uv`.

## 1. Look, then ask only what the images cannot tell you

Read every photo. For a video, pull frames first:
`python -c "from ari_sim.perceive import frames_from_video as f; print(f('clip.mp4', 8, 'runs/<name>/frames'))"`

You need one sentence for the task: what is picked, from where, and where it goes. If the person did not say and the images do not make it obvious, ask that one question. Do not ask for dimensions: estimating them, with their uncertainty, is the job.

Measure instead of guessing wherever a scale reference lies on the plane you care about. Read the pixel coordinates of four points of known position (the four rim corners of a 600 x 400 Euro tote, of a pallet, of an A4 sheet), then:
`ari-sim measure --ref U V X Y --ref ... (4 or more) --point U V --point ...`
Each point comes back in metres in the reference's frame, with a `sigma`: use it as the param's sigma. Points must lie on the same plane as the reference. With `GEMINI_API_KEY` set by the person, `ari-sim detect photo.jpg "black tote" "shelf"` returns the pixels; without it you read the pixels yourself, which is the plugin's own fallback. Never ask for a key.

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

- Every dimension comes from a scale reference you can actually see, measured with `ari-sim measure` where the geometry allows. List them. Never invent one.
- Every estimated dimension that affects reach is a param with a `sigma`. 10–20 % of the value is normal from one photo.
- Model what constrains a robot; leave out the rest. What a robot cell would remove goes in `assumptions`.
- `robot.base_search` is where a base could physically stand.

## 2b. Use real objects and real robots where the libraries have them

[SimReady Central](https://huggingface.co/spaces/nvidia/simready-central) indexes thousands of real objects and the arms and grippers their makers publish. Use it for what the photos show and the parametric components would only approximate. Never bulk-download: each step costs only what it fetches.

1. `ari-sim assets search <plain words>` for each object that matters (the drop container, a workbench, the items picked).
2. `ari-sim assets show <id>` for the two or three best hits, then open the thumbnail and compare it with the photos. Pick by eye, not by name.
3. `ari-sim assets get <id>`. Compare its `extent` with your estimate from the photos: if they disagree by more than your sigma, either the asset is the wrong one or your estimate is; say which. Read `warnings` and do not rely on a flagged mass.
4. Add it as an `asset` component with `"source": "simready:<id>"` (fixture, container or item).

If an object matters to the task and no library holds it, load Google's `blender-creation` skill and model it from the photos as a MuJoCo asset (it needs Blender; `ari-sim doctor` says whether it is installed). Then `ari-sim audit path/to/model.xml --scenarios all` before using it. Without Blender, keep the parametric components and say so.

Robots. `ari-sim robots list` shows the presets (reach, payload, joint speeds from datasheets), six- and seven-axis. To try another arm, find it with `ari-sim assets search`, check it with `ari-sim robots inspect <id>`, then add it to `robot.candidates` as `{"name", "source": "simready:<id>", "reach", "payload", "vmax_deg": [one speed per joint, deg/s], "datasheet": "<url>"}`. The robot file carries joint limits but not speeds, reach or payload: take them from the manufacturer's datasheet and cite it. If you cannot find the speeds, do not guess; say the arm could not be timed.

Say in the report which library assets you used, with their attribution line.

## 3. Check the rebuild against the photos

`ari-sim check runs/<name>/scene.json --out runs/<name>/out` writes `scene.png`, `colliders.png` (the shapes the planner avoids) and the station as a MuJoCo asset (`out/<station>/model.xml`). Compare `scene.png` with the photos: left/right order, heights, obstacles near the work zone. Fix the scene and re-check until it matches as well as the photos allow. Two or three rounds is typical; stop when the remaining differences are inside your sigmas.

Then:

- `ari-sim audit runs/<name>/scene.json --out runs/<name>/out`: Google's `mujoco-audit` static check on the station asset. Fix every error before going on.
- `ari-sim showcase runs/<name>/scene.json --out runs/<name>/out`: Google's turntable video and keyframe sheet (visual, collision overlay, collision only). Look at the keyframes: a container whose opening is capped by a collider, or an obstacle with no collider, will give a wrong answer.

## 4. Run the diagnostic

`ari-sim diagnose runs/<name>/scene.json --out runs/<name>/out` (two to four minutes). Every arm is tried at a grid of base positions and pedestal heights with Google's `motion-planning` planner; the best cell gets a full collision-free cycle planned and timed for every target; then the scene is re-simulated on dozens of plausible variants. Read `out/report.md` and look at `out/diagnostic.png`.

Show it live: `ari-sim studio runs/<name>/scene.json --out runs/<name>/out` starts Google's MuJoCo Studio daemon on a workspace in `out/studio/` and prints the address. Give the person the link (in Antigravity: `[MuJoCo Studio](sidecar://robotics/mujoco_studio/)`). They can watch each cycle, press `3` to see the collision shapes, and edit `scene.json` or the plugin files: saving reloads the view. After an edit, read the printed status URL and fix any error it reports.

## 4b. Iterate until a configuration works

The first run is rarely the answer. The use case is fixed (the station, the parts, the kind of tool); everything robotic is yours to change. If the verdict is not `reach_ok`, read which step failed in `report.md` and change the configuration, one thing at a time:

- Targets out of reach: longer arms from `ari-sim robots list`, then from the libraries; a wider `base_search`; another pedestal range; a vertical axis.
- Drop unreachable: move the base towards the drop, or propose moving the drop container (a layout change the person must accept: say so).
- Path blocked or collisions: another model of the same kind of tool (longer, slimmer), another base side, an obstacle the robot cell would remove (move it to `assumptions`).
- Over payload: a stronger arm, a lighter tool of the same kind.

Keep each run's folder (`out_1`, `out_2`...). When two configurations both pass and you must choose, follow Google's `sim2real-iteration` rule: compare them on the same variants (same `--variants`, same seed), never on one nominal run. Stop when a configuration passes, or when arms, bases and tool models are exhausted; then say plainly what was tried and what blocks. Do not change the kind of tool to get a better score.

## 5. Execute the pick with physics (experimental)

Only when the tool is the 2-finger gripper and the picks are shelf-type: `ari-sim execute runs/<name>/scene.json --episodes 60 --out runs/<name>/out/episodes`. Each episode follows Google's `iteration` skill: stages (locate the part with the station camera, approach, grasp, carry, release), each measured, success decided after the gripper has let go and the part is at rest. Report the outcome counts and the camera localisation error as they are; the scripted grasp is a weak baseline by design.

When a stage fails repeatedly, do what `iteration` says: give a separate subagent the video only and ask what happened, before changing code. For a grasp that needs final alignment, load `visual-servoing` and add a wrist camera. For a policy instead of a script, load `mujoco-playground-rl`.

## 6. Towards a real arm (only if the person has one)

If the chosen arm is one the plugin drives (SO-101, I2RT YAM, Franka FR3 duo), the same controller can run on the real arm: load `mujoco-playground` and the arm's bring-up skill (`so101-bringup`, `so101-calibration`, `i2rt-yam-bringup`, `i2rt-yam-ik`, `fr3-duo-control`), verify the controller with `sim2real-gauntlet` first (it needs `mujoco_playground`), and follow the plugin's safety rules. For any other arm, the diagnostic is the deliverable: an integrator takes it from there.

## 7. Report to the person

Lead with the verdict and the two or three dimensions to measure on site. Then: the kind of tool and why the parts call for it, the model chosen and its payload margin, the configurations tried and why the final one won, robot and base position, share of targets with a full collision-free cycle, cycle time as a ceiling, how the result holds across the variants. Always say what was assumed and that grasp reliability is not assessed by the diagnostic. Send `diagnostic.png`, `cycle.mp4` and the Studio link.

If the verdict is `no_feasible_layout`, look at which step failed (drop unreachable? targets out of reach?) and say what layout change would fix it, rather than only reporting the failure.

If the person wants to publish the run, load `share-example`.
