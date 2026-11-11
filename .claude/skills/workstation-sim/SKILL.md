---
name: workstation-sim
description: Turn photos or a video of a workstation into a robot simulation and a reach diagnostic with this repo. Use when someone shares pictures of a workstation, a production cell, a pick station or a manual task and asks whether a robot could do it, where to put it, or wants "a sim of this station".
---

# Workstation diagnostic from photos

You do the reading: you look at the images and write `scene.json`. The code does the rest. Work in a new folder `runs/<short-name>/`.

## 0. Get the code

The steps below run from a checkout of the repo. If the current folder has no `ari_sim/` (for example when this skill came from the plugin), clone it, install it in a virtual environment with Python 3.10 or later, and work from there:
`git clone https://github.com/ari-robotics/can-a-robot-do-it && cd can-a-robot-do-it && pip install -e ".[simready]" && ari-sim setup`

## 1. Look, then ask only what the images cannot tell you

Read every photo. For a video, pull frames first:
`python -c "from ari_sim.perceive import frames_from_video as f; print(f('clip.mp4', 8, 'runs/<name>/frames'))"`

You need one sentence for the task: what is picked, from where, and where it goes. If the person did not say and the images do not make it obvious, ask that one question. Do not ask for dimensions: estimating them, with their uncertainty, is the job.

## 2. Write `runs/<name>/scene.json`

Follow `SCENE_RULES` in `ari_sim/perceive.py` exactly (frame, scale references, params with sigma, component library, assumptions) and use `examples/goods_to_person_station/scene.json` as the format reference. The points that matter most:

- Every dimension comes from a scale reference you can actually see. List them. Never invent one.
- Every estimated dimension that affects reach is a param with a `sigma`. 10–20 % of the value is normal from one photo.
- Model what constrains a robot; leave out the rest. What a robot cell would remove goes in `assumptions`.
- `robot.base_search` is where a base could physically stand.

## 2b. Use real objects where the catalog has them

The NVIDIA SimReady catalog holds 2,677 real objects (crates, bins, workbenches, conveyors, pallets, hand tools, boxes, parts) with true dimensions, mass and friction. Use it for what the photos show and the parametric components would only approximate. Never bulk-download: each step below costs only what it fetches.

1. `ari-sim assets search <a few words>` for each object that matters (the drop container, a workbench, the items picked). Use the catalog's words: `ari-sim assets categories` lists them (`crate`, `bin`, `handTool`, `box`, `pallet`, `furniture`...). Each hit shows its download size.
2. `ari-sim assets show <name>` for the two or three best hits, then open the thumbnail and compare it with the photos: same kind of object, same proportions, same openings. Pick by eye, not by name.
3. `ari-sim assets get <name>` for the one you chose. Read its `extent` against your estimate from the photos: if they disagree by more than your sigma, either the asset is the wrong one or your estimate is, say which. Read `warnings`: catalog data is sometimes wrong (a 36 cm item declared at 9 g); do not use an asset with a warning for mass-sensitive decisions.
4. Add it as an `asset` component (fixture, container or item). Its dimensions are now known; only its position stays a param with a sigma.

For robots, `ari-sim robots list` gives the arms with reach and payload; put the ones that fit the task in `robot.candidates`. Models are fetched on first use.

Say in the report which catalog assets you used, with their attribution line (CC-BY 4.0).

## 3. Check it against the photos

`ari-sim check runs/<name>/scene.json --out runs/<name>/out`, then open `out/scene.png` and compare it with the photos: left/right order, heights, obstacles near the work zone. Fix the scene and re-check until it matches as well as the photos allow. Two or three rounds is typical; stop when the remaining differences are inside your sigmas.

## 4. Run the diagnostic

`ari-sim diagnose runs/<name>/scene.json --out runs/<name>/out` (a few minutes). Read `out/report.md` and look at `out/diagnostic.png`.

## 5. Report to the person

Lead with the verdict and the two or three dimensions to measure on site. Then: robot and base position, share of targets with a full collision-free cycle, cycle time as a ceiling, how the result holds across the variants. Always say what was assumed and that grasp reliability is not assessed. Send `diagnostic.png` and `cycle.mp4`.

If the verdict is `no_feasible_layout`, look at which step failed (drop unreachable? targets out of reach?) and say what layout change would fix it, for example moving the drop container in front of the robot, rather than only reporting the failure.

## Optional: physics grasping

Only for shelf-type picks: `ari-sim grasp runs/<name>/scene.json --episodes 60 --out runs/<name>/out/episodes`. Report the outcome counts as they are; the scripted expert is a weak baseline by design.
