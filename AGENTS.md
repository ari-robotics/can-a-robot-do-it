# For coding agents working in this repository

This repo answers one question for a person: "can a robot do this workstation?", from their photos or video.

If the person shares photos or a video of a workstation, asks whether a robot could do a task, or asks you to install or set this up: read `.claude/skills/workstation-sim/SKILL.md` and follow it. Do not create a virtual environment and do not run pip: the launcher `.claude/skills/workstation-sim/scripts/ari-sim` installs everything on its first call (start with `.claude/skills/workstation-sim/scripts/ari-sim doctor`).

Rules that hold everywhere here:

- Never ask the person for an API key. Gemini is used only if `GEMINI_API_KEY` is already set.
- The kind of gripper follows from the parts, never from which tool scores better in simulation.
- Never get blocked on a missing preset: search the libraries, or ask the person one short question.
- Fetch assets one at a time, never in bulk.
- All computation goes through Google DeepMind's robotics plugin (`ari_sim/google.py`); do not add a home-made planner or renderer.

To work on the code itself: `.venv` or the launcher's environment, `pytest -q`, and the example in `examples/goods_to_person_station/`.
