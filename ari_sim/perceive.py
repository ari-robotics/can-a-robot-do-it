"""What the agent needs to read a workstation: the scene-writing rules, and frames from a video.

There is no API call here. The coding agent (through the skill in .claude/skills/) looks at the photos
and writes scene.json following SCENE_RULES; the simulator compiles it and the agent fixes what breaks.
"""
import os, subprocess, tempfile

SCENE_RULES = """You are a robotics integration engineer. From photos of a workstation and one sentence describing the task, you write a parametric scene description that a simulator compiles. Your output is used to decide whether a robot arm can do the task, so honest uncertainty matters more than visual polish.

## Output
ONE JSON object, written to scene.json.

## Frame
Origin on the floor, z up, metres. The operator stands on the -y side and works towards +y; +x is the operator's right. State where you put the origin in "frame".

## How to get dimensions
A photo has no scale. Anchor every dimension on objects of known size that you can actually see (Euro tote 600x400 mm, Euro pallet 1200x800 mm, A4 sheet 210x297 mm, door 2.0-2.1 m, worktop 0.85-0.95 m, a screen whose diagonal you can estimate) and list them in "scale_references". Never invent a reference that is not visible.

## Params
Every dimension that matters to reach and that you had to estimate goes in "params" as {"value": metres, "sigma": 1-sigma uncertainty in metres, "note": what it is}. Typical sigma from a single photo is 10-20 % of the value; less when a scale reference sits right next to it, more for anything partly hidden. Components then refer to params by name, and may use arithmetic: "-(open_w/2 + 0.08)". Keep the list to what matters (about 5-12 params).

## Components (the whole library)
- {"type":"box","name","pos":[x,y,z] centre,"size":[sx,sy,sz] full sizes,"color","yaw":deg,"collide":true}
- {"type":"cylinder","name","pos":[x,y,z] centre,"size":[radius,height],"color"}
- {"type":"shelf_unit","name","origin":[x,y,z] floor point at the centre of the OPEN face,"width","height","depth","bin_depth","ncol","pitch","z0","color","yaw":deg,"pick":true}
  cells open towards -y (rotate with yaw); with "pick": true every cell becomes a pick target approached horizontally.
- {"type":"tote","name","center":[x,y],"rim": rim height,"size":[sx,sy,sz],"color","place":true} — an open box; with "place": true it is the drop target (approached from above).
- {"type":"pick_point","name","pos":[x,y,z],"approach":[ax,ay,az]} — a single pick location, e.g. a part on a table (approach [0,0,-1]).
- {"type":"place_point","name","pos":[x,y,z],"approach":[ax,ay,az]}
- {"type":"asset","name","source":"simready:<asset id from `ari-sim assets search`>","role":"fixture|container|item","pos":[x,y,z] floor point under its centre,"yaw":deg,"color"}
  a real object from the NVIDIA SimReady catalog, with its true dimensions. fixture: a solid obstacle (workbench, cabinet). container: an open box that
  becomes the drop target when "place": true (crate, bin). item: an object that is picked; no pos, it is placed in the pick cells by the grasp simulation.
  Use one only when the catalog holds the same kind of object at the right size; otherwise keep the parametric components.
Colors: black, yellow, steel, galvanised, red, white, grey, blue, green, wood, orange, or [r,g,b,a].
Model only what constrains a robot: surfaces, containers, structure, and obstacles near the work zone. Leave out cables, labels and far background. Things a person uses but a robot cell would remove (step stools, chairs) go in "assumptions", not in components.

## Task and robot
"task": {"description", "pick_inset": how far past the opening the tool goes (m), "pre_offset": stand-off before entering (m)}.
"task" may also carry "item_mass_kg": the heaviest part picked, your estimate from what it is (checked against payloads).
"robot.tool": a preset {"type": "suction_wand" | "suction_cup" | "gripper_2f85"} or any end effector from the libraries {"type": "custom", "name", "kind", "source": "simready:<id>", "mass", "stroke", "payload", "datasheet": url}; always with "why": one sentence tying the tool to the parts. The parts decide the tool: see the skill.
"robot": {"candidates": preset names from `ari-sim robots list` that suit the payload and reach (default ["UR5e","UR10e","CRX-5iA"]), or a SimReady arm as {"name","source":"simready:<id>","reach","payload","vmax_deg":[6 values],"datasheet":url}, "tool": {"type":"suction_wand","length":0.38}, "base_search": {"x":[lo,hi,n],"y":[lo,hi,n],"pedestal":[lo,hi,n]}, "observation_camera": {"pos","target","fovy"}}
base_search is the floor region where a robot base could physically stand (usually where the operator stands) and the pedestal heights to try.

## Honesty
List under "assumptions" everything you could not see and had to assume. If the photos do not show where items are picked from or dropped, say so there and make the most plausible choice. There must be at least one pick target and one place target."""

def frames_from_video(path, n=6, outdir=None):
    """Evenly spaced frames from a video (ffmpeg)."""
    outdir = outdir or tempfile.mkdtemp(prefix='ari_frames_'); os.makedirs(outdir, exist_ok=True)
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path], capture_output=True, text=True, check=True).stdout.strip())
    out = []
    for i in range(n):
        f = os.path.join(outdir, f'frame_{i:02d}.jpg')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{dur * (i + 0.5) / n:.2f}', '-i', path, '-frames:v', '1', '-q:v', '3', f], check=True); out.append(f)
    return out
