"""Photos or a video of a workstation -> scene.json, with Claude doing the reading.

The loop is agentic in the plain sense: the model proposes a scene, the code compiles it, and whatever
breaks (schema, physics, or a visual mismatch against the photos) goes back to the model to fix.
Needs ANTHROPIC_API_KEY. Without a key, use the Claude Code skill in .claude/skills/ instead: same steps, Claude Code does the reading.
"""
import os, io, json, base64, subprocess, tempfile, glob, re
import numpy as np
from PIL import Image
from . import spec as S

MODEL = os.environ.get('ARI_SIM_MODEL', 'claude-opus-5-5')
HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(os.path.dirname(HERE), 'examples', 'goods_to_person_station', 'scene.json')

SYSTEM = """You are a robotics integration engineer. From photos of a workstation and one sentence describing the task, you write a parametric scene description that a simulator compiles. Your output is used to decide whether a robot arm can do the task, so honest uncertainty matters more than visual polish.

## Output
Reply with ONE JSON object and nothing else.

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
Colors: black, yellow, steel, galvanised, red, white, grey, blue, green, wood, orange, or [r,g,b,a].
Model only what constrains a robot: surfaces, containers, structure, and obstacles near the work zone. Leave out cables, labels and far background. Things a person uses but a robot cell would remove (step stools, chairs) go in "assumptions", not in components.

## Task and robot
"task": {"description", "pick_inset": how far past the opening the tool goes (m), "pre_offset": stand-off before entering (m)}.
"robot": {"candidates": ["UR5e","UR10e","CRX-5iA"], "tool": {"type":"suction_wand","length":0.38}, "base_search": {"x":[lo,hi,n],"y":[lo,hi,n],"pedestal":[lo,hi,n]}, "observation_camera": {"pos","target","fovy"}}
base_search is the floor region where a robot base could physically stand (usually where the operator stands) and the pedestal heights to try.

## Honesty
List under "assumptions" everything you could not see and had to assume. If the photos do not show where items are picked from or dropped, say so there and make the most plausible choice. There must be at least one pick target and one place target."""

def frames_from_video(path, n=6, outdir=None):
    """Evenly spaced frames from a video (ffmpeg)."""
    outdir = outdir or tempfile.mkdtemp(prefix='ari_frames_')
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path], capture_output=True, text=True, check=True).stdout.strip())
    out = []
    for i in range(n):
        f = os.path.join(outdir, f'frame_{i:02d}.jpg')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{dur * (i + 0.5) / n:.2f}', '-i', path, '-frames:v', '1', '-q:v', '3', f], check=True); out.append(f)
    return out

def _img_block(path_or_array, max_side=1568):
    im = Image.open(path_or_array) if isinstance(path_or_array, str) else Image.fromarray(path_or_array)
    im = im.convert('RGB'); im.thumbnail((max_side, max_side)); b = io.BytesIO(); im.save(b, 'JPEG', quality=88)
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': base64.b64encode(b.getvalue()).decode()}}

def _json(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m: raise S.SceneError('the reply contained no JSON object')
    return json.loads(m.group(0))

def _check(sc):
    """Schema + compile check. Returns (resolved, None) or (None, error message for the model)."""
    try:
        sc = S.load(sc); R = S.resolve(sc)
        if not R['picks']: return None, 'The scene has no pick target. Add a shelf_unit with "pick": true or a pick_point.'
        if not R['places']: return None, 'The scene has no place target. Add a tote with "place": true or a place_point.'
        if 'base_search' not in sc.get('robot', {}): return None, 'robot.base_search is missing.'
        from .build import build
        build('UR10e', R, sc['robot'].get('tool'))
        return R, None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'

def perceive(images, task, out_path, video=None, critic_rounds=2, log=print):
    """images: list of photo paths. video: optional path, frames are added to the photos."""
    import anthropic
    client = anthropic.Anthropic(); images = list(images)
    if video: images += frames_from_video(video)
    if not images: raise ValueError('give at least one photo or a video')
    example = open(EXAMPLE).read()
    content = [_img_block(p) for p in images[:12]] + [{'type': 'text', 'text': f'Task: {task}\n\nHere is a complete example of the format, for a different workstation:\n{example}\n\nNow write the scene JSON for the workstation in the photos.'}]
    msgs = [{'role': 'user', 'content': content}]
    def ask():
        r = client.messages.create(model=MODEL, max_tokens=8000, system=SYSTEM, messages=msgs)
        text = ''.join(b.text for b in r.content if b.type == 'text'); msgs.append({'role': 'assistant', 'content': text}); return text
    sc = None
    for attempt in range(4):                       # propose -> compile -> repair
        text = ask()
        try: sc = _json(text); R, err = _check(sc)
        except Exception as e: R, err = None, f'Could not parse your reply as JSON: {e}'
        if err is None: break
        log(f'  scene rejected ({err[:90]}), asking for a fix'); msgs.append({'role': 'user', 'content': f'The simulator rejected the scene:\n{err}\nReply with the corrected full JSON only.'})
    else:
        raise S.SceneError('could not obtain a valid scene after 4 attempts')
    from .build import build, scene_view, render
    for rnd in range(critic_rounds):               # render -> compare with photos -> correct
        m, d, info = build('UR10e', R, sc['robot'].get('tool')); v = scene_view(R)
        sheet = np.hstack([render(m, d, v, 800, 600, az=118), render(m, d, v, 800, 600, az=62), render(m, d, v, 800, 600, az=90, el=-60)])
        msgs.append({'role': 'user', 'content': [_img_block(sheet, 2000), {'type': 'text', 'text': 'These are three renders of the scene you wrote (operator-side right, operator-side left, from above; the arm is just a placeholder). Compare them with the photos: relative positions, heights, what is left or right of what, missing obstacles near the work zone. If something is wrong, reply with the corrected full JSON only. If it matches as well as the photos allow, reply with the single word OK.'}]})
        text = ask()
        if text.strip().upper().startswith('OK'): log(f'  visual check passed (round {rnd + 1})'); break
        try: sc2 = _json(text); R2, err = _check(sc2)
        except Exception as e: err = str(e)
        if err: log(f'  correction rejected ({err[:80]}), keeping previous scene'); break
        sc, R = sc2, R2; log(f'  scene corrected after visual check (round {rnd + 1})')
    json.dump(sc, open(out_path, 'w'), indent=2, ensure_ascii=False)
    return sc
