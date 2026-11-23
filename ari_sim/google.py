"""Google DeepMind's robotics plugin (github.com/google-deepmind/robotics-antigravity-plugin): found, checked, imported.

Everything this package computes goes through that plugin's tools. Nothing is vendored: the skills are read from where
a coding agent installed them (`npx skills add google-deepmind/robotics-antigravity-plugin`), or fetched once at a
pinned commit into the asset cache.
"""
import os, sys, importlib, subprocess, shutil
from . import assets

REPO = 'https://github.com/google-deepmind/robotics-antigravity-plugin.git'
COMMIT = '3dde33820355c70fe8d8e33e42707cdcb742ceb4'      # plugin 0.1.12
CACHE = os.path.join(assets.HOME, 'robotics-antigravity-plugin')

# What each stage of the diagnostic uses. `module` is importable through load(); `script` runs on its own with `uv run`.
TOOLS = {
    'motion-planning':         dict(file='scripts/motion_planner.py', module='motion_planner', stage='plan',     use='IK, collision checks, joint paths and Cartesian strokes for every candidate arm'),
    'rgb-object-localization': dict(file='scripts/rgb_localization.py', module='rgb_localization', stage='perceive', use='metric positions from a photo and a scale reference; object detection with Gemini when a key is set'),
    'visual-servoing':         dict(file='scripts/ibvs.py', module='ibvs', stage='execute',  use='wrist-camera alignment before a physics grasp'),
    'mujoco-audit':            dict(file='scripts/audit_scenarios.py', script=True, stage='rebuild', use='static and physics audit of the rebuilt station and of every part'),
    'mujoco-showcase-shot':    dict(file='scripts/render_showcase_shot.py', script=True, stage='report', use='turntable video and inspection sheets of the station'),
    'mujoco-studio-sidecar':   dict(file='scripts/studiod.py', script=True, stage='rebuild', use='live 3D view of the station and of the winning cell'),
    'blender-creation':        dict(file='SKILL.md', stage='rebuild', use='an object no library holds, modelled from the photos (needs Blender)'),
    'iteration':               dict(file='SKILL.md', stage='execute',  use='staged execution, success measured after release and settle'),
    'sim2real-gauntlet':       dict(file='src/robosim_gauntlet/cli.py', stage='verify', use='perturbation tiers and paired comparison of two cells (needs mujoco_playground)'),
    'sim2real-iteration':      dict(file='SKILL.md', stage='verify',   use='how to compare two candidate cells without fooling yourself'),
    'mujoco-playground':       dict(file='scripts/playground_sim2real.py', stage='real', use='the same controller in simulation and on a real SO-101 or YAM'),
    'mujoco-playground-rl':    dict(file='scripts/rl_task_check.py', stage='real', use='a learned policy when the scripted grasp is not enough'),
    'i2rt-yam-ik':             dict(file='scripts/yam_ik.py', stage='real', use='IK on a real YAM'),
    'i2rt-yam-bringup':        dict(file='SKILL.md', stage='real', use='bring-up of a real YAM'),
    'so101-bringup':           dict(file='SKILL.md', stage='real', use='bring-up of a real SO-101'),
    'so101-calibration':       dict(file='SKILL.md', stage='real', use='hand-eye calibration of a real SO-101'),
    'fr3-duo-control':         dict(file='SKILL.md', stage='real', use='a real Franka Research 3 duo cell'),
    'share-example':           dict(file='SKILL.md', stage='report', use='package the run for GitHub Discussions'),
}

def _roots():
    env = os.environ.get('ARI_GOOGLE_SKILLS'); home = os.path.expanduser('~')
    return ([env] if env else []) + [os.path.join(home, '.claude', 'skills'), os.path.join(home, '.agents', 'skills'),
            os.path.join(home, '.gemini', 'config', 'plugins', 'robotics', 'skills'), os.path.join(CACHE, 'plugins', 'robotics', 'skills')]

def path(skill, need=True):
    """Folder of one of the plugin's skills, wherever it is installed."""
    f = TOOLS[skill]['file']
    for r in _roots():
        if os.path.exists(os.path.join(r, skill, f)): return os.path.join(r, skill)
    if need:          # no agent installed the skills on this machine: fetch the plugin once, at its pinned commit
        print("[ari-sim] fetching Google's robotics plugin (once)", file=sys.stderr); fetch()
        return os.path.join(CACHE, 'plugins', 'robotics', 'skills', skill)

def fetch():
    """The plugin at its pinned commit, in the asset cache. Used when no agent has installed the skills."""
    if not os.path.isdir(os.path.join(CACHE, '.git')):
        os.makedirs(assets.HOME, exist_ok=True); assets._run(['git', 'init', '-q', CACHE]); assets._run(['git', 'remote', 'add', 'origin', REPO], cwd=CACHE)
        assets._run(['git', 'fetch', '-q', '--depth', '1', 'origin', COMMIT], cwd=CACHE); assets._run(['git', 'checkout', '-q', COMMIT], cwd=CACHE)
    return CACHE

def load(skill):
    """Import a skill's Python module (motion_planner, rgb_localization, ibvs)."""
    t = TOOLS[skill]; d = os.path.join(path(skill), os.path.dirname(t['file']))
    if d not in sys.path: sys.path.insert(0, d)
    return importlib.import_module(t['module'])

def run(skill, args, timeout=1800):
    """Run a skill's standalone script in the isolated environment its header declares (`uv run`)."""
    if not shutil.which('uv'): raise RuntimeError('`uv` is needed to run Google\'s standalone scripts: https://docs.astral.sh/uv/')
    script = os.path.join(path(skill), TOOLS[skill]['file'])
    return subprocess.run(['uv', 'run', '--quiet', script] + list(args), capture_output=True, text=True, timeout=timeout)

def status():
    """One row per tool: where it is, or what is missing."""
    rows = []
    for k, t in TOOLS.items():
        p = path(k, need=False); rows.append(dict(skill=k, stage=t['stage'], use=t['use'], path=p, ok=p is not None))
    return rows
