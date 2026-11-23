"""The rebuilt station and the winning cell, live in MuJoCo Studio (Google's `mujoco-studio-sidecar` skill).

A Studio workspace is three small files the person can open and edit: the simulation plugin rebuilds the scene from
scene.json through this package and replays the planned cycles; saving scene.json or a plugin hot-reloads the view."""
import os, sys, json, time, subprocess, tempfile, urllib.request
from . import google

MESSAGES = '''"""Messages between sim_plugin.py and viewer_plugin.py."""
import dataclasses
from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class CycleConfig(messages.Snapshot):
  """UI -> sim: which pick target to replay, and whether to loop over all of them."""
  target: int = 0
  loop_all: bool = True


@dataclasses.dataclass(frozen=True)
class CycleTelemetry(messages.Snapshot):
  """sim -> UI: what is playing."""
  robot: str = ''
  target: str = ''
  n_targets: int = 0
  cycle_s: float = 0.0
'''

SIM = '''"""Simulation side: the station rebuilt from scene.json, the chosen arm at its base, the planned cycles replayed.

Edit scene.json (or this file) and save: the scene is rebuilt in place."""
import json
import os

import messages as user_messages
import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
import numpy as np

from ari_sim import build
from ari_sim import robots
from ari_sim import spec as scene_spec

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, {scene!r})
CYCLE = os.path.join(HERE, {cycle!r})


class SimPlugin(plugin.SimPlugin):
  """Builds the cell and replays the planner's trajectories."""

  target: int = 0
  loop_all: bool = True
  t0: float = 0.0

  def build_spec(self) -> mujoco.MjSpec:
    sc = scene_spec.load(SCENE)
    cyc = json.load(open(CYCLE))
    for cand in sc.get('robot', {{}}).get('candidates', []):  # arms the scene declares itself (library models)
      robots.register(cand)
    _, _, info = build.scene(cyc['robot'], scene_spec.resolve(sc), cyc['tool'], cyc['base'], name=sc['name'])
    self.robot, self.dt, self.qadr = cyc['robot'], cyc['dt'], info['qadr']
    self.paths = [(t['name'], np.asarray(t['q']), t['t']) for t in cyc['targets']]
    spec = info['spec']
    spec.option.gravity = [0, 0, 0]  # a replay of planned joint paths, not a controller
    return spec

  @messages.handler
  def on_load_model(self, event: messages.LoadModelEvent) -> bool:
    if event.path:
      return False
    self.handle.set_model(self.build_spec())
    return True

  @messages.handler
  def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
    if self in event.plugins:
      self.handle.reload()

  @messages.handler
  def on_cycle_config(self, snapshot: user_messages.CycleConfig) -> None:
    self.target, self.loop_all = snapshot.target, snapshot.loop_all

  @messages.handler
  def on_post_step(self, event: messages.PostStepEvent) -> None:
    data = event.data
    if not self.paths:
      return
    name, q, cycle_s = self.paths[self.target % len(self.paths)]
    i = int((data.time - self.t0) / self.dt)
    if i >= len(q) + 15:
      self.t0 = float(data.time)
      if self.loop_all:
        self.target = (self.target + 1) % len(self.paths)
      i = 0
    data.qpos[self.qadr] = q[min(i, len(q) - 1)]
    data.qvel[:] = 0
    self.handle.send_to_viewer(user_messages.CycleTelemetry(
        robot=self.robot, target=name, n_targets=len(self.paths), cycle_s=float(cycle_s)))
'''

VIEWER = '''"""Viewer side: pick which cycle to watch. Press 3 to see the collision shapes the planner avoids."""
import messages as user_messages
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio.app import plugins as app_plugins

imgui = app_plugins.imgui


class ViewerPlugin(plugin.ViewerPlugin):
  target: int = 0
  loop_all: bool = True
  telemetry: user_messages.CycleTelemetry = user_messages.CycleTelemetry()
  window_open: bool = True

  @messages.handler
  def on_telemetry(self, snapshot: user_messages.CycleTelemetry) -> None:
    self.telemetry = snapshot

  @messages.handler
  def on_build_gui(self, event: messages.BuildGuiEvent) -> None:
    del event
    if not self.window_open:
      return
    expanded, self.window_open = imgui.Begin('Can a robot do it###CanARobotDoItWindow', self.window_open)
    try:
      if not expanded:
        return
      t = self.telemetry
      imgui.TextUnformatted(f'{{t.robot}}: {{t.target}}  (cycle about {{t.cycle_s:.1f}} s)')
      imgui.TextUnformatted(f'{{t.n_targets}} targets with a full collision-free cycle')
      imgui.Separator()
      c1, self.loop_all = imgui.Checkbox('Play every target in turn', self.loop_all)
      c2, target = imgui.InputInt('Target', self.target)
      if c2:
        self.target = max(0, target)
      if c1 or c2:
        self.viewer.send_to_sim(user_messages.CycleConfig(target=self.target, loop_all=self.loop_all))
    finally:
      imgui.End()
'''

def workspace(scene_path, outdir):
    """Write <outdir>/studio/{messages,sim_plugin,viewer_plugin}.py for a diagnosed scene."""
    cyc = os.path.join(outdir, 'cycle_paths.json')
    if not os.path.exists(cyc): raise FileNotFoundError(f'{cyc} not found: run `ari-sim diagnose` on this scene first')
    ws = os.path.join(outdir, 'studio'); os.makedirs(ws, exist_ok=True)
    rel = lambda p: os.path.relpath(os.path.abspath(p), ws)
    for name, text in (('messages.py', MESSAGES), ('sim_plugin.py', SIM.format(scene=rel(scene_path), cycle=rel(cyc))), ('viewer_plugin.py', VIEWER.replace('{{', '{').replace('}}', '}'))):
        open(os.path.join(ws, name), 'w').write(text)
    return ws

def _get(url):
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=5) as r: return json.load(r)

def launch(ws, test_steps=0, wait=40):
    """Start Google's studiod on the workspace. Returns {url, control, status}. test_steps > 0: run headless and exit."""
    script = os.path.join(google.path('mujoco-studio-sidecar'), google.TOOLS['mujoco-studio-sidecar']['file'])
    rt = os.path.join(tempfile.gettempdir(), 'ari_sim_studio'); os.makedirs(rt, exist_ok=True); sess = os.path.join(rt, 'session.json')
    if os.path.exists(sess): os.remove(sess)
    cmd = [sys.executable, script, f'--workspace_dir={os.path.abspath(ws)}', f'--session_file={sess}', f'--runtime_dir={rt}']
    if test_steps:
        r = subprocess.run(cmd + [f'--headless_test_steps={test_steps}'], capture_output=True, text=True, timeout=300)
        return dict(ok=r.returncode == 0, log=(r.stdout + r.stderr)[-3000:])
    log = open(os.path.join(rt, 'studiod.log'), 'w'); p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(wait * 2):
        time.sleep(0.5)
        if p.poll() is not None: raise RuntimeError('studiod stopped: ' + open(log.name).read()[-2000:])
        if os.path.exists(sess):
            try:
                s = json.load(open(sess)); st = _get(f"http://127.0.0.1:{s['control_port']}/_studio/status")
                return dict(pid=p.pid, session=s, status=st, log=log.name)
            except Exception: continue
    raise RuntimeError('studiod did not come up: ' + open(log.name).read()[-2000:])
