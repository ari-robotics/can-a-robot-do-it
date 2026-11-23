"""EXPERIMENTAL — the pick executed with contact physics in the rebuilt station.

The chosen arm with a Robotiq 2F-85, real scanned objects (or the scene's own parts) dropped in the pick cells at
random poses. One episode is run the way Google's `iteration` skill asks: staged (locate, approach, grasp, carry,
release), each stage measured, and success decided only after the gripper has let go and the part has come to rest.
The part is located from the station camera with `rgb-object-localization` (ray-plane projection of its mask), the
motions come from `motion-planning`. Each episode is recorded (camera image, joints, gripper, action) as raw material
for imitation learning.

Limits: horizontal-approach pick targets only (shelf cells); one fixed side-grasp strategy; arms that ship with
position actuators (MuJoCo Menagerie models)."""
import os, json, warnings, collections
import numpy as np, mujoco
warnings.filterwarnings('ignore')
from . import spec as S, assets, build, google
from .plan import Cell, frame

GRIPPER = dict(type='gripper_2f85', kind='fingers', length=0.15, mass=0.93, stroke=0.085, payload=5.0)
SIDE = (np.pi / 2, -np.pi / 2)          # tool spins that put the fingers' closing axis horizontal

class Env:
    def __init__(self, sc, base, names, robot='UR10e', values=None, seed=0, img=(320, 240), perception='camera'):
        self.sc = sc; self.R = S.resolve(sc, values); self.names = list(names); self.rng = np.random.default_rng(seed); self.perception = perception
        cam = sc.get('robot', {}).get('observation_camera') or dict(pos=[-0.85, -1.25, 2.25], target=[0, 0.25, 1.15])
        self.m, self.d, self.info = build.scene(robot, self.R, GRIPPER, base, objects=self.names, camera=cam, name=sc['name']); m = self.m
        self.cell = Cell(robot, GRIPPER).place(self.R, *base); self.cell.spins = SIDE; self.qa = self.info['qadr']; self.site = self.info['tip_site']; self.obj = self.info['objects']
        jid = {m.jnt_qposadr[j]: j for j in range(m.njnt)}; arm_j = [jid[a] for a in self.qa]
        self.act = [next((k for k in range(m.nu) if m.actuator_trnid[k, 0] == j), None) for j in arm_j]
        if None in self.act: raise S.SceneError(f'execute: {robot} has no joint actuators in its model; physics execution needs a MuJoCo Menagerie arm')
        self.grip = m.actuator('g_fingers_actuator').id; self.gq = m.jnt_qposadr[m.actuator_trnid[self.grip, 0]] if m.actuator_trntype[self.grip] == 0 else None
        self.qadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in self.obj]; self.vadr = [m.jnt_dofadr[m.body_jntadr[b]] for b in self.obj]; self.verts = []
        for b in self.obj:
            g0 = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH][0]
            mid = m.geom_dataid[g0]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[g0]); v = v @ Rg.reshape(3, 3).T + m.geom_pos[g0]
            self.verts.append(v[:: max(1, len(v) // 3000)].copy())
            mass = float(np.clip(m.body_mass[b], 0.03, 0.35)); k = mass / m.body_mass[b]; m.body_mass[b] *= k; m.body_inertia[b] *= k
        # the real 2F-85 grips far harder than the stock model's 5 N limit; silicone pads
        m.actuator_forcerange[self.grip] = [-12, 12]
        for g in range(m.ngeom):
            if 'pad' in m.body(m.geom_bodyid[g]).name and m.geom_contype[g] > 0: m.geom_friction[g, 0] = 1.5
        m.opt.timestep = 0.001; m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        mujoco.mj_setConst(m, self.d); mujoco.mj_resetData(m, self.d); self.park = self.d.qpos.copy(); self.dt = m.opt.timestep
        self.targets = [t for t in self.R['picks'] if abs(t['approach'][2]) < 0.3]
        if not self.targets: raise S.SceneError('execute: no pick target with a horizontal approach (shelf cells) in this scene')
        pl = self.R['places'][0]; self.place = pl; c = self.cell; c.spins = (0.0, np.pi / 2)
        self.qhome, _ = c.reach(pl['pre'] + np.array([0, 0, 0.10]), pl['approach'], restarts=24)
        self.qdrop, _ = c.reach(pl['drop'] + np.array([0, 0, 0.04]), pl['approach'], self.qhome, restarts=8) if self.qhome is not None else (None, '')
        if self.qhome is None or self.qdrop is None: raise S.SceneError('execute: the drop target is not reachable from this base position')
        self.qdrop = c.near(self.qdrop, self.qhome); c.spins = SIDE
        self.ren = mujoco.Renderer(m, img[1], img[0]); self.seg = mujoco.Renderer(m, 480, 640); self.seg.enable_segmentation_rendering(); self.camid = m.camera('station_cam').id
        self.loc = google.load('rgb-object-localization') if perception == 'camera' else None
    # ---------- perception: where is the part, from the station camera
    def locate(self, i, z_plane):
        """Centre of part i on the horizontal plane z_plane, from its mask in the station camera (ray-plane projection)."""
        m, d = self.m, self.d; self.seg.update_scene(d, camera=self.camid); ids = self.seg.render()[:, :, 0]
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj[i]]; mask = np.isin(ids, geoms).astype(np.uint8) * 255
        if mask.sum() < 255 * 60: return None
        f = 0.5 * 480 / np.tan(np.radians(m.cam_fovy[self.camid]) / 2); K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.0]])
        T = np.eye(4); T[:3, :3] = d.cam_xmat[self.camid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]); T[:3, 3] = d.cam_xpos[self.camid]
        r = self.loc.ray_plane_contour_pose(mask, K, T, float(z_plane), min_area_px=40.0)
        return None if r is None else np.array([r['x_m'], r['y_m']])
    # ---------- objects
    def points(self, i):
        b = self.obj[i]; return self.verts[i] @ self.d.xmat[b].reshape(3, 3).T + self.d.xpos[b]
    def put(self, i, xyz, yaw):
        a = self.qadr[i]; self.d.qpos[a:a + 3] = xyz; self.d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]; self.d.qvel[self.vadr[i]:self.vadr[i] + 6] = 0
    def reset(self, k=4):
        m, d, rng = self.m, self.d, self.rng
        mujoco.mj_resetData(m, d); d.qpos[:] = self.park; d.qpos[self.qa] = self.qhome; d.ctrl[self.act] = self.qhome; d.ctrl[self.grip] = 0
        idx = rng.choice(len(self.targets), size=min(k, len(self.obj), len(self.targets)), replace=False)
        chosen = rng.choice(len(self.obj), size=len(idx), replace=False); self.placed = {}
        for oi, ti in zip(chosen, idx):
            t = self.targets[ti]; a = t['approach']; f = frame(a)[:, 0]; v = self.verts[oi]; yaw0 = np.arctan2(a[1], a[0]) - np.pi / 2
            yaw = yaw0 + float(rng.choice([0, np.pi / 2, np.pi, -np.pi / 2]) + rng.normal(0, 0.15))
            p = t['face'] + a * rng.uniform(0.09, 0.15) + f * rng.uniform(-0.025, 0.025)
            self.put(oi, [p[0], p[1], t['support_z'] - v[:, 2].min() + 0.004], yaw); self.placed[int(oi)] = int(ti)
        mujoco.mj_forward(m, d)
        for _ in range(700): mujoco.mj_step(m, d)
        return self.placed
    # ---------- execution
    def _rec(self, tgt):
        if self.log is None: return
        d = self.d; self.ren.update_scene(d, camera=self.camid)
        self.log['img'].append(self.ren.render().copy()); self.log['q'].append(d.qpos[self.qa].copy()); self.log['grip'].append(float(d.ctrl[self.grip]))
        self.log['tcp'].append(d.site_xpos[self.site].copy()); self.log['obj'].append(np.concatenate([d.xpos[self.obj[tgt]], d.xquat[self.obj[tgt]]])); self.log['act'].append(np.append(d.ctrl[self.act], d.ctrl[self.grip]))
    def _step(self, tgt):
        mujoco.mj_step(self.m, self.d); self.k += 1
        if not np.isfinite(self.d.qacc).all() or np.abs(self.d.qvel).max() > 500: raise FloatingPointError('unstable')
        if self.k % 200 == 0: self._rec(tgt)                       # 5 Hz
        if self.video and self.k % self.vevery == 0: self.video(self)
    def follow(self, traj, tgt, grip, slow=1.5):
        """Track a planner trajectory (one row per plan.DT) with the arm's position actuators, `slow` times slower."""
        from .plan import DT
        P = np.asarray(traj); n = max(int(round((len(P) - 1) * DT * slow / self.dt)), 1)
        for k in range(1, n + 1):
            x = k / n * (len(P) - 1); i = min(int(x), len(P) - 2) if len(P) > 1 else 0
            self.d.ctrl[self.act] = P[i] + (P[i + 1] - P[i]) * (x - i) if len(P) > 1 else P[0]; self.d.ctrl[self.grip] = grip; self._step(tgt)
        return n * self.dt
    def hold(self, t, tgt, grip):
        self.d.ctrl[self.grip] = grip
        for _ in range(int(t / self.dt)): self._step(tgt)
    def settle(self, tgt, b, tmax=2.0):
        """Wait until the part is at rest (the `iteration` rule: measure after release and settle)."""
        for _ in range(int(tmax / self.dt)):
            self._step(tgt)
            if self.k % 100 == 0 and np.linalg.norm(self.d.cvel[b]) < 0.02: return True
        return False
    def episode(self, record=True, video=None, vevery=33):
        self._out = {}
        try: return self._episode(record, video, vevery)
        except FloatingPointError: return dict(self._out, result='sim_unstable'), None
    def _episode(self, record, video, vevery):
        self.log = dict(img=[], q=[], grip=[], tcp=[], obj=[], act=[]) if record else None; self.k = 0; self.video = video; self.vevery = vevery
        placed = self.reset(); d = self.d; c = self.cell
        tgt = int(self.rng.choice(list(placed))); t = self.targets[placed[tgt]]; b = self.obj[tgt]; a = t['approach']; f = frame(a)[:, 0]
        # stage 1: locate the part
        pts = self.points(tgt) - t['face']; u = pts @ f; v = pts @ a; z = pts[:, 2] + t['face'][2]
        out = self._out = dict(object=self.names[tgt], target=t['name'], width=float(u.max() - u.min()), height=float(z.max() - z.min()))
        if v.min() < -0.03 or d.xpos[b][2] < t['support_z'] - 0.03: return dict(out, result='fell_at_reset'), None
        if u.max() - u.min() > 0.074: return dict(out, result='too_wide'), None
        zg = float(np.clip((z.min() + z.max()) / 2, t['support_z'] + 0.05, t['top_z'] - 0.05)); ug = float((u.min() + u.max()) / 2); vc = float((v.min() + v.max()) / 2)
        if self.loc is not None:
            xy = self.locate(tgt, zg)
            if xy is None: out['located'] = 'not_seen'
            else:
                rel = np.array([xy[0], xy[1], 0]) - np.array([t['face'][0], t['face'][1], 0]); uc, vcam = float(rel @ f), float(rel @ a)
                out['loc_err_m'] = float(np.hypot(uc - ug, vcam - vc))
                if abs(uc) < 0.2 and -0.03 < vcam < 0.5: out['located'] = 'camera'; ug, vc = uc, vcam
                else: out['located'] = 'camera_rejected'          # a mask cut by a shelf edge projects outside the cell: not believed
        vg = float(np.clip(vc, vc - (v.max() - v.min()) / 2 + 0.02, vc - (v.max() - v.min()) / 2 + 0.065))
        P = lambda uu, vv, zz: np.array([t['face'][0], t['face'][1], 0]) + f * uu + a * vv + np.array([0, 0, zz])
        pre, grasp = P(ug, -0.13, zg), P(ug, vg, zg)
        # stage 2: plan approach, insertion, lift and retreat
        sols = c.reach_near(pre, a, self.qhome); ok = None
        for qpre in sols[:2]:
            ins = c.stroke(qpre, grasp)
            if ins[0] is None: continue
            path = c.move(self.qhome, qpre)
            if path[0] is None: continue
            lift = c.stroke(ins[0][-1], grasp + [0, 0, 0.012]); ret = c.stroke(lift[0][-1], pre + [0, 0, 0.012]) if lift[0] is not None else (None, '')
            if ret[0] is not None: ok = (path[0], ins[0], lift[0], ret[0]); break
        if ok is None: return dict(out, result='unreachable' if not sols else 'no_path'), None
        path, ins, lift, ret = ok
        # stage 3: approach, grasp, lift; measured: is the part in the hand
        T = self.follow(path, tgt, 0) + self.follow(ins, tgt, 0, slow=2.5); self.hold(0.6, tgt, 255)
        T += self.follow(lift, tgt, 255, slow=3) + self.follow(ret, tgt, 255, slow=2.5)
        held = np.linalg.norm(d.xpos[b] - d.site_xpos[self.site]) < 0.12 and ((self.points(tgt) - t['face']) @ a).min() < 0
        back = np.vstack([path[::-1]])
        if not held: self.follow(back, tgt, 0); return dict(out, result='grasp_failed', time=T + 0.6), self.log
        # stage 4: carry and release; measured after the part has come to rest
        down, _ = c.move(self.qhome, self.qdrop)
        if down is None: down = np.vstack([self.qhome, self.qdrop])
        T += self.follow(back, tgt, 255, slow=2) + self.follow(down, tgt, 255, slow=2)
        self.hold(0.5, tgt, 0); self.follow(down[::-1], tgt, 0); rest = self.settle(tgt, b)
        p = d.xpos[b]; bx = self.place.get('box')
        if bx: ok = abs(p[0] - bx['center'][0]) < bx['half'][0] + 0.01 and abs(p[1] - bx['center'][1]) < bx['half'][1] + 0.01 and bx['rim'] - bx['depth'] - 0.04 < p[2] < bx['rim'] + 0.03
        else: ok = np.linalg.norm(p[:2] - self.place['drop'][:2]) < 0.15
        return dict(out, result='success' if ok and rest else ('not_settled' if ok else 'dropped'), time=T + 1.1, final=[float(x) for x in p]), self.log

def run_batch(sc, base, episodes, outdir, robot='UR10e', seed=0, per_scene=6, perception='camera', log=print):
    """Episodes over the scene's `item` assets if it declares any, else over all graspable scanned objects; 6 objects
    loaded at a time. Saves one .npz per attempted episode + stats.json."""
    items = [it['source'] for it in S.resolve(sc)['items']]
    names = items or assets.graspable_objects(); os.makedirs(outdir, exist_ok=True)
    # near-equal groups, episodes in proportion to group size: every object gets about the same share
    groups = [list(g) for g in np.array_split(names, int(np.ceil(len(names) / per_scene)))]
    cum = np.round(episodes * np.cumsum([0] + [len(g) for g in groups]) / len(names)).astype(int); rows = []
    for gi, g in enumerate(groups):
        if cum[gi + 1] == cum[gi]: continue
        env = Env(sc, base, g, robot=robot, seed=seed + gi, perception=perception)
        for ep in range(cum[gi + 1] - cum[gi]):
            out, lg = env.episode(record=True); out.update(group=gi, ep=ep); rows.append(out)
            if lg is not None and len(lg['q']) > 5:
                np.savez_compressed(os.path.join(outdir, f'g{gi:02d}_e{ep:03d}_{out["result"]}.npz'), img=np.array(lg['img'], np.uint8), q=np.array(lg['q'], np.float32),
                                    grip=np.array(lg['grip'], np.float32), tcp=np.array(lg['tcp'], np.float32), obj=np.array(lg['obj'], np.float32), act=np.array(lg['act'], np.float32),
                                    meta=json.dumps(out, default=float))
        json.dump(rows, open(os.path.join(outdir, 'stats.json'), 'w'), default=float, indent=1); log(f'  group {gi + 1}/{len(groups)}: {len(rows)} episodes')
    errs = [r['loc_err_m'] for r in rows if 'loc_err_m' in r]; stats = dict(collections.Counter(r['result'] for r in rows))
    if errs: stats['camera_localisation_error_m'] = dict(median=round(float(np.median(errs)), 4), max=round(float(max(errs)), 4), n=len(errs))
    return stats
