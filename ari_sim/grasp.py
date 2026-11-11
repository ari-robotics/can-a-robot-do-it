"""EXPERIMENTAL — physics grasping in the reconstructed workstation.

UR10e + Robotiq 2F-85, Google Scanned Objects dropped at the pick targets with random pose. A scripted expert
(it knows the object pose) does a side grasp and carries the object to the drop. Each episode is recorded
(camera image, joints, gripper, action) so successes can serve as demonstrations for imitation learning.

Limits of this version: horizontal-approach pick targets only (shelf cells); one fixed grasp strategy.
"""
import os, json, warnings, collections
import numpy as np, mujoco
warnings.filterwarnings('ignore')
from . import spec as S, assets
from .build import build, set_base
from .robots import ROBOTS

ROBOT = 'UR10e'; VMAX = ROBOTS[ROBOT]['vmax']
R_DOWN = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float).T      # approach straight down, fingers close along world X

def _frames(a):
    """Gripper orientations for a horizontal approach a: z = a, y = finger closing axis (horizontal), x = y x z."""
    a = np.asarray(a, float); f = np.cross([0, 0, 1.0], a); f /= np.linalg.norm(f); out = []
    for y in (f, -f): out.append(np.column_stack([np.cross(y, a), y, a]))
    return out, f

class Env:
    def __init__(self, sc, base, names, values=None, seed=0, img=(160, 120)):
        self.sc = sc; self.R = S.resolve(sc, values); self.names = list(names); self.rng = np.random.default_rng(seed)
        cam = sc.get('robot', {}).get('observation_camera') or dict(pos=[-0.85, -1.25, 2.25], target=[0, 0.25, 1.15])
        self.m, self.d, self.info = build(ROBOT, self.R, dict(type='gripper_2f85'), objects=self.names, camera=cam)
        m = self.m; set_base(m, self.d, self.info, *base); self.dp = mujoco.MjData(m)
        self.site = self.info['tip_site']; self.lo, self.hi = m.jnt_range[:6, 0].copy(), m.jnt_range[:6, 1].copy()
        bn = [m.body(i).name for i in range(m.nbody)]; self.obj = self.info['objects']; objset = set(self.obj)
        robot_b = {i for i in range(1, m.nbody) if i not in objset and bn[i] not in ('pedestal', 'item')}
        grip_b = {i for i in range(m.nbody) if bn[i].startswith('g_')}
        self.rgeom = np.array([m.geom_bodyid[g] in robot_b and (m.geom_contype[g] > 0 or m.geom_conaffinity[g] > 0) for g in range(m.ngeom)])
        self.ggeom = np.array([m.geom_bodyid[g] in grip_b for g in range(m.ngeom)])
        self.qadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in self.obj]; self.vadr = [m.jnt_dofadr[m.body_jntadr[b]] for b in self.obj]
        self.verts = []
        for b in self.obj:
            g0 = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH][0]
            mid = m.geom_dataid[g0]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[g0]); v = v @ Rg.reshape(3, 3).T + m.geom_pos[g0]
            self.verts.append(v[:: max(1, len(v) // 3000)].copy())
            mass = float(np.clip(m.body_mass[b], 0.03, 0.35)); k = mass / m.body_mass[b]; m.body_mass[b] *= k; m.body_inertia[b] *= k
        # the real 2F-85 grips far harder than the stock model's 5 N limit; silicone pads
        ga = m.actuator('g_fingers_actuator').id; m.actuator_forcerange[ga] = [-12, 12]
        for g in range(m.ngeom):
            if 'pad' in bn[m.geom_bodyid[g]] and m.geom_contype[g] > 0: m.geom_friction[g, 0] = 1.5
        m.opt.timestep = 0.001; m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        mujoco.mj_setConst(m, self.d); mujoco.mj_resetData(m, self.d); self.park = self.d.qpos.copy(); self.dt = m.opt.timestep
        self.targets = [t for t in self.R['picks'] if abs(t['approach'][2]) < 0.3]
        if not self.targets: raise S.SceneError('grasp: no pick target with a horizontal approach (shelf cells) in this scene')
        pl = self.R['places'][0]; self.place = pl; dz = np.array([0, 0, 0.04])
        self.qhome = self._solve(pl['pre'] + np.array([0, 0, 0.10]), [R_DOWN], [], tries=40)
        self.qdrop = self._solve(pl['drop'] + dz, [R_DOWN], [self.qhome] if self.qhome is not None else [], tries=10)
        if self.qhome is None or self.qdrop is None: raise S.SceneError('grasp: the drop target is not reachable from this base position')
        self.qdrop = self.near(self.qdrop, self.qhome)
        self.ren = mujoco.Renderer(m, img[1], img[0]); self.camid = m.camera('station_cam').id
    # ---------- kinematics / planning on a scratch MjData
    def sync(self): self.dp.qpos[:] = self.d.qpos; self.dp.qpos[6:14] = 0
    def ik(self, p, Rd, q0, iters=120, lam=0.05):
        m, d = self.m, self.dp; jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); q = np.array(q0, float)
        for _ in range(iters):
            d.qpos[:6] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            R = d.site_xmat[self.site].reshape(3, 3); ep = p - d.site_xpos[self.site]
            er = 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 1.5e-3 and np.linalg.norm(er) < 1e-2 and np.trace(R.T @ Rd) > 2.5: return q, True
            mujoco.mj_jacSite(m, d, jp, jr, self.site); J = np.vstack([jp[:, :6], jr[:, :6]]); e = np.concatenate([ep, er])
            e *= min(1.0, 0.35 / max(np.linalg.norm(e), 1e-9))
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e), self.lo, self.hi)
        return q, False
    def free(self, q, ignore=-1):
        m, d = self.m, self.dp; d.qpos[:6] = q; mujoco.mj_fwdPosition(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -0.003: continue
            r1, r2 = self.rgeom[c.geom1], self.rgeom[c.geom2]
            if not r1 and not r2: continue
            if r1 and r2:
                if self.ggeom[c.geom1] and self.ggeom[c.geom2]: continue
                return False
            other = c.geom2 if r1 else c.geom1
            if ignore >= 0 and m.geom_bodyid[other] == ignore: continue
            return False
        return True
    def near(self, q, ref):
        q = np.array(q, float)
        for j in range(6):
            for k in (-2 * np.pi, 2 * np.pi):
                c = q[j] + k
                if self.lo[j] <= c <= self.hi[j] and abs(c - ref[j]) < abs(q[j] - ref[j]): q[j] = c
        return q
    def _solve(self, p, Rs, seeds, tries=8, ignore=-1):
        self.sync()
        for q0 in list(seeds) + [self.rng.uniform(-np.pi, np.pi, 6) for _ in range(tries)]:
            for Rd in Rs:
                q, ok = self.ik(np.asarray(p, float), Rd, q0)
                if not ok: continue
                q = np.clip((q + np.pi) % (2 * np.pi) - np.pi, self.lo, self.hi)
                if self.free(q, ignore): self._R = Rd; return q
        return None
    def seg_free(self, a, b, ignore=-1):
        n = int(np.ceil(np.max(np.abs(b - a)) / 0.05)) + 1
        return all(self.free(a + (b - a) * s, ignore) for s in np.linspace(0, 1, n))
    def rrt(self, a, b, iters=3000, step=0.3):
        rng = self.rng
        if self.seg_free(a, b): return [a, b]
        Ta = [(a, -1)]; Tb = [(b, -1)]; blo = np.maximum(np.minimum(a, b) - 1.2, self.lo); bhi = np.minimum(np.maximum(a, b) + 1.2, self.hi)
        def extend(T, q):
            i = min(range(len(T)), key=lambda k: np.linalg.norm(T[k][0] - q)); qn = T[i][0]; v = q - qn; L = np.linalg.norm(v)
            new = q if L <= step else qn + v / L * step
            if self.seg_free(qn, new): T.append((new, i)); return new, L <= step
            return None, False
        def trace(T, i):
            p = []
            while i != -1: p.append(T[i][0]); i = T[i][1]
            return p
        fwd = True
        for _ in range(iters):
            new, _r = extend(Ta, rng.uniform(blo, bhi))
            if new is not None:
                while True:
                    nb, reached = extend(Tb, new)
                    if nb is None: break
                    if reached:
                        path = trace(Ta, len(Ta) - 1)[::-1] + trace(Tb, len(Tb) - 1)[1:]
                        if not fwd: path = path[::-1]
                        for _ in range(80):
                            if len(path) < 3: break
                            i, j = sorted(rng.choice(len(path), 2, replace=False))
                            if j - i > 1 and self.seg_free(path[i], path[j]): path = path[:i + 1] + path[j:]
                        return path
            Ta, Tb = Tb, Ta; fwd = not fwd
        return None
    def cart(self, q0, p0, p1, Rd, n=8, ignore=-1):
        out = [np.array(q0)]; q = q0
        for s in np.linspace(0, 1, n + 1)[1:]:
            q2, ok = self.ik(np.array(p0) + (np.array(p1) - np.array(p0)) * s, Rd, q, iters=80)
            if not ok or not self.free(q2, ignore): return None
            out.append(q2); q = q2
        return out
    # ---------- objects
    def points(self, i):
        b = self.obj[i]; return self.verts[i] @ self.d.xmat[b].reshape(3, 3).T + self.d.xpos[b]
    def put(self, i, xyz, yaw):
        a = self.qadr[i]; self.d.qpos[a:a + 3] = xyz; self.d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]; self.d.qvel[self.vadr[i]:self.vadr[i] + 6] = 0
    def reset(self, k=4):
        m, d, rng = self.m, self.d, self.rng
        mujoco.mj_resetData(m, d); d.qpos[:] = self.park; d.qpos[:6] = self.qhome; d.ctrl[:6] = self.qhome; d.ctrl[6] = 0
        idx = rng.choice(len(self.targets), size=min(k, len(self.obj), len(self.targets)), replace=False)
        chosen = rng.choice(len(self.obj), size=len(idx), replace=False); self.placed = {}
        for oi, ti in zip(chosen, idx):
            t = self.targets[ti]; a = t['approach']; _, f = _frames(a); v = self.verts[oi]; yaw0 = np.arctan2(a[1], a[0]) - np.pi / 2
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
        self.log['img'].append(self.ren.render().copy()); self.log['q'].append(d.qpos[:6].copy()); self.log['grip'].append(float(d.qpos[6]))
        self.log['tcp'].append(d.site_xpos[self.site].copy()); self.log['obj'].append(np.concatenate([d.xpos[self.obj[tgt]], d.xquat[self.obj[tgt]]])); self.log['act'].append(d.ctrl[:7].copy())
    def _step(self, tgt):
        mujoco.mj_step(self.m, self.d); self.k += 1
        if not np.isfinite(self.d.qacc).all() or np.abs(self.d.qvel).max() > 500: raise FloatingPointError('unstable')
        if self.k % 200 == 0: self._rec(tgt)                       # 5 Hz
        if self.video and self.k % self.vevery == 0: self.video(self)
    def follow(self, path, tgt, grip, speed=0.6):
        Pm = np.array(path); tot = np.sum(np.abs(np.diff(Pm, axis=0)), axis=0); T = max(1.5 * np.max(tot / VMAX) / speed, 0.25)
        seg = np.linalg.norm(np.diff(Pm, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)]); n = int(T / self.dt)
        for k in range(1, n + 1):
            s = k / n; s = s * s * (3 - 2 * s); x = s * cum[-1]
            i = min(np.searchsorted(cum, x, side='right') - 1, len(seg) - 1); u = (x - cum[i]) / max(seg[i], 1e-9)
            self.d.ctrl[:6] = Pm[i] + (Pm[i + 1] - Pm[i]) * u; self.d.ctrl[6] = grip; self._step(tgt)
        return T
    def hold(self, t, tgt, grip):
        self.d.ctrl[6] = grip
        for _ in range(int(t / self.dt)): self._step(tgt)
    def episode(self, record=True, video=None, vevery=33):
        self._out = {}
        try: return self._episode(record, video, vevery)
        except FloatingPointError: return dict(self._out, result='sim_unstable'), None
    def _episode(self, record, video, vevery):
        self.log = dict(img=[], q=[], grip=[], tcp=[], obj=[], act=[]) if record else None; self.k = 0; self.video = video; self.vevery = vevery
        placed = self.reset(); d = self.d
        tgt = int(self.rng.choice(list(placed))); t = self.targets[placed[tgt]]; b = self.obj[tgt]; a = t['approach']; Rs, f = _frames(a)
        pts = self.points(tgt) - t['face']; u = pts @ f; v = pts @ a; z = pts[:, 2] + t['face'][2]
        out = self._out = dict(object=self.names[tgt], target=t['name'], width=float(u.max() - u.min()), height=float(z.max() - z.min()))
        if v.min() < -0.03 or d.xpos[b][2] < t['support_z'] - 0.03: return dict(out, result='fell_at_reset'), None
        if u.max() - u.min() > 0.074: return dict(out, result='too_wide'), None
        zg = float(np.clip((z.min() + z.max()) / 2, t['support_z'] + 0.05, t['top_z'] - 0.05)); ug = float((u.min() + u.max()) / 2)
        vg = float(np.clip((v.min() + v.max()) / 2, v.min() + 0.02, v.min() + 0.065))
        P = lambda uu, vv, zz: np.array([t['face'][0], t['face'][1], 0]) + f * uu + a * vv + np.array([0, 0, zz])
        pre, grasp = P(ug, -0.13, zg), P(ug, vg, zg)
        qpre = self._solve(pre, Rs, [self.qhome], tries=10)
        if qpre is None: return dict(out, result='unreachable'), None
        Rd = self._R; qpre = self.near(qpre, self.qhome)
        ins = self.cart(qpre, pre, grasp, Rd, ignore=b); path = self.rrt(self.qhome, qpre) if ins is not None else None
        if ins is None or path is None: return dict(out, result='no_path'), None
        up, pre_up = grasp + [0, 0, 0.012], pre + [0, 0, 0.012]
        lift = self.cart(ins[-1], grasp, up, Rd, n=2, ignore=b); ret = self.cart(lift[-1], up, pre_up, Rd, ignore=b) if lift is not None else None
        if ret is None: return dict(out, result='no_path'), None
        T = self.follow(path, tgt, 0, speed=0.5) + self.follow(ins, tgt, 0, speed=0.35); self.hold(0.6, tgt, 255)
        T += self.follow(lift, tgt, 255, speed=0.3) + self.follow(ret, tgt, 255, speed=0.35)
        held = np.linalg.norm(d.xpos[b] - d.site_xpos[self.site]) < 0.12 and ((self.points(tgt) - t['face']) @ a).min() < 0
        back = [ret[-1], path[-1]] + path[::-1][1:]
        if not held: self.follow(back, tgt, 0); return dict(out, result='grasp_failed', time=T + 0.6), self.log
        T += self.follow(back, tgt, 255, speed=0.35) + self.follow([self.qhome, self.qdrop], tgt, 255, speed=0.35)
        self.hold(0.5, tgt, 0); self.follow([self.qdrop, self.qhome], tgt, 0, speed=0.5); self.hold(0.3, tgt, 0)
        p = d.xpos[b]; bx = self.place.get('box')
        if bx: ok = abs(p[0] - bx['center'][0]) < bx['half'][0] + 0.01 and abs(p[1] - bx['center'][1]) < bx['half'][1] + 0.01 and bx['rim'] - bx['depth'] - 0.04 < p[2] < bx['rim'] + 0.03
        else: ok = np.linalg.norm(p[:2] - self.place['drop'][:2]) < 0.15
        return dict(out, result='success' if ok else 'dropped', time=T + 1.1, final=[float(x) for x in p]), self.log

def run_batch(sc, base, episodes, outdir, seed=0, per_scene=6, log=print):
    """Episodes over all graspable objects, 6 objects loaded at a time. Saves one .npz per attempted episode + stats.json."""
    names = assets.graspable_objects(); os.makedirs(outdir, exist_ok=True)
    # near-equal groups, episodes in proportion to group size: every object gets about the same share
    groups = [list(g) for g in np.array_split(names, int(np.ceil(len(names) / per_scene)))]
    cum = np.round(episodes * np.cumsum([0] + [len(g) for g in groups]) / len(names)).astype(int); rows = []
    for gi, g in enumerate(groups):
        env = Env(sc, base, g, seed=seed + gi)
        for ep in range(cum[gi + 1] - cum[gi]):
            out, lg = env.episode(record=True); out.update(group=gi, ep=ep); rows.append(out)
            if lg is not None and len(lg['q']) > 5:
                np.savez_compressed(os.path.join(outdir, f'g{gi:02d}_e{ep:03d}_{out["result"]}.npz'), img=np.array(lg['img'], np.uint8), q=np.array(lg['q'], np.float32),
                                    grip=np.array(lg['grip'], np.float32), tcp=np.array(lg['tcp'], np.float32), obj=np.array(lg['obj'], np.float32), act=np.array(lg['act'], np.float32),
                                    meta=json.dumps(out, default=float))
        json.dump(rows, open(os.path.join(outdir, 'stats.json'), 'w'), default=float, indent=1); log(f'  group {gi + 1}/{len(groups)}: {len(rows)} episodes')
    return dict(collections.Counter(r['result'] for r in rows))
