"""Kinematics and planning for the kinematic (suction wand) diagnostic: IK, collision, RRT-Connect, straight-line moves."""
import numpy as np, mujoco

class Kin:
    def __init__(self, m, d, info, rng):
        self.m, self.d, self.info, self.rng = m, d, info, rng
        self.site = info['tip_site']; self.lo, self.hi = m.jnt_range[:6, 0].copy(), m.jnt_range[:6, 1].copy()
    def ik(self, p, a, q0, iters=150, lam=0.05):
        """Damped least squares: tool tip at p, tool axis along a (spin about the axis stays free)."""
        m, d = self.m, self.d; jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); q = np.array(q0, float)
        for _ in range(iters):
            d.qpos[:6] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            ax = d.site_xmat[self.site].reshape(3, 3)[:, 2]; ep = p - d.site_xpos[self.site]; er = np.cross(ax, a)
            if np.linalg.norm(ep) < 2e-3 and np.linalg.norm(er) < 1.5e-2 and ax @ a > 0: return q, True
            if ax @ a < -0.2: er = er + 0.5 * np.cross(ax, [0.3, 0.2, 1.0])
            mujoco.mj_jacSite(m, d, jp, jr, self.site); J = np.vstack([jp[:, :6], jr[:, :6]]); e = np.concatenate([ep, er])
            e = e * min(1.0, 0.35 / max(np.linalg.norm(e), 1e-9))
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e), self.lo, self.hi)
        return q, False
    def free(self, q):
        m, d = self.m, self.d; d.qpos[:6] = q; mujoco.mj_fwdPosition(m, d); env = self.info['env']
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -0.004: continue
            if c.geom1 in env and c.geom2 in env: continue
            return False
        return True
    def solve(self, p, a, seeds=(), tries=5):
        """IK from several seeds; returns (q, 'ok') or (None, 'out_of_reach' | 'collision')."""
        hit = False
        for q0 in list(seeds) + [self.rng.uniform(-np.pi, np.pi, 6) for _ in range(tries)]:
            q, ok = self.ik(np.asarray(p, float), np.asarray(a, float), q0)
            if not ok: continue
            q = np.clip((q + np.pi) % (2 * np.pi) - np.pi, self.lo, self.hi)
            if self.free(q): return q, 'ok'
            hit = True
        return None, ('collision' if hit else 'out_of_reach')
    def near(self, q, ref):
        q = np.array(q, float)
        for j in range(6):
            for k in (-2 * np.pi, 2 * np.pi):
                c = q[j] + k
                if self.lo[j] <= c <= self.hi[j] and abs(c - ref[j]) < abs(q[j] - ref[j]): q[j] = c
        return q
    def seg_free(self, a, b):
        n = int(np.ceil(np.max(np.abs(b - a)) / 0.05)) + 1
        return all(self.free(a + (b - a) * s) for s in np.linspace(0, 1, n))
    def rrt(self, a, b, iters=4000, step=0.3):
        """RRT-Connect in joint space, then shortcutting."""
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
    def cart(self, q0, p0, p1, axis, n=8):
        """Straight-line tool motion p0 -> p1 by chained IK; None if it leaves reach or collides."""
        out = [np.array(q0)]; q = q0
        for s in np.linspace(0, 1, n + 1)[1:]:
            q2, ok = self.ik(np.array(p0) + (np.array(p1) - np.array(p0)) * s, np.array(axis, float), q, iters=80)
            if not ok or not self.free(q2): return None
            out.append(q2); q = q2
        return out

def move_time(path, vmax):
    """One blended move through the waypoints with a smoothstep profile (peak speed = 1.5 x mean)."""
    tot = np.sum(np.abs(np.diff(np.array(path), axis=0)), axis=0); return float(1.5 * np.max(tot / vmax))

def along(path, s):
    P = np.array(path); seg = np.linalg.norm(np.diff(P, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-9: return P[0]
    x = s * cum[-1]; i = min(np.searchsorted(cum, x, side='right') - 1, len(seg) - 1); u = (x - cum[i]) / max(seg[i], 1e-9)
    return P[i] + (P[i + 1] - P[i]) * u
