"""One candidate robot cell: an arm with its tool, standing somewhere in the rebuilt station.

All kinematics and planning are Google's `motion-planning` skill (motion_planner.py): multi-start IK with collision
filtering, bidirectional RRT for free-space moves, straight Cartesian strokes for insertions, time-parameterised
trajectories. This file only translates: the station becomes a Scene3D of primitives in the robot's frame, a pick
target becomes a pose, and the planner's answers become 'ok' | 'out_of_reach' | 'collision' | 'path_blocked'.
"""
import numpy as np
from . import google, build
from .robots import ROBOTS

DT = 0.04                       # control period of the planned trajectories (s)
TCP_SPEED = 1.0                 # m/s cap on the tool in free-space moves

def frame(a, spin=0.0):
    """Rotation with z along the approach `a`; `spin` turns the tool about it (matters for fingers, not for suction)."""
    a = np.asarray(a, float); a = a / np.linalg.norm(a); up = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(up, a); x /= np.linalg.norm(x); y = np.cross(a, x); c, s = np.cos(spin), np.sin(spin)
    return np.column_stack([c * x + s * y, -s * x + c * y, a])

class Cell:
    def __init__(self, robot, tool):
        self.mp = google.load('motion-planning'); self.robot = robot; self.xml, self.meta = build.robot_mjcf(robot, tool)
        self.tool = self.meta['tool']; self.pl = self.mp.RobotArmPlanner(self.xml, joint_names=self.meta['joints'], ee_site_name='tip')
        self.vmax = float(np.min(ROBOTS[robot]['vmax'])); self.spins = (0.0, np.pi / 2) if self.tool.get('kind') == 'fingers' else (0.0,)
        self.base = np.zeros(3); self.scene = self.mp.Scene3D(); self.home = np.clip(np.zeros(self.pl.dof), self.pl.q_min + 0.02, self.pl.q_max - 0.02)

    def place(self, resolved, x, y, z):
        """Stand the arm at floor position (x, y) on a pedestal of height z: the station is re-expressed in the arm's frame."""
        self.base = np.array([x, y, z], float); off = self.base; sc = self.mp.Scene3D()
        sc.add_aligned_box('floor', [0, 0, -z - 0.05], [6, 6, 0.05], group='floor')
        if z > 0.06: sc.add_aligned_box('pedestal', [0, 0, -z / 2 - 0.04], [0.10, 0.10, z / 2 - 0.04], group='pedestal')
        for g in resolved['geoms']:
            if not g['collide'] or g['shape'] == 'mesh': continue
            c = g['pos'] - off
            if g['shape'] == 'box':
                if abs(g['yaw']) < 1e-6: sc.add_aligned_box(g['name'], c, g['size'] / 2, group='station')
                else:
                    a = np.radians(g['yaw']); R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                    sc.add_oriented_box(g['name'], c, g['size'] / 2, R, group='station')
            else: sc.add_vertical_cylinder(g['name'], c, float(g['size'][0]), float(g['size'][1]), group='station')
        self.scene = sc; return self

    def reach(self, p, a, seed=None, restarts=6):
        """Joint configuration that puts the tool tip at world point p, approaching along a. Returns (q | None, status)."""
        p = np.asarray(p, float) - self.base; hit = False
        for spin in self.spins:
            r = self.pl.solve_ik(p, frame(a, spin), q_seed=seed, scene=self.scene, num_restarts=restarts)
            if r['success']: return np.asarray(r['q'], float), 'ok'
            hit = hit or r.get('reason') == 'ik_converged_but_in_collision'
        return None, ('collision' if hit else 'out_of_reach')

    def stroke(self, q, p, a=None):
        """Straight-line tool move from configuration q to world point p (insertion, retreat, descent into a container)."""
        r = self.pl.plan_cartesian_stroke(q, np.asarray(p, float) - self.base, None, scene=self.scene, clearance_tol_m=0.002,
                                          max_joint_vel_rad_s=self.vmax, control_dt_s=DT)
        return (np.asarray(r['q_trajectory'], float), float(r['duration_s'])) if r['solved'] else (None, r.get('reason', 'failed'))

    def move(self, q0, q1, seed=1):
        """Collision-free free-space move between two configurations, timed at the arm's joint speed limit."""
        r = self.pl.plan_arm(self.scene, q0, q1, max_joint_vel_rad_s=self.vmax, control_dt_s=DT, seed=seed, max_cartesian_vel_m_s=TCP_SPEED, safety_margin_m=0.01)
        return (np.asarray(r['q_trajectory'], float), float(r['duration_s'])) if r['solved'] else (None, r.get('reason', 'failed'))

    def near(self, q, ref):
        """The same pose with every joint turned to the revolution closest to `ref` (arms whose joints span more than one turn)."""
        q = np.array(q, float)
        for j in range(len(q)):
            for k in (-2 * np.pi, 2 * np.pi):
                c = q[j] + k
                if self.pl.q_min[j] <= c <= self.pl.q_max[j] and abs(c - ref[j]) < abs(q[j] - ref[j]): q[j] = c
        return q

    def reach_near(self, p, a, ref, seeds=()):
        """Solutions for the same target from several seeds, closest to `ref` first: short moves, same arm posture."""
        out = []
        for s in [ref] + [x for x in seeds if x is not None]:
            q, st = self.reach(p, a, s, restarts=6)
            if q is not None and self.clearance(self.near(q, ref))[0]: out.append(self.near(q, ref))
        out.sort(key=lambda q: float(np.abs(q - ref).max())); return out

    def clearance(self, q):
        c = self.pl.check_arm_config(self.scene, q); return c['valid'], c.get('min_clearance_m')

    def tip(self, q):
        p, R = self.pl.fk(q); return np.asarray(p) + self.base, np.asarray(R)
