"""Compile a resolved scene + a robot into a MuJoCo model."""
import os
import numpy as np, mujoco
from . import assets
from .robots import ROBOTS, GRIPPER_XML

def build(robot, resolved, tool=None, objects=(), camera=None):
    """tool: {"type": "suction_wand", "length": 0.38} (kinematic only) or {"type": "gripper_2f85"} (physics grasping).
    objects: names of scanned objects to add as free bodies (parked off-scene until placed)."""
    tool = tool or dict(type='suction_wand', length=0.38); R = ROBOTS[robot]
    xml = os.path.join(assets.MENAGERIE, R['xml'])
    if not os.path.exists(xml): raise FileNotFoundError(f'{xml} not found. Run `ari-sim setup` first.')
    spec = mujoco.MjSpec.from_file(xml); wb = spec.worldbody
    fl = wb.add_geom(); fl.name = 'env_floor'; fl.type = mujoco.mjtGeom.mjGEOM_PLANE; fl.size = [8, 8, 0.1]; fl.rgba = [0.80, 0.80, 0.77, 1]
    l = wb.add_light(); l.pos = [1.5, -3.0, 5]; l.dir = [-0.25, 0.5, -1]; l.diffuse = [0.55, 0.55, 0.55]; l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    l2 = wb.add_light(); l2.pos = [-2.5, -2, 4]; l2.dir = [0.5, 0.4, -1]; l2.diffuse = [0.25, 0.25, 0.25]; l2.castshadow = False; l2.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    tex = spec.add_texture(); tex.name = 'sky'; tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX; tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    tex.rgb1 = [0.95, 0.945, 0.93]; tex.rgb2 = [0.95, 0.945, 0.93]; tex.width = 64; tex.height = 384
    spec.visual.global_.offwidth = 1920; spec.visual.global_.offheight = 1080
    spec.visual.headlight.ambient = [0.42, 0.42, 0.42]; spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]; spec.visual.headlight.specular = [0, 0, 0]
    for g in resolved['geoms']:
        e = wb.add_geom(); e.name = 'env_' + g['name']; e.pos = g['pos']; e.rgba = g['rgba']
        if g['shape'] == 'box': e.type = mujoco.mjtGeom.mjGEOM_BOX; e.size = g['size'] / 2
        else: e.type = mujoco.mjtGeom.mjGEOM_CYLINDER; e.size = [g['size'][0], g['size'][1] / 2, 0]
        a = np.radians(g['yaw']) / 2; e.quat = [np.cos(a), 0, 0, np.sin(a)]
        e.contype = 1 if g['collide'] else 0; e.conaffinity = 1 if g['collide'] else 0
    ped = wb.add_body(); ped.name = 'pedestal'; ped.mocap = True
    pg = ped.add_geom(); pg.name = 'ped_geom'; pg.type = mujoco.mjtGeom.mjGEOM_BOX; pg.size = [0.11, 0.11, 0.5]; pg.rgba = [0.2, 0.2, 0.22, 1]; pg.contype = 0; pg.conaffinity = 0
    it = wb.add_body(); it.name = 'item'; it.mocap = True; it.pos = [0, 0, -1]
    ig = it.add_geom(); ig.type = mujoco.mjtGeom.mjGEOM_BOX; ig.size = [0.06, 0.045, 0.03]; ig.rgba = [0.75, 0.55, 0.35, 1]; ig.contype = 0; ig.conaffinity = 0
    tipb = spec.body(R['tip']); st = spec.site('attachment_site'); gripper = tool['type'] == 'gripper_2f85'
    if gripper:
        for k in list(spec.keys): spec.delete(k)
        gs = mujoco.MjSpec.from_file(os.path.join(assets.MENAGERIE, GRIPPER_XML))
        for k in list(gs.keys): gs.delete(k)
        spec.attach(gs, prefix='g_', site=st)
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; spec.option.impratio = 10
        for i, name in enumerate(objects):
            osp = mujoco.MjSpec.from_file(os.path.join(assets.GSO, 'models', name, 'model.xml'))
            fr = wb.add_frame(); fr.pos = [3.0 + 0.5 * i, -3.0, 0.3]
            nb = fr.attach_body(osp.worldbody.first_body(), f'o{i}_', ''); nb.add_freejoint()
    else:
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, np.array(st.quat)); Rm = Rm.reshape(3, 3); Lw = float(tool.get('length', 0.38))
        p0 = np.array(st.pos); p1 = p0 + Rm[:, 2] * Lw
        tg = tipb.add_geom(); tg.name = 'tool'; tg.type = mujoco.mjtGeom.mjGEOM_CAPSULE; tg.fromto = list(p0) + list(p1 - Rm[:, 2] * 0.02)
        tg.size = [0.018, 0, 0]; tg.rgba = [0.18, 0.33, 1.0, 1]
        ts = tipb.add_site(); ts.name = 'tip'; ts.pos = p1; ts.quat = st.quat; ts.size = [0.012, 0.012, 0.012]; ts.rgba = [0.18, 0.33, 1, 1]
    if camera:                      # fixed observation camera: {"pos": [..], "target": [..], "fovy": 60}
        cp = np.array(camera['pos'], float); f = np.array(camera['target'], float) - cp; f /= np.linalg.norm(f)
        rt = np.cross(f, [0, 0, 1]); rt /= np.linalg.norm(rt); up = np.cross(rt, f)
        cm = wb.add_camera(); cm.name = 'station_cam'; cm.pos = cp; cm.fovy = camera.get('fovy', 60)
        cq = np.zeros(4); mujoco.mju_mat2Quat(cq, np.column_stack([rt, up, -f]).flatten()); cm.quat = cq
    m = spec.compile(); d = mujoco.MjData(m); bid = m.body(R['base']).id
    info = dict(robot=robot, base_id=bid, tip_site=m.site('g_pinch' if gripper else 'tip').id, ped=m.body('pedestal').mocapid[0], ped_geom=m.geom('ped_geom').id,
                item=m.body('item').mocapid[0], base_pos0=m.body_pos[bid].copy(), env={i for i in range(m.ngeom) if m.geom(i).name.startswith('env_')},
                objects=[m.body(f'o{i}_model').id for i in range(len(objects))] if gripper else [], tool=tool)
    return m, d, info

def set_base(m, d, info, x, y, z):
    """Robot on a pedestal of height z at floor position (x, y)."""
    m.body_pos[info['base_id']] = info['base_pos0'] + np.array([x, y, z])
    d.mocap_pos[info['ped']] = [x, y, z / 2]; m.geom_size[info['ped_geom']] = [0.11, 0.11, max(z / 2, 0.01)]

def scene_view(resolved):
    """A camera that frames the whole workstation from the operator side."""
    P = np.array([g['pos'] for g in resolved['geoms']]); S = np.array([np.pad(g['size'], (0, 3 - len(g['size'])), constant_values=g['size'][-1]) for g in resolved['geoms']])
    lo = (P - S / 2).min(0); hi = (P + S / 2).max(0); c = (lo + hi) / 2; span = float(np.linalg.norm(hi - lo))
    return dict(lookat=(c[0], c[1], max(c[2], 1.0)), dist=max(1.2 * span, 3.0))

def render(m, d, view, w=1280, h=720, az=112, el=-14):
    r = mujoco.Renderer(m, h, w); cam = mujoco.MjvCamera(); cam.lookat[:] = view['lookat']; cam.distance = view['dist']; cam.azimuth = az; cam.elevation = el
    mujoco.mj_forward(m, d); r.update_scene(d, cam); img = r.render().copy(); r.close(); return img
