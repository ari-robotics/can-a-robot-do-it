"""MuJoCo models, written with MjSpec to Google's asset standards (rules/mujoco_asset_standards.md of the robotics plugin):
visual geoms in group 2 with no mass, collision geoms in group 3 shown semi-transparent, hollow containers as walls,
one semantic root body with no transform, the `discrete` integrator in scenes.

Three things are built here:
  robot_mjcf    an arm with its tool, as a standalone MJCF: what Google's motion planner loads
  station       the rebuilt workstation as a reusable asset (model.xml): what the audit, the showcase shot and Studio load
  scene         arm + station + parts in one model, for rendering and for physics
"""
import os, re, json, hashlib
import numpy as np, mujoco
from . import assets
from .robots import ROBOTS, GRIPPER_XML

G = mujoco.mjtGeom
COLLIDER = [0.20, 0.75, 0.35, 0.50]; WALL = [0.25, 0.55, 0.90, 0.45]          # the standard's colours for group 3

def _slug(s): return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_') or 'station'

def _absolute(spec, xml):
    """Mesh and texture files as absolute paths, so the spec can be attached elsewhere and written anywhere."""
    d = os.path.dirname(os.path.abspath(xml))
    for m in spec.meshes:
        if m.file and not os.path.isabs(m.file): m.file = os.path.join(d, spec.meshdir, m.file)
    for t in spec.textures:
        if t.file and not os.path.isabs(t.file): t.file = os.path.join(d, spec.texturedir, t.file)
    spec.meshdir = ''; spec.texturedir = ''
    for k in list(spec.keys): spec.delete(k)
    return spec

def robot_spec(robot, tool):
    """(MjSpec of the arm with its tool mounted, meta). The tool tip is the site `tip`: z along the approach."""
    R = ROBOTS[robot]; kind = tool['type']
    assets.fetch_robots([robot], gripper=kind == 'gripper_2f85')
    if 'simready' in R:
        from . import simready; spec, base, tipb, _ = simready.robot(R['simready'])
        for k in list(spec.keys): spec.delete(k)
    else:
        xml = os.path.join(assets.MENAGERIE, R['xml']); spec = _absolute(mujoco.MjSpec.from_file(xml), xml); base, tipb = R['base'], R['tip']
    arm = [j.name for j in spec.joints if j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
    st = spec.site(R.get('site', 'attachment_site')); tool = dict(tool)
    Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, np.array(st.quat)); Rm = Rm.reshape(3, 3); p0 = np.array(st.pos)
    if kind == 'gripper_2f85':
        gx = os.path.join(assets.MENAGERIE, GRIPPER_XML); gs = _absolute(mujoco.MjSpec.from_file(gx), gx)
        spec.attach(gs, prefix='g_', site=st); spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; spec.option.impratio = 10
        pinch = spec.site('g_pinch'); tip = spec.body(pinch.parent.name).add_site(); tip.pos = pinch.pos; tip.quat = pinch.quat
    else:
        if tool.get('source'):          # a real end effector from the libraries, as its maker modelled it
            from . import simready; ts, measured = simready.tool(tool['source']); tool.setdefault('length', measured)
            for k in list(ts.keys): ts.delete(k)
            spec.attach(ts, prefix='t_', site=st)
        L = float(tool.get('length', 0.38)); p1 = p0 + Rm[:, 2] * L; body = spec.body(tipb)
        if not tool.get('source'):
            g = body.add_geom(); g.name = 'tool'; g.type = G.mjGEOM_CAPSULE; g.fromto = list(p0) + list(p1 - Rm[:, 2] * 0.02); g.size = [0.018, 0, 0]; g.rgba = [0.18, 0.33, 1.0, 1]
        tip = body.add_site(); tip.pos = p1; tip.quat = st.quat
    tip.name = 'tip'; tip.size = [0.012, 0.012, 0.012]; tip.rgba = [0.18, 0.33, 1, 1]
    return spec, dict(robot=robot, base=base, joints=arm, tool=tool)

def robot_mjcf(robot, tool):
    """The arm and its tool as an MJCF file in the asset cache. Returns (path, meta)."""
    key = hashlib.sha1(json.dumps([robot, ROBOTS[robot].get('xml') or ROBOTS[robot].get('simready'), {k: v for k, v in tool.items() if k != 'why'}], sort_keys=True, default=str).encode()).hexdigest()[:12]
    d = os.path.join(assets.HOME, 'cells'); path = os.path.join(d, f'{_slug(robot)}_{key}.xml'); mpath = path[:-4] + '.json'
    if os.path.exists(path) and os.path.exists(mpath): return path, json.load(open(mpath))
    spec, meta = robot_spec(robot, tool); m = spec.compile(); os.makedirs(d, exist_ok=True)
    meta['base_pos'] = [float(v) for v in m.body(meta['base']).pos]
    open(path, 'w').write(spec.to_xml()); json.dump(meta, open(mpath, 'w')); return path, meta

def _defaults(spec, name):
    """The standard's two classes: `visual` (group 2, massless, no contact) and `collision` (group 3, semi-transparent)."""
    pre = f'{name}_' if spec.find_default('visual') or spec.find_default('collision') else ''     # menagerie arms use the same class names
    main = spec.add_default(name, spec.default)
    v = spec.add_default(pre + 'visual', main); v.geom.group = 2; v.geom.contype = 0; v.geom.conaffinity = 0; v.geom.density = 0
    c = spec.add_default(pre + 'collision', main); c.geom.group = 3; c.geom.contype = 1; c.geom.conaffinity = 1; c.geom.condim = 4
    c.geom.friction = [0.8, 0.02, 0.005]; c.geom.solref = [0.005, 1]; c.geom.rgba = COLLIDER
    return v, c

def _station_into(spec, body, resolved, name):
    v, c = _defaults(spec, name)
    for g in resolved['geoms']:
        a = np.radians(g['yaw']) / 2; quat = [np.cos(a), 0, 0, np.sin(a)]
        if g['rgba'][3] > 0:                            # alpha 0 marks a collision-only helper (walls around a library mesh)
            e = body.add_geom(v); e.name = g['name'] + '_visual'; e.pos = g['pos']; e.quat = quat; e.rgba = g['rgba']
            if g['shape'] == 'mesh':
                ms = spec.add_mesh(); ms.name = g['name'] + '_mesh'; ms.file = g['mesh']; ms.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                e.type = G.mjGEOM_MESH; e.meshname = ms.name
            elif g['shape'] == 'box': e.type = G.mjGEOM_BOX; e.size = g['size'] / 2
            else: e.type = G.mjGEOM_CYLINDER; e.size = [g['size'][0], g['size'][1] / 2, 0]
        if g['collide'] and g['shape'] != 'mesh':
            k = body.add_geom(c); k.name = g['name'] + '_col'; k.pos = g['pos']; k.quat = quat; k.mass = 1.0
            k.type = G.mjGEOM_BOX if g['shape'] == 'box' else G.mjGEOM_CYLINDER
            k.size = g['size'] / 2 if g['shape'] == 'box' else [g['size'][0], g['size'][1] / 2, 0]
            if re.search(r'_(x|y)-?1$|_side[LR]$|_div\d+$|_back$', g['name']): k.rgba = WALL
    for t in resolved['picks'][:400]:
        s = body.add_site(); s.name = t['name'] + '_pick_site'; s.group = 4; s.pos = t['pick']; s.size = [0.02, 0.02, 0.02]; s.rgba = [0.95, 0.25, 0.25, 0.35]
    for t in resolved['places']:
        s = body.add_site(); s.name = t['name'] + '_place_site'; s.group = 4; s.pos = t['drop']; s.size = [0.03, 0.03, 0.03]; s.rgba = [0.15, 0.80, 0.85, 0.30]

def station(resolved, name, outdir):
    """The rebuilt workstation as a self-contained asset: <outdir>/<name>/model.xml, no floor, no lights, no robot."""
    name = _slug(name); spec = mujoco.MjSpec(); spec.modelname = name; spec.compiler.autolimits = True
    body = spec.worldbody.add_body(); body.name = name; _station_into(spec, body, resolved, name)
    spec.compile(); d = os.path.join(outdir, name); os.makedirs(d, exist_ok=True); p = os.path.join(d, 'model.xml'); open(p, 'w').write(spec.to_xml()); return p

def scene(robot, resolved, tool, base=(0, 0, 0), objects=(), camera=None, name='station'):
    """Arm at `base` (x, y, pedestal height) + station + free parts, in one compiled model. Returns (model, data, info).
    objects: scanned objects (GSO names) or SimReady assets ("simready:<id>") added as free bodies, parked off-scene."""
    path, meta = robot_mjcf(robot, tool); spec = mujoco.MjSpec.from_file(path); wb = spec.worldbody; x, y, z = (float(v) for v in base)
    b = spec.body(meta['base']); b.pos = np.array(meta['base_pos']) + [x, y, z]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    fl = wb.add_geom(); fl.name = 'floor'; fl.type = G.mjGEOM_PLANE; fl.size = [8, 8, 0.1]; fl.rgba = [0.80, 0.80, 0.77, 1]
    l = wb.add_light(); l.pos = [1.5, -3.0, 5]; l.dir = [-0.25, 0.5, -1]; l.diffuse = [0.55, 0.55, 0.55]; l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    l2 = wb.add_light(); l2.pos = [-2.5, -2, 4]; l2.dir = [0.5, 0.4, -1]; l2.diffuse = [0.25, 0.25, 0.25]; l2.castshadow = False; l2.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    tex = spec.add_texture(); tex.name = 'sky'; tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX; tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    tex.rgb1 = [0.95, 0.945, 0.93]; tex.rgb2 = [0.95, 0.945, 0.93]; tex.width = 64; tex.height = 384
    spec.visual.global_.offwidth = 1920; spec.visual.global_.offheight = 1080
    spec.visual.headlight.ambient = [0.42, 0.42, 0.42]; spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]; spec.visual.headlight.specular = [0, 0, 0]
    sb = wb.add_body(); sb.name = _slug(name); _station_into(spec, sb, resolved, _slug(name))
    if z > 0.02:
        pg = wb.add_geom(); pg.name = 'pedestal'; pg.type = G.mjGEOM_BOX; pg.size = [0.11, 0.11, z / 2]; pg.pos = [x, y, z / 2]; pg.rgba = [0.2, 0.2, 0.22, 1]; pg.contype = 0; pg.conaffinity = 0
    it = wb.add_body(); it.name = 'item'; it.mocap = True; it.pos = [0, 0, -1]
    ig = it.add_geom(); ig.type = G.mjGEOM_BOX; ig.size = [0.06, 0.045, 0.03]; ig.rgba = [0.75, 0.55, 0.35, 1]; ig.contype = 0; ig.conaffinity = 0
    for i, nm in enumerate(objects):
        if nm.startswith('simready:'):        # real mass and friction from the asset, convex hull for contact
            from . import simready; am = simready.get(nm.split(':', 1)[1])
            ms = spec.add_mesh(); ms.name = f'o{i}_mesh'; ms.file = am['mesh']
            nb = wb.add_body(); nb.name = f'o{i}_model'; nb.pos = [3.0 + 0.5 * i, -3.0, 0.3]; nb.add_freejoint()
            og = nb.add_geom(); og.type = G.mjGEOM_MESH; og.meshname = ms.name; og.rgba = [0.85, 0.55, 0.2, 1]
            if am.get('mass'): og.mass = am['mass']
            if am.get('friction'): og.friction[0] = am['friction']
            continue
        ox = os.path.join(assets.GSO, 'models', nm, 'model.xml'); osp = _absolute(mujoco.MjSpec.from_file(ox), ox)
        fr = wb.add_frame(); fr.pos = [3.0 + 0.5 * i, -3.0, 0.3]
        nb = fr.attach_body(osp.worldbody.first_body(), f'o{i}_', ''); nb.add_freejoint()
    if camera:                      # fixed observation camera: {"pos": [..], "target": [..], "fovy": 60}
        cp = np.array(camera['pos'], float); f = np.array(camera['target'], float) - cp; f /= np.linalg.norm(f)
        rt = np.cross(f, [0, 0, 1]); rt /= np.linalg.norm(rt); up = np.cross(rt, f)
        cm = wb.add_camera(); cm.name = 'station_cam'; cm.pos = cp; cm.fovy = camera.get('fovy', 60)
        cq = np.zeros(4); mujoco.mju_mat2Quat(cq, np.column_stack([rt, up, -f]).flatten()); cm.quat = cq
    m = spec.compile(); d = mujoco.MjData(m)
    info = dict(robot=robot, tool=meta['tool'], spec=spec, qadr=np.array([m.joint(j).qposadr[0] for j in meta['joints']]), tip_site=m.site('tip').id,
                item=m.body('item').mocapid[0], objects=[m.body(f'o{i}_model').id for i in range(len(objects))])
    return m, d, info

def scene_view(resolved):
    """A camera that frames the whole workstation from the operator side."""
    P = np.array([g['pos'] + (0, 0, g['size'][2] / 2 if g['shape'] == 'mesh' else 0) for g in resolved['geoms']]); S = np.array([np.pad(g['size'], (0, 3 - len(g['size'])), constant_values=g['size'][-1]) for g in resolved['geoms']])
    lo = (P - S / 2).min(0); hi = (P + S / 2).max(0); c = (lo + hi) / 2; span = float(np.linalg.norm(hi - lo))
    return dict(lookat=(c[0], c[1], max(c[2], 1.0)), dist=max(1.2 * span, 3.0))

def render(m, d, view, w=1280, h=720, az=112, el=-14, colliders=False):
    r = mujoco.Renderer(m, h, w); cam = mujoco.MjvCamera(); cam.lookat[:] = view['lookat']; cam.distance = view['dist']; cam.azimuth = az; cam.elevation = el
    opt = mujoco.MjvOption(); opt.geomgroup[3] = int(colliders)
    mujoco.mj_forward(m, d); r.update_scene(d, cam, opt); img = r.render().copy(); r.close(); return img
