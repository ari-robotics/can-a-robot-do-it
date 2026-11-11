"""NVIDIA SimReady assets for the agent to pick from, one at a time: objects, fixtures, robot arms, grippers.

Search goes through the SimReady Central API (natural-language search over every SimReady dataset on Hugging Face,
manufacturers' robots included); if it is down, a name index of nvidia/simready-assets is the fallback. An asset is
then fetched alone, geometry and physics only (no textures), and read by MuJoCo 4's own OpenUSD importer, which
needs `pip install "ari-sim[simready]"` (mujoco[usd]).

An asset id is `<org>/<repo>/<path to root USD>@<revision>`, pinned so the same id is the same file everywhere. For
nvidia/simready-assets the short folder name also works (`crate_blue_a01`), at the revision pinned below.
"""
import os, re, json, functools, urllib.request, urllib.parse
import numpy as np
from . import assets

CENTRAL = 'https://nvidia-simready-central.hf.space'
REPO = 'nvidia/simready-assets'
REVISION = '7879e0e50423290300ac20ecd6e727325f680b60'
ROOT = os.path.join(assets.HOME, 'simready')
_KEEP = re.compile(r'\.(usd|usda|usdc)$|(^|/)com\.nvidia\.simready\.packaging\.json$')

def _open(url, timeout=60): return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ari-sim'}), timeout=timeout)

def _tree(repo, rev, path='', recursive=False):
    """Every entry of a dataset folder at a revision (follows pagination)."""
    url, out = f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{path}' + ('?recursive=true' if recursive else ''), []
    while url:
        r = _open(url); out += json.load(r)
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get('Link', '')); url = m.group(1) if m else None
    return out

# ---------- ids
def parse(ref):
    """Asset id, short name or Hugging Face URL -> dict(repo, rev, path, folder, id)."""
    ref = ref.split(':', 1)[1] if ref.startswith('simready:') else ref
    m = re.match(r'https://huggingface.co/datasets/([^/]+/[^/]+)/(?:blob|resolve)/([0-9a-f]{40})/(.+)$', ref)
    if m: repo, rev, path = m.groups()
    elif '@' in ref:
        body, rev = ref.rsplit('@', 1); parts = body.split('/')
        if len(parts) < 3 or not re.fullmatch(r'[0-9a-f]{40}', rev): raise KeyError(f'bad asset id {ref!r}: expected <org>/<repo>/<path>@<40-char revision>')
        repo, path = '/'.join(parts[:2]), '/'.join(parts[2:])
        if not re.search(r'\.usd[ac]?$', path):                 # <org>/<repo>/<folder>@<rev>: the root USD is looked up
            return dict(repo=repo, rev=rev, path=None, folder=parts[2], id=ref)
    elif '/' not in ref: repo, rev, path = REPO, REVISION, None
    else: raise KeyError(f'bad asset id {ref!r}')
    folder = path.split('/')[0] if path else ref
    return dict(repo=repo, rev=rev, path=path, folder=folder, id=f'{repo}/{path}@{rev}' if path else ref)

def _url(a, path): return f"https://huggingface.co/datasets/{a['repo']}/resolve/{a['rev']}/{path}"
def _dir(a): return os.path.join(ROOT, a['repo'].replace('/', '__'), f"{a['folder']}@{a['rev'][:12]}")

# ---------- search
def names():
    """Folder names of nvidia/simready-assets, cached: the fallback index when the search API is unreachable."""
    p = os.path.join(ROOT, 'index.json')
    if os.path.exists(p):
        ix = json.load(open(p))
        if ix['revision'] == REVISION: return ix['names']
    os.makedirs(ROOT, exist_ok=True)
    ns = sorted(e['path'] for e in _tree(REPO, REVISION) if e['type'] == 'directory')
    json.dump(dict(revision=REVISION, names=ns), open(p, 'w')); return ns

def tokens(name):
    out = set()
    for part in name.split('_'):
        if re.fullmatch(r'[a-z]\d+', part): continue
        out.add(part.lower()); out.update(w.lower() for w in re.findall(r'[a-z]+|[A-Z][a-z]*', part) if len(w) > 1)
    return out

SYNONYMS = dict(tote=['crate', 'bin', 'container'], bin=['crate', 'container'], shelf=['rack', 'storage'], shelving=['rack', 'storage'],
                table=['workbench', 'table'], bench=['workbench'], carton=['cardboard', 'box'], part=['component'], parts=['component'])

def search_names(query, n=10):
    """Fallback: rank nvidia/simready-assets names by the words they contain."""
    q = [w for w in re.findall(r'[a-z0-9]+', query.lower()) if len(w) > 1]
    q = list(dict.fromkeys(x for w in q for x in [w] + SYNONYMS.get(w, [])))
    scored = []
    for name in names():
        tk = tokens(name); s = sum(2 if w in tk else 1 if any(t.startswith(w) or (len(w) > 3 and w in t) for t in tk) else 0 for w in q)
        if s: scored.append((-s, len(name), name))
    return [name for _, _, name in sorted(scored)[:n]]

# manufacturers' datasets, indexed by folder name when the search API is down, so robots and grippers stay findable
MAKERS = ['FANUC-CORPORATION/SimReady-Assets', 'TeradyneRobotics/UniversalRobots', 'Robotiq-Official/simready-assets', 'onrobot/simready-grippers',
          'frankarobotics/simready-assets', 'standardbots/simready-core']

def maker_assets():
    """[(folder, id)] over the manufacturers' datasets at their current revision, cached."""
    p = os.path.join(ROOT, 'makers.json')
    if os.path.exists(p): return [tuple(x) for x in json.load(open(p))]
    out = []
    for repo in MAKERS:
        try:
            rev = json.load(_open(f'https://huggingface.co/api/datasets/{repo}'))['sha']
            out += [(e['path'], f"{repo}/{e['path']}@{rev}") for e in _tree(repo, rev) if e['type'] == 'directory' and not e['path'].startswith('.')]
        except Exception: continue
    if out: os.makedirs(ROOT, exist_ok=True); json.dump(out, open(p, 'w'))
    return out

def search(query, n=10):
    """Natural-language search over all SimReady datasets (robots included). Returns [(id, source)], source 'central' or 'names'."""
    try:
        hits = json.load(_open(f'{CENTRAL}/search?' + urllib.parse.urlencode(dict(description=query, limit=n * 2)), timeout=45))
        out = []
        for h in hits:
            i = parse(h['url'])['id']
            if i not in out: out.append(i)
        return [(i, 'central') for i in out[:n]]
    except Exception:
        q = [w for w in re.findall(r'[a-z0-9]+', query.lower()) if len(w) > 1]
        mk = sorted((-sum(w in re.sub(r'[^a-z0-9]', '', (f + i.split('/')[0]).lower()) for w in q), f, i) for f, i in maker_assets())
        return [(i, 'names') for s, _, i in mk if s < 0][:n // 2] + [(i, 'names') for i in search_names(query, n)]

# ---------- files
def files(ref):
    a = parse(ref); p = os.path.join(_dir(a), 'tree.json')
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump([dict(path=e['path'], size=e.get('size', 0)) for e in _tree(a['repo'], a['rev'], a['folder'], recursive=True) if e['type'] == 'file'], open(p, 'w'))
    return json.load(open(p))

def _wanted(ref):
    """USD layers and the packaging card only: no textures, thumbnails, material libraries or reports."""
    return [f for f in files(ref) if _KEEP.search(f['path']) and '/.thumbs/' not in f['path']]

def download_mb(ref): return sum(f['size'] for f in _wanted(ref)) / 1e6

def _root_usd(ref):
    a = parse(ref)
    if a['path']: return a['path']
    roots = [f['path'] for f in files(ref) if re.fullmatch(rf"{re.escape(a['folder'])}/simready_usd/[^/]+\.usda?", f['path'])]
    if not roots: raise KeyError(f'{ref}: no root USD found')
    return roots[0]

def fetch(ref, max_mb=200):
    """Download one asset's USD layers (resumable, pinned revision). Returns the local root USD path."""
    a = parse(ref)
    if a['repo'] == REPO and not a['path'] and a['folder'] not in names(): raise KeyError(f'{ref!r} is not in the SimReady catalog; use `ari-sim assets search`')
    want = _wanted(ref); mb = sum(f['size'] for f in want) / 1e6
    if mb > max_mb: raise ValueError(f"{a['folder']}: {mb:.0f} MB of geometry, above the {max_mb} MB limit (raise it with --max-mb if this asset is really needed)")
    for f in want:
        dst = os.path.join(_dir(a), 'src', f['path'])
        if not os.path.exists(dst) or os.path.getsize(dst) != f['size']:
            os.makedirs(os.path.dirname(dst), exist_ok=True); urllib.request.urlretrieve(_url(a, f['path']), dst + '.part'); os.replace(dst + '.part', dst)
    return os.path.join(_dir(a), 'src', _root_usd(ref))

def thumbnail(ref):
    """A 256 px preview from the search API (a few kB), or the dataset's own thumbnail if the API is down."""
    a = parse(ref); dst = os.path.join(_dir(a), 'thumbnail.jpg'); os.makedirs(_dir(a), exist_ok=True)
    if os.path.exists(dst): return dst
    try:
        data = _open(f'{CENTRAL}/images?' + urllib.parse.urlencode(dict(asset_url=_url(a, _root_usd(ref)).replace('/resolve/', '/blob/'))), timeout=45).read()
        open(dst, 'wb').write(data); return dst
    except Exception:
        th = [f for f in files(ref) if '/.thumbs/' in f['path'] and f['path'].endswith('.png') and 'dark' not in f['path'] and 'transparent' not in f['path']]
        if not th: raise FileNotFoundError(f'{ref}: no thumbnail')
        dst = dst[:-4] + '.png'; urllib.request.urlretrieve(_url(a, min(th, key=lambda f: f['size'])['path']), dst); return dst

# ---------- MuJoCo import
def _decode(path):
    """MuJoCo 4's OpenUSD importer, plus the fixes SimReady files need."""
    import mujoco
    try: s = mujoco.MjSpec.from_file(path)
    except ValueError as e:
        if 'decoder' in str(e): raise ImportError('reading SimReady assets needs MuJoCo with OpenUSD: pip install "ari-sim[simready]"') from None
        raise
    for j in list(s.joints):                                  # articulation roots come as free joints on a massless body
        if j.type == mujoco.mjtJoint.mjJNT_FREE: s.delete(j)
    for g in list(s.geoms):                                   # grasp markers and other zero-size helpers
        if g.type != mujoco.mjtGeom.mjGEOM_MESH and not np.any(g.size): s.delete(g)
    for _ in range(3):
        try: s.compile(); return s
        except ValueError as e:
            msg = str(e)
            if 'mesh volume is too small' in msg:              # tiny decorative meshes; masses are declared on the bodies
                for ms in s.meshes: ms.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            elif 'fullinertia and diagonal inertia' in msg:
                for b in s.bodies:
                    if np.any(b.fullinertia) and np.any(b.inertia): b.fullinertia = [np.nan] * 6
            else: raise
    return s

def get(ref, max_mb=200):
    """An object or fixture: fetched, imported, flattened to one mesh with its true extent, mass and friction. Returns its meta."""
    a = parse(ref); meta_p = os.path.join(_dir(a), 'meta.json')
    if os.path.exists(meta_p): return json.load(open(meta_p))
    import mujoco
    s = _decode(fetch(ref, max_mb)); m = s.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    V, F, n = [], [], 0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        k = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]]
        V.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]); F.append(f + n); n += len(v)
    if not V: raise ValueError(f'{ref}: no mesh in the USD')
    V = np.vstack(V); F = np.vstack(F); lo, hi = V.min(0), V.max(0); V -= [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
    obj = os.path.join(_dir(a), 'mesh.obj')
    with open(obj, 'w') as f:
        f.write(''.join(f'v {x:.6f} {y:.6f} {z:.6f}\n' for x, y, z in V)); f.write(''.join(f'f {i + 1} {j + 1} {k + 1}\n' for i, j, k in F))
    ext = hi - lo; mass = float(m.body_mass[1:].sum()) or None
    friction = _usd_friction(fetch(ref, max_mb))                # MuJoCo's importer leaves USD physics materials at its default
    lic = license(ref)
    warn = []   # flag what a person should double-check before trusting the numbers
    if mass and mass / max(float(np.prod(ext)), 1e-9) < 10: warn.append(f'mass {mass:.3g} kg is implausibly low for a {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m object')
    if max(ext) > 5 or max(ext) < 0.005: warn.append(f'size {max(ext):.3g} m looks wrong (unit error?)')
    meta = dict(id=a['id'], name=a['folder'], mesh=obj, extent=[round(float(x), 4) for x in ext], triangles=len(F), mass=mass, friction=friction, warnings=warn,
                license=lic, attribution=f"{a['folder']}, SimReady ({a['repo']}), {lic}", download_mb=round(download_mb(ref), 1))
    json.dump(meta, open(meta_p, 'w'), indent=1); return meta

def license(ref):
    """Licence declared in the asset's SimReady packaging card (downloaded with the asset)."""
    a = parse(ref); card = os.path.join(_dir(a), 'src', a['folder'], 'com.nvidia.simready.packaging.json')
    return json.load(open(card)).get('license', 'see dataset card') if os.path.exists(card) else 'see dataset card'

def _usd_friction(root):
    """Static friction of the first USD physics material, if the asset declares one."""
    from pxr import Usd, UsdPhysics
    st = Usd.Stage.Open(root)                                  # keep the stage alive while its prims are walked
    for p in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.MaterialAPI):
            v = p.GetAttribute('physics:staticFriction').Get()
            if v is not None: return round(float(v), 3)
    return None

def _flange(root, s):
    """(body name, pos, quat) of the tool flange with z out of the flange: `tool0` (z out) if declared, else `flange` (x out)."""
    import mujoco
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(root); xc = UsdGeom.XformCache(); bodies = {b.name for b in s.bodies}
    found = {p.GetName(): p for p in st.Traverse() if p.GetName() in ('tool0', 'flange')}
    prim = found.get('tool0') or found.get('flange')
    if prim is None: raise ValueError('no tool0 or flange frame in the robot USD')
    par = prim.GetParent()
    while par and str(par.GetPath()) not in bodies: par = par.GetParent()
    if not par: raise ValueError(f'{prim.GetPath()}: no robot link above the flange')
    rel = xc.GetLocalToWorldTransform(prim) * xc.GetLocalToWorldTransform(par).GetInverse()
    t = np.array(rel.ExtractTranslation()); q = rel.ExtractRotationQuat(); q = np.array([q.GetReal(), *q.GetImaginary()])
    if prim.GetName() == 'flange':                            # REP-199 flange: x points out of the tool plate; turn it so z does
        out = np.zeros(4); mujoco.mju_mulQuat(out, q, np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])); q = out
    return str(par.GetPath()), t, q

@functools.lru_cache(maxsize=None)
def _tool(rid):
    import mujoco
    s = _decode(fetch(rid)); m = s.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d); top = 0.0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        k = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        top = max(top, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].max()))
    return s, round(top, 3)

def tool(ref):
    """An end effector from SimReady: (MjSpec copy to mount on a flange, its length along the flange axis in metres)."""
    s, length = _tool(parse(ref)['id'])
    return s.copy(), length

@functools.lru_cache(maxsize=None)
def _robot(rid):
    import mujoco
    root = fetch(rid); s = _decode(root); tip, pos, quat = _flange(root, s)
    st = s.body(tip).add_site(); st.name = 'attachment_site'; st.pos = pos; st.quat = quat
    m = s.compile(); b = m.body(tip).id
    while m.body_parentid[b] != 0: b = m.body_parentid[b]
    hinges = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    return s, m.body(b).name, tip, hinges

def robot(ref):
    """A robot arm from SimReady: (MjSpec copy with an `attachment_site` at the flange, base body, tip body, hinge joint names)."""
    s, base, tip, hinges = _robot(parse(ref)['id'])
    return s.copy(), base, tip, hinges
