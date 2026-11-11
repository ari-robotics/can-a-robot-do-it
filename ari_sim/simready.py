"""NVIDIA SimReady assets (OpenUSD, CC-BY 4.0) for the agent to pick from, one at a time.

The catalog holds 2,677 assets and about 490 GB. Nothing is bulk-downloaded: the agent searches the names
(the category is the first word: handTool_, crate_, pallet_...), looks at a thumbnail, and only then fetches
one asset's geometry and physics layers (no textures), which are converted to a MuJoCo mesh with real-world
dimensions, density and friction. Reading USD needs `pip install "ari-sim[simready]"` (usd-core).
"""
import os, re, json, urllib.request
import numpy as np
from . import assets

REPO = 'nvidia/simready-assets'
REVISION = '7879e0e50423290300ac20ecd6e727325f680b60'   # pinned: the same asset is the same file everywhere
API = f'https://huggingface.co/api/datasets/{REPO}'
FILES = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
ROOT = os.path.join(assets.HOME, 'simready')

def _open(url): return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ari-sim'}), timeout=60)

def _tree(path='', recursive=False):
    """Every entry of a folder of the dataset at the pinned revision (follows pagination)."""
    url, out = f'{API}/tree/{REVISION}/{path}' + ('?recursive=true' if recursive else ''), []
    while url:
        r = _open(url); out += json.load(r)
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get('Link', '')); url = m.group(1) if m else None
    return out

def names():
    """All asset names, cached after the first call (a few API calls, a few hundred kB)."""
    p = os.path.join(ROOT, 'index.json')
    if os.path.exists(p):
        ix = json.load(open(p))
        if ix['revision'] == REVISION: return ix['names']
    os.makedirs(ROOT, exist_ok=True)
    ns = sorted(e['path'] for e in _tree() if e['type'] == 'directory')
    json.dump(dict(revision=REVISION, names=ns), open(p, 'w')); return ns

def category(name): return name.split('_')[0]

def tokens(name):
    """crate_blue_a01 -> {'crate', 'blue'}; handTool_screwdriver_phillips_ph2_a01 -> {'hand', 'tool', 'handtool', 'screwdriver', ...}"""
    out = set()
    for part in name.split('_'):
        if re.fullmatch(r'[a-z]\d+', part): continue                       # variant code such as a01
        out.add(part.lower()); out.update(w.lower() for w in re.findall(r'[a-z]+|[A-Z][a-z]*', part) if len(w) > 1)
    return out

def categories():
    c = {}
    for n in names(): c[category(n)] = c.get(category(n), 0) + 1
    return sorted(c.items(), key=lambda kv: -kv[1])

# words people use for a workstation that the catalog spells differently
SYNONYMS = dict(tote=['crate', 'bin', 'container'], bin=['crate', 'container'], shelf=['rack', 'storage'], shelving=['rack', 'storage'],
                table=['workbench', 'table'], bench=['workbench'], carton=['cardboard', 'box'], part=['component'], parts=['component'])

def search(query, n=10):
    """Names ranked by how many query words they contain (whole word first, then prefix)."""
    q = [w for w in re.findall(r'[a-z0-9]+', query.lower()) if len(w) > 1]
    q = list(dict.fromkeys(x for w in q for x in [w] + SYNONYMS.get(w, [])))
    scored = []
    for name in names():
        tk = tokens(name); s = sum(2 if w in tk else 1 if any(t.startswith(w) or (len(w) > 3 and w in t) for t in tk) else 0 for w in q)
        if s: scored.append((-s, len(name), name))
    return [name for _, _, name in sorted(scored)[:n]]

def files(name):
    """The asset's file list, cached."""
    p = os.path.join(ROOT, name, 'tree.json')
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump([dict(path=e['path'], size=e.get('size', 0)) for e in _tree(name, recursive=True) if e['type'] == 'file'], open(p, 'w'))
    return json.load(open(p))

def _wanted(name):
    """Geometry and physics only: the root USD, its payload layers and the packaging card. No textures, no thumbnails."""
    out = []
    for f in files(name):
        rel = f['path'].split('/', 1)[1]
        if rel.startswith('simready_usd/payloads/') or re.fullmatch(r'simready_usd/[^/]+\.usda?', rel) or rel == 'com.nvidia.simready.packaging.json': out.append(f)
    return out

def download_mb(name): return sum(f['size'] for f in _wanted(name)) / 1e6

def thumbnail(name):
    """Download the smallest preview image of the asset and return its local path."""
    th = [f for f in files(name) if '/.thumbs/' in f['path'] and f['path'].endswith('.png') and 'dark' not in f['path'] and 'transparent' not in f['path']]
    if not th: raise FileNotFoundError(f'{name}: no thumbnail in the catalog')
    f = min(th, key=lambda f: f['size']); dst = os.path.join(ROOT, name, 'thumbnail.png')
    if not os.path.exists(dst): urllib.request.urlretrieve(FILES + f['path'], dst)
    return dst

def get(name, max_mb=200):
    """Fetch one asset's geometry and physics and convert it for MuJoCo. Returns its meta (extent, density, friction, mesh path)."""
    meta_p = os.path.join(ROOT, name, 'meta.json')
    if os.path.exists(meta_p): return json.load(open(meta_p))
    if name not in names(): raise KeyError(f'{name!r} is not in the SimReady catalog; use `ari-sim assets search`')
    want = _wanted(name); mb = sum(f['size'] for f in want) / 1e6
    if mb > max_mb: raise ValueError(f'{name}: {mb:.0f} MB of geometry, above the {max_mb} MB limit (raise it with --max-mb if this asset is really needed)')
    for f in want:
        dst = os.path.join(ROOT, name, 'src', f['path'].split('/', 1)[1])
        if not os.path.exists(dst) or os.path.getsize(dst) != f['size']:
            os.makedirs(os.path.dirname(dst), exist_ok=True); urllib.request.urlretrieve(FILES + f['path'], dst + '.part'); os.replace(dst + '.part', dst)
    meta = _convert(name, os.path.join(ROOT, name)); meta['download_mb'] = round(mb, 1)
    json.dump(meta, open(meta_p, 'w'), indent=1); return meta

def _convert(name, folder):
    try: from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError: raise ImportError('reading SimReady assets needs usd-core: pip install "ari-sim[simready]"') from None
    src = os.path.join(folder, 'src'); root = [os.path.join(src, 'simready_usd', f) for f in sorted(os.listdir(os.path.join(src, 'simready_usd'))) if f.endswith(('.usd', '.usda'))][0]
    st = Usd.Stage.Open(root); mpu = UsdGeom.GetStageMetersPerUnit(st); up = UsdGeom.GetStageUpAxis(st); dp = st.GetDefaultPrim()
    vs = dp.GetVariantSets()
    for v in vs.GetNames():                                                 # the MuJoCo flavour of the physics layers
        opts = vs.GetVariantSet(v).GetVariantNames()
        if 'Enabled' in opts: vs.GetVariantSet(v).SetVariantSelection('Enabled' if v == 'MuJoCo' else 'Disabled' if 'Disabled' in opts else 'Enabled')
    xc = UsdGeom.XformCache(); V, F, n, density, friction, mass = [], [], 0, None, None, None
    for prim in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.MaterialAPI) and density is None:
            density = prim.GetAttribute('physics:density').Get(); friction = prim.GetAttribute('physics:staticFriction').Get()
        if prim.HasAPI(UsdPhysics.MassAPI) and (prim.GetAttribute('physics:mass').Get() or 0) > 0: mass = float(prim.GetAttribute('physics:mass').Get())
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide: continue
        mesh = UsdGeom.Mesh(prim); pts = mesh.GetPointsAttr().Get(); cnt = mesh.GetFaceVertexCountsAttr().Get(); idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not cnt: continue
        M = np.array(xc.GetLocalToWorldTransform(prim))
        with np.errstate(all='ignore'): P = (np.c_[np.array(pts, float), np.ones(len(pts))] @ M)[:, :3] * mpu   # some numpy builds warn spuriously here
        if not np.isfinite(P).all(): raise ValueError(f'{name}: non-finite vertex in {prim.GetPath()}')
        k = 0
        for c in cnt:
            F += [[idx[k] + n, idx[k + j] + n, idx[k + j + 1] + n] for j in range(1, c - 1)]; k += c
        V.append(P); n += len(P)
    if not V: raise ValueError(f'{name}: no mesh found in the USD')
    V = np.vstack(V); F = np.array(F, int); mass_declared = mass is not None
    if up == 'Y': V = V[:, [0, 2, 1]] * [1, -1, 1]                           # to z up
    lo, hi = V.min(0), V.max(0); V -= [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]   # origin: floor point under the centre
    obj = os.path.join(folder, 'mesh.obj')
    with open(obj, 'w') as f:
        f.write(''.join(f'v {x:.6f} {y:.6f} {z:.6f}\n' for x, y, z in V)); f.write(''.join(f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in F))
    t = V[F]; vol = abs(np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6
    card = os.path.join(folder, 'src', 'com.nvidia.simready.packaging.json'); lic = json.load(open(card)).get('license', 'CC-BY-4.0') if os.path.exists(card) else 'CC-BY-4.0'
    mass = mass if mass else (round(vol * density, 3) if density else None); ext = hi - lo; warn = []
    # catalog data is not always right (an item 36 cm tall declared at 9 g): flag what a person should double-check
    if mass and mass / max(float(np.prod(ext)), 1e-9) < 10: warn.append(f'mass {mass:.3g} kg is implausibly low for a {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m object')
    if max(ext) > 5 or max(ext) < 0.005: warn.append(f'size {max(ext):.3g} m looks wrong (unit error?)')
    return dict(name=name, category=category(name), mesh=obj, extent=[round(float(x), 4) for x in ext], triangles=len(F),
                density=float(density) if density else None, friction=float(friction) if friction else None,
                mass=mass, mass_source='declared' if mass and mass_declared else 'mesh volume x density', warnings=warn,
                license=lic, attribution=f'{name}, NVIDIA SimReady ({REPO}), {lic}', revision=REVISION)
