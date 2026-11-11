"""Where robot and object assets live, and how to fetch them."""
import os, subprocess, random, re, json

HOME = os.environ.get('ARI_SIM_ASSETS', os.path.join(os.path.expanduser('~'), '.cache', 'ari_sim'))
MENAGERIE = os.path.join(HOME, 'mujoco_menagerie')
GSO = os.path.join(HOME, 'gso')
GSO_COMMIT = '6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c'   # pinned so the seeded object subset is the same everywhere
MENAGERIE_COMMIT = '98a70c569c4c1b6da467a20012f766ee1f8dc521'

def _run(cmd, cwd=None): subprocess.run(cmd, cwd=cwd, check=True)

def _fetch(url, commit, dest):
    """One pinned commit, file contents fetched lazily: only what gets checked out is downloaded."""
    _run(['git', 'init', '-q', dest]); _run(['git', 'remote', 'add', 'origin', url], cwd=dest)
    _run(['git', 'fetch', '-q', '--filter=blob:none', '--depth', '1', 'origin', commit], cwd=dest)
    _run(['git', 'sparse-checkout', 'init', '--no-cone'], cwd=dest)

def _checkout(dest, commit, dirs):
    """Check out these folders, adding to what is already there."""
    sc = os.path.join(dest, '.git', 'info', 'sparse-checkout'); fresh = os.listdir(dest) == ['.git']
    lines = set() if fresh else set(open(sc).read().split())   # a fresh init holds git's default rules, which match everything
    with open(sc, 'w') as f: f.write('\n'.join(sorted(lines | {f'/{d}/' for d in dirs})) + '\n')
    _run(['git', 'checkout', '-q', commit] if fresh else ['git', 'sparse-checkout', 'reapply'], cwd=dest)

def setup(n_objects=110, seed=3):
    """Download the robot and gripper models this code uses from MuJoCo Menagerie, and a random subset of Google Scanned Objects (MJCF, CC-BY 4.0)."""
    from .robots import ROBOTS, GRIPPER_XML
    os.makedirs(HOME, exist_ok=True)
    dirs = sorted({r['xml'].split('/')[0] for r in ROBOTS.values()} | {GRIPPER_XML.split('/')[0]})
    if not os.path.isdir(os.path.join(MENAGERIE, '.git')): _fetch('https://github.com/google-deepmind/mujoco_menagerie.git', MENAGERIE_COMMIT, MENAGERIE)
    missing = [d for d in dirs if not os.path.isdir(os.path.join(MENAGERIE, d))]
    if missing: _checkout(MENAGERIE, MENAGERIE_COMMIT, missing)
    if n_objects and not os.path.isdir(os.path.join(GSO, 'models')):
        _fetch('https://github.com/kevinzakka/mujoco_scanned_objects.git', GSO_COMMIT, GSO)
        names = subprocess.run(['git', 'ls-tree', '--name-only', f'{GSO_COMMIT}:models'], cwd=GSO, capture_output=True, text=True, check=True).stdout.split()
        # shoes, furniture and other bulky items never fit a bin or a parallel gripper: skip them up front
        bad = re.compile(r'ASICS|Adrenaline|Reebok|Nike|Timberland|UGG|adidas|Womens|Mens|Shoe|Boot|Sneaker|Tieks|Motherboard|Dollhouse|Sandal|Loafer|Rack|Hamper|Basket|Table|Chair|Sofa|Pot|Pan|Skillet|Plate|Bowl|Bag|Backpack|Pail|Lamp|Laptop|Router', re.I)
        pool = [n for n in names if not bad.search(n)]; random.Random(seed).shuffle(pool)
        _checkout(GSO, GSO_COMMIT, [f'models/{n}' for n in pool[:n_objects]])
    return HOME

def graspable_objects(max_dim=0.20, min_thin=0.015, max_thin=0.07):
    """Objects from the downloaded subset that fit a shelf cell and an 85 mm parallel gripper. Cached."""
    import mujoco
    cache = os.path.join(GSO, 'meta.json'); root = os.path.join(GSO, 'models')
    if not os.path.isdir(root): raise FileNotFoundError('No scanned objects found. Run `ari-sim setup` first.')
    if os.path.exists(cache): meta = json.load(open(cache))
    else:
        meta = []
        for n in sorted(os.listdir(root)):
            try: m = mujoco.MjModel.from_xml_path(os.path.join(root, n, 'model.xml'))
            except Exception: continue
            v = m.mesh_vert[m.mesh_vertadr[0]:m.mesh_vertadr[0] + m.mesh_vertnum[0]]
            meta.append(dict(name=n, ext=[float(x) for x in (v.max(0) - v.min(0))]))
        json.dump(meta, open(cache, 'w'))
    return [r['name'] for r in meta if max(r['ext']) <= max_dim and min_thin <= min(r['ext']) <= max_thin]
