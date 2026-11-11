"""Where robot and object assets live, and how to fetch them."""
import os, subprocess, random, re, json

HOME = os.environ.get('ARI_SIM_ASSETS', os.path.join(os.path.expanduser('~'), '.cache', 'ari_sim'))
MENAGERIE = os.path.join(HOME, 'mujoco_menagerie')
GSO = os.path.join(HOME, 'gso')
GSO_COMMIT = '6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c'   # pinned so the seeded object subset is the same everywhere

def _run(cmd, cwd=None): subprocess.run(cmd, cwd=cwd, check=True)

def setup(n_objects=110, seed=3):
    """Download MuJoCo Menagerie (robots, gripper) and a random subset of Google Scanned Objects (MJCF, CC-BY 4.0)."""
    os.makedirs(HOME, exist_ok=True)
    if not os.path.isdir(MENAGERIE):
        _run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/google-deepmind/mujoco_menagerie.git', MENAGERIE])
    if n_objects and not os.path.isdir(os.path.join(GSO, 'models')):
        _run(['git', 'init', '-q', GSO]); _run(['git', 'remote', 'add', 'origin', 'https://github.com/kevinzakka/mujoco_scanned_objects.git'], cwd=GSO)
        _run(['git', 'fetch', '-q', '--filter=blob:none', '--depth', '1', 'origin', GSO_COMMIT], cwd=GSO)
        names = subprocess.run(['git', 'ls-tree', '--name-only', f'{GSO_COMMIT}:models'], cwd=GSO, capture_output=True, text=True, check=True).stdout.split()
        # shoes, furniture and other bulky items never fit a bin or a parallel gripper: skip them up front
        bad = re.compile(r'ASICS|Adrenaline|Reebok|Nike|Timberland|UGG|adidas|Womens|Mens|Shoe|Boot|Sneaker|Tieks|Motherboard|Dollhouse|Sandal|Loafer|Rack|Hamper|Basket|Table|Chair|Sofa|Pot|Pan|Skillet|Plate|Bowl|Bag|Backpack|Pail|Lamp|Laptop|Router', re.I)
        pool = [n for n in names if not bad.search(n)]; random.Random(seed).shuffle(pool); pick = pool[:n_objects]
        _run(['git', 'sparse-checkout', 'init', '--no-cone'], cwd=GSO)
        with open(os.path.join(GSO, '.git', 'info', 'sparse-checkout'), 'w') as f: f.write('\n'.join(f'/models/{n}/' for n in pick) + '\n')
        _run(['git', 'checkout', '-q', GSO_COMMIT], cwd=GSO)
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
