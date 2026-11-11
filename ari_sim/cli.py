"""ari-sim: a workstation scene written by your coding agent -> simulation -> diagnostic."""
import argparse, json, os, sys, glob

def main(argv=None):
    ap = argparse.ArgumentParser(prog='ari-sim', description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('setup', help='download robot models and scanned objects'); s.add_argument('--objects', type=int, default=110)
    c = sub.add_parser('check', help='validate a scene.json and render it'); c.add_argument('scene'); c.add_argument('--out', default='out')
    d = sub.add_parser('diagnose', help='scene.json -> robot choice, reach, cycle time, uncertainty')
    d.add_argument('--variants', type=int, default=40); d.add_argument('--no-video', action='store_true')
    d.add_argument('scene'); d.add_argument('--out', default='out')
    g = sub.add_parser('grasp', help='physics grasping episodes with the scene\'s item assets, or scanned objects (experimental)')
    g.add_argument('scene'); g.add_argument('--episodes', type=int, default=20); g.add_argument('--out', default='out/episodes'); g.add_argument('--seed', type=int, default=0)
    g.add_argument('--base', nargs=3, type=float, metavar=('X', 'Y', 'PEDESTAL'), help='robot base; default: best_fixed from <scene dir>/out/diagnostic.json')
    ac = sub.add_parser('assets', help='find and fetch real objects from the NVIDIA SimReady catalog, one at a time').add_subparsers(dest='acmd', required=True)
    q = ac.add_parser('search', help='rank catalog names against a few words'); q.add_argument('query', nargs='+'); q.add_argument('-n', type=int, default=10)
    ac.add_parser('categories', help='catalog categories and how many assets each holds')
    v = ac.add_parser('show', help='download one asset\'s thumbnail, to compare with the photos'); v.add_argument('name')
    t = ac.add_parser('get', help='download one asset\'s geometry and physics and convert it for MuJoCo'); t.add_argument('name'); t.add_argument('--max-mb', type=float, default=200)
    rc = sub.add_parser('robots', help='robot arms available, and fetching their models').add_subparsers(dest='rcmd', required=True)
    rc.add_parser('list', help='arms with reach, payload and whether the model is on disk')
    r = rc.add_parser('get', help='download these arms\' models from MuJoCo Menagerie'); r.add_argument('names', nargs='+')
    a = ap.parse_args(argv)
    if a.cmd == 'setup':
        from . import assets; print('assets in', assets.setup(a.objects)); return
    if a.cmd == 'assets':
        from . import simready as SR
        if a.acmd == 'categories': print('\n'.join(f'{k:24s}{v}' for k, v in SR.categories())); return
        if a.acmd == 'search':
            hits = SR.search(' '.join(a.query), a.n)
            if not hits: print('no match; try the catalog\'s own words (`ari-sim assets categories`)'); return
            for h in hits: print(f'{h:56s}{SR.download_mb(h):8.1f} MB')
            return
        if a.acmd == 'show': print(SR.thumbnail(a.name)); return
        m = SR.get(a.name, a.max_mb); print(json.dumps({k: m[k] for k in ('name', 'extent', 'mass', 'mass_source', 'friction', 'triangles', 'warnings', 'attribution')}, indent=1)); return
    if a.cmd == 'robots':
        from . import assets; from .robots import ROBOTS, DEFAULT_CANDIDATES, menagerie_dir
        if a.rcmd == 'get': print('fetched:', assets.fetch_robots(a.names, gripper=False) or 'nothing, already on disk'); return
        for k, R in ROBOTS.items():
            on = os.path.isdir(os.path.join(assets.MENAGERIE, menagerie_dir(k)))
            print(f"{k:10s} reach {R['reach']:.2f} m  payload {R['payload']:4.1f} kg  {'on disk' if on else 'fetched on first use'}{'  (default)' if k in DEFAULT_CANDIDATES else ''}")
        return
    from . import spec as S
    sc = S.load(a.scene)
    if a.cmd == 'check':
        import numpy as np; from PIL import Image; from .build import build, scene_view, render
        R = S.resolve(sc); m, dd, info = build('UR10e', R, sc['robot'].get('tool')); v = scene_view(R); os.makedirs(a.out, exist_ok=True)
        Image.fromarray(np.hstack([render(m, dd, v, 960, 720, az=118), render(m, dd, v, 960, 720, az=62)])).save(os.path.join(a.out, 'scene.png'))
        print(f"OK: {len(R['geoms'])} primitives, {len(R['picks'])} pick targets, {len(R['places'])} place targets -> {a.out}/scene.png"); return
    if a.cmd == 'diagnose':
        from .diagnose import diagnose; from .report import write_all, VERDICT
        print('Running the diagnostic…'); res = diagnose(sc, n_variants=a.variants); write_all(sc, res, a.out, with_video=not a.no_video)
        print('\n' + VERDICT[res['verdict']]); print(f"Best: {res['best_robot']} at {res['best_fixed']['x']:.2f}, {res['best_fixed']['y']:.2f}, pedestal {res['best_fixed']['pedestal']:.2f} m "
              f"— {res['best_fixed']['reached']}/{res['n_targets']} targets"); print('Report in', a.out); return
    if a.cmd == 'grasp':
        from .grasp import run_batch
        base = a.base
        if base is None:
            dj = os.path.join(os.path.dirname(os.path.abspath(a.scene)), 'out', 'diagnostic.json')
            if not os.path.exists(dj): sys.exit('No --base given and no diagnostic.json found: run `ari-sim diagnose` first or pass --base X Y PEDESTAL.')
            bf = json.load(open(dj))['best_fixed']; base = [bf['x'], bf['y'], bf['pedestal']]
        stats = run_batch(sc, base, a.episodes, a.out, seed=a.seed); print(stats)

if __name__ == '__main__': main()
