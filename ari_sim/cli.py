"""ari-sim: photos or video of a workstation -> simulation -> diagnostic."""
import argparse, json, os, sys, glob

def main(argv=None):
    ap = argparse.ArgumentParser(prog='ari-sim', description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('setup', help='download robot models and scanned objects'); s.add_argument('--objects', type=int, default=110)
    p = sub.add_parser('perceive', help='photos/video -> scene.json (needs ANTHROPIC_API_KEY)')
    r = sub.add_parser('run', help='photos/video -> scene.json -> diagnostic, in one go')
    for q in (p, r):
        q.add_argument('--photos', nargs='*', default=[]); q.add_argument('--video'); q.add_argument('--task', required=True); q.add_argument('--out', default='out')
    c = sub.add_parser('check', help='validate a scene.json and render it'); c.add_argument('scene'); c.add_argument('--out', default='out')
    d = sub.add_parser('diagnose', help='scene.json -> robot choice, reach, cycle time, uncertainty')
    for q in (d, r): q.add_argument('--variants', type=int, default=40); q.add_argument('--no-video', action='store_true')
    d.add_argument('scene'); d.add_argument('--out', default='out')
    g = sub.add_parser('grasp', help='physics grasping episodes with scanned objects (experimental)')
    g.add_argument('scene'); g.add_argument('--episodes', type=int, default=20); g.add_argument('--out', default='out/episodes'); g.add_argument('--seed', type=int, default=0)
    g.add_argument('--base', nargs=3, type=float, metavar=('X', 'Y', 'PEDESTAL'), help='robot base; default: best_fixed from <scene dir>/out/diagnostic.json')
    a = ap.parse_args(argv)
    if a.cmd == 'setup':
        from . import assets; print('assets in', assets.setup(a.objects)); return
    from . import spec as S
    if a.cmd in ('perceive', 'run'):
        from .perceive import perceive
        os.makedirs(a.out, exist_ok=True); scene_path = os.path.join(a.out, 'scene.json')
        print('Reading the workstation…'); perceive(a.photos, a.task, scene_path, video=a.video); print('scene written to', scene_path)
        if a.cmd == 'perceive': return
        a.scene = scene_path
    sc = S.load(a.scene)
    if a.cmd == 'check':
        import numpy as np; from PIL import Image; from .build import build, scene_view, render
        R = S.resolve(sc); m, dd, info = build('UR10e', R, sc['robot'].get('tool')); v = scene_view(R); os.makedirs(a.out, exist_ok=True)
        Image.fromarray(np.hstack([render(m, dd, v, 960, 720, az=118), render(m, dd, v, 960, 720, az=62)])).save(os.path.join(a.out, 'scene.png'))
        print(f"OK: {len(R['geoms'])} primitives, {len(R['picks'])} pick targets, {len(R['places'])} place targets -> {a.out}/scene.png"); return
    if a.cmd in ('diagnose', 'run'):
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
