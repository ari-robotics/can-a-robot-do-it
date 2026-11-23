"""ari-sim: a workstation scene written by your coding agent -> simulation -> diagnostic."""
import argparse, json, os, sys, glob

def main(argv=None):
    ap = argparse.ArgumentParser(prog='ari-sim', description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('setup', help='download robot models and scanned objects'); s.add_argument('--objects', type=int, default=110)
    c = sub.add_parser('check', help='validate a scene.json and render it'); c.add_argument('scene'); c.add_argument('--out', default='out')
    d = sub.add_parser('diagnose', help='scene.json -> robot choice, reach, cycle time, uncertainty')
    d.add_argument('--variants', type=int, default=40); d.add_argument('--no-video', action='store_true')
    d.add_argument('scene'); d.add_argument('--out', default='out')
    g = sub.add_parser('execute', aliases=['grasp'], help='the pick with contact physics: staged episodes with the scene\'s item assets, or scanned objects (experimental)')
    g.add_argument('scene'); g.add_argument('--episodes', type=int, default=20); g.add_argument('--out', default='out/episodes'); g.add_argument('--seed', type=int, default=0)
    g.add_argument('--base', nargs=3, type=float, metavar=('X', 'Y', 'PEDESTAL'), help='robot base; default: best_fixed from <scene dir>/out/diagnostic.json')
    g.add_argument('--robot', help='arm; default: best_robot from the diagnostic'); g.add_argument('--perception', choices=['camera', 'truth'], default='camera', help='locate the part from the station camera (default) or from simulator state')
    ac = sub.add_parser('assets', help='find and fetch real objects from the NVIDIA SimReady catalog, one at a time').add_subparsers(dest='acmd', required=True)
    q = ac.add_parser('search', help='natural-language search over all SimReady datasets, robots included'); q.add_argument('query', nargs='+'); q.add_argument('-n', type=int, default=8)
    v = ac.add_parser('show', help='download one asset\'s thumbnail (a few kB), to compare with the photos'); v.add_argument('id')
    t = ac.add_parser('get', help='download one object\'s geometry and physics and import it into MuJoCo'); t.add_argument('id'); t.add_argument('--max-mb', type=float, default=200)
    rc = sub.add_parser('robots', help='robot arms: presets, and checking a SimReady arm before using it').add_subparsers(dest='rcmd', required=True)
    rc.add_parser('list', help='preset arms with reach, payload and where the model comes from')
    r = rc.add_parser('get', help='download these presets\' models'); r.add_argument('names', nargs='+')
    ri = rc.add_parser('inspect', help='import a SimReady arm and show its joints, limits and flange'); ri.add_argument('id')
    sub.add_parser('doctor', help="which of Google's robotics tools are installed, and what each one is used for")
    au = sub.add_parser('audit', help="Google's mujoco-audit on the rebuilt station (static) or on one asset (physics scenarios)")
    au.add_argument('scene', help='scene.json, or an asset model.xml'); au.add_argument('--out', default='out'); au.add_argument('--scenarios', default='', help="e.g. 'all' or '1,4,5' for an asset model.xml")
    sh = sub.add_parser('showcase', help="Google's showcase shot: turntable video and keyframe sheet of the rebuilt station"); sh.add_argument('scene'); sh.add_argument('--out', default='out'); sh.add_argument('--duration', type=float, default=8.0)
    stu = sub.add_parser('studio', help='open the station and the winning cell in MuJoCo Studio (needs a diagnosed scene)'); stu.add_argument('scene'); stu.add_argument('--out', default='out'); stu.add_argument('--test', action='store_true', help='headless self-test, then exit')
    me = sub.add_parser('measure', help='metres from pixels: a scale reference in a photo gives positions on its plane')
    me.add_argument('--ref', nargs=4, type=float, action='append', required=True, metavar=('U', 'V', 'X', 'Y'), help='a reference point: pixel u v, known position x y in metres (at least 4)')
    me.add_argument('--point', nargs=2, type=float, action='append', required=True, metavar=('U', 'V')); me.add_argument('--pixel-sigma', type=float, default=4.0)
    de = sub.add_parser('detect', help='pixels of named objects in a photo, with Gemini (only if GEMINI_API_KEY is set)'); de.add_argument('image'); de.add_argument('labels', nargs='+')
    a = ap.parse_args(argv)
    if a.cmd == 'setup':
        from . import assets, google
        if not google.path('motion-planning', need=False): print("fetching Google's robotics plugin into", google.fetch())
        print('assets in', assets.setup(a.objects)); return
    if a.cmd == 'doctor':
        from . import google; import shutil, importlib.util as iu; rows = google.status(); stage = None
        for r in sorted(rows, key=lambda r: ['perceive', 'rebuild', 'plan', 'execute', 'verify', 'report', 'real'].index(r['stage'])):
            if r['stage'] != stage: stage = r['stage']; print(f'\n{stage}')
            print(f"  {'ok     ' if r['ok'] else 'MISSING'} {r['skill']:26s} {r['use']}")
        import mujoco; print(f"\nMuJoCo {mujoco.__version__}; Studio {'ok' if iu.find_spec('mujoco_studio_wasm') else 'missing (pip install -U mujoco>=4.0.1)'}; uv {'ok' if shutil.which('uv') else 'missing'}; "
                             f"Blender {'ok' if shutil.which('blender') else 'not installed (only needed to model an object no library holds)'}; mujoco_playground {'ok' if iu.find_spec('mujoco_playground') else 'not installed (only needed for the gauntlet and real SO-101 / YAM arms)'}")
        if any(not r['ok'] for r in rows): print('\nMissing tools: run `ari-sim setup` or `npx skills add google-deepmind/robotics-antigravity-plugin`.'); sys.exit(1)
        return
    if a.cmd == 'measure':
        from .perceive import measure; print(json.dumps(measure(a.ref, a.point, a.pixel_sigma), indent=1)); return
    if a.cmd == 'detect':
        from .perceive import detect; print(json.dumps(detect(a.image, a.labels), indent=1)); return
    if a.cmd == 'audit' and a.scene.endswith('.xml'):
        from . import google; os.makedirs(a.out, exist_ok=True)
        r = google.run('mujoco-audit', [f'--xml_path={os.path.abspath(a.scene)}', f'--out_dir={os.path.abspath(a.out)}'] + ([f'--scenarios={a.scenarios}'] if a.scenarios else ['--static_only']))
        print((r.stdout or r.stderr)[-4000:]); sys.exit(r.returncode)
    if a.cmd == 'assets':
        from . import simready as SR
        if a.acmd == 'search':
            hits = SR.search(' '.join(a.query), a.n)
            if not hits: print('no match'); return
            if hits[0][1] == 'names': print('(search API unreachable: matching names in nvidia/simready-assets instead)')
            for h, _ in hits: print(f'{SR.download_mb(h):7.1f} MB  {h}')
            return
        if a.acmd == 'show': print(SR.thumbnail(a.id)); return
        m = SR.get(a.id, a.max_mb); print(json.dumps({k: m[k] for k in ('id', 'extent', 'mass', 'friction', 'triangles', 'warnings', 'attribution')}, indent=1)); return
    if a.cmd == 'robots':
        from . import assets; from .robots import ROBOTS, DEFAULT_CANDIDATES, menagerie_dir
        if a.rcmd == 'get':
            from . import simready as SR
            for k in a.names:
                if 'simready' in ROBOTS.get(k, {}): SR.fetch(ROBOTS[k]['simready'])
            assets.fetch_robots(a.names, gripper=False); print('ready:', ', '.join(a.names)); return
        if a.rcmd == 'inspect':
            import numpy as np; from . import simready as SR
            s, base, tip, hinges = SR.robot(a.id); m = s.compile(); st = m.site('attachment_site')
            print(f'{len(hinges)} hinge joints{"" if len(hinges) == 6 else "  (the planner needs exactly 6)"}; base {base}; flange on {tip}')
            for j in hinges: print(f'  {j.split("/")[-1]:24s} limits {np.degrees(m.joint(j).range).round(1).tolist()} deg')
            print(f'mass {m.body_mass.sum():.1f} kg; flange at {np.round(m.site_pos[st.id], 3).tolist()} from {tip.split("/")[-1]}')
            print('joint speeds, reach and payload are not in the file: take them from the datasheet (see robots.register)'); return
        for k, R in ROBOTS.items():
            src = f"Menagerie {menagerie_dir(k)}" if 'xml' in R else 'SimReady ' + R['simready'].split('/')[0]
            print(f"{k:10s} reach {R['reach']:.2f} m  payload {R['payload']:4.1f} kg  {src}{'  (default)' if k in DEFAULT_CANDIDATES else ''}")
        return
    from . import spec as S
    sc = S.load(a.scene)
    if a.cmd == 'check':
        import numpy as np; from PIL import Image; from . import build as B; from .robots import tool as tool_spec, register, DEFAULT_CANDIDATES
        R = S.resolve(sc); rob = sc.get('robot', {}); robot = register((rob.get('candidates') or DEFAULT_CANDIDATES)[-1]); bs = rob.get('base_search') or dict(x=[0, 0, 1], y=[-0.8, -0.8, 1], pedestal=[0.8, 0.8, 1])
        base = [sum(bs[k][:2]) / 2 for k in ('x', 'y', 'pedestal')]; m, dd, info = B.scene(robot, R, tool_spec(rob.get('tool')), base, name=sc['name']); v = B.scene_view(R); os.makedirs(a.out, exist_ok=True)
        Image.fromarray(np.hstack([B.render(m, dd, v, 960, 720, az=118), B.render(m, dd, v, 960, 720, az=62)])).save(os.path.join(a.out, 'scene.png'))
        Image.fromarray(np.hstack([B.render(m, dd, v, 960, 720, az=118, colliders=True), B.render(m, dd, v, 960, 720, az=62, colliders=True)])).save(os.path.join(a.out, 'colliders.png'))
        p = B.station(R, sc['name'], a.out)
        print(f"OK: {len(R['geoms'])} primitives, {len(R['picks'])} pick targets, {len(R['places'])} place targets -> {a.out}/scene.png, colliders.png (what the planner avoids), {p}"); return
    if a.cmd in ('audit', 'showcase'):
        from . import google, build as B; p = os.path.abspath(B.station(S.resolve(sc), sc['name'], a.out))
        if a.cmd == 'audit':
            r = google.run('mujoco-audit', [f'--xml_path={p}', '--static_only']); txt = r.stdout[r.stdout.find('{'):r.stdout.rfind('}') + 1]
            open(os.path.join(a.out, 'station_audit.json'), 'w').write(txt or '{}'); rep = json.loads(txt or '{}')
            print(f"station asset {p}\nstatic audit: {'passed' if rep.get('passed') else 'FAILED'}; {len(rep.get('errors', []))} errors, {len(rep.get('warnings', []))} warnings")
            for e in rep.get('errors', []) + rep.get('warnings', []): print('  -', e)
            if not txt: print(r.stderr[-2000:])
            sys.exit(0 if rep.get('passed') else 1)
        out = os.path.join(os.path.abspath(a.out), 'showcase', 'station_turntable.mp4')
        r = google.run('mujoco-showcase-shot', [f'--asset={p}', f'--output={out}', f'--duration={a.duration}']); print((r.stdout + r.stderr)[-600:]); sys.exit(r.returncode)
    if a.cmd == 'studio':
        from . import studio; ws = studio.workspace(a.scene, a.out); r = studio.launch(ws, test_steps=200 if a.test else 0)
        if a.test: print('Studio workspace', ws, 'loads' if r['ok'] else 'FAILED\n' + r['log']); sys.exit(0 if r['ok'] else 1)
        s_ = r['session']; print(f"MuJoCo Studio is running (pid {r['pid']}): open http://127.0.0.1:{s_.get('http_port', 8080)}/\n  workspace: {ws}\n  session:   {json.dumps({k: v for k, v in s_.items() if 'port' in k or 'url' in k})}\n  status:    http://127.0.0.1:{s_['control_port']}/_studio/status\n  log:       {r['log']}"); return
    if a.cmd == 'diagnose':
        from .diagnose import diagnose; from .report import write_all, VERDICT
        print('Running the diagnostic…'); res = diagnose(sc, n_variants=a.variants); write_all(sc, res, a.out, with_video=not a.no_video)
        print('\n' + VERDICT[res['verdict']]); print(f"Best: {res['best_robot']} at {res['best_fixed']['x']:.2f}, {res['best_fixed']['y']:.2f}, pedestal {res['best_fixed']['pedestal']:.2f} m "
              f"— {res['best_fixed']['reached']}/{res['n_targets']} targets"); print('Report in', a.out); return
    if a.cmd in ('execute', 'grasp'):
        from .execute import run_batch
        base = a.base; robot = a.robot; dj = os.path.join(os.path.dirname(os.path.abspath(a.scene)), 'out', 'diagnostic.json'); diag = json.load(open(dj)) if os.path.exists(dj) else None
        if base is None:
            if not diag: sys.exit('No --base given and no diagnostic.json found: run `ari-sim diagnose` first or pass --base X Y PEDESTAL.')
            bf = diag['best_fixed']; base = [bf['x'], bf['y'], bf['pedestal']]
        robot = robot or (diag or {}).get('best_robot') or 'UR10e'
        from .robots import tool as tool_spec
        if tool_spec(sc.get('robot', {}).get('tool'))['kind'] != 'fingers':
            print('note: this runs a 2-finger gripper; the scene\'s tool is suction, so these results do not apply to it')
        stats = run_batch(sc, base, a.episodes, a.out, robot=robot, seed=a.seed, perception=a.perception); print(json.dumps(stats, indent=1))

if __name__ == '__main__': main()
