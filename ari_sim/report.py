"""Turn a diagnostic into things a person can look at: a summary image, a cycle video, a Markdown report."""
import os, json, subprocess, numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
from . import spec as S, assets
from .build import build, set_base, scene_view, render
from .kin import move_time, along
from .robots import ROBOTS, TOOLS

INK = (17, 17, 17); BLUE = (47, 84, 255); MUT = (95, 97, 94); LINE = (212, 211, 204); SIL = (184, 187, 184); PAPER = (242, 241, 237)
def _font(size, bold=False, mono=False):
    for p in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'Mono' if mono else ''}{'-Bold' if bold else ''}.ttf",
              '/System/Library/Fonts/Helvetica.ttc', 'C:/Windows/Fonts/arial.ttf'):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()
WHY = dict(out_of_reach='out of reach', collision='collision', path_blocked='path blocked', place='drop unreachable')

def figure(sc, res, path):
    W, H = 1600, 900; im = Image.new('RGB', (W, H), PAPER); d = ImageDraw.Draw(im); bf = res['best_fixed']
    d.text((60, 44), f"{sc['name']} — workstation diagnostic", fill=INK, font=_font(38, True))
    d.text((60, 100), f"{res['best_robot']} on a {bf['pedestal']:.1f} m pedestal · scene rebuilt from photos · simulation, not a measurement", fill=MUT, font=_font(19, mono=True))
    cyc = {c['name']: c for c in res.get('cycle', {}).get('per_target', [])}; cells = res['cells']; names = res['targets']
    d.text((60, 170), 'PICK TARGETS', fill=BLUE, font=_font(18, mono=True))
    if all(c[0] is not None for c in cells):
        ncol = max(c[0] for c in cells) + 1; nrow = max(c[1] for c in cells) + 1; cw = min(120, 520 // ncol); ch = min(70, 600 // nrow)
        for nm, (c, r) in zip(names, cells):
            X = 60 + c * (cw + 8); Y = 210 + (nrow - 1 - r) * (ch + 8); cy = cyc.get(nm, {})
            if cy.get('ok'): col, lab, tc = BLUE, f"{cy['t']:.1f} s", (255, 255, 255)
            else: col, lab, tc = (SIL if cy.get('why') == 'path_blocked' else PAPER), WHY.get(cy.get('why'), '—'), (INK if cy.get('why') == 'path_blocked' else MUT)
            d.rectangle([X, Y, X + cw, Y + ch], fill=col, outline=LINE if col == PAPER else col, width=2)
            d.text((X + cw / 2, Y + ch / 2), lab, fill=tc, font=_font(15), anchor='mm')
        ybot = 210 + nrow * (ch + 8) + 10
    else:
        for i, nm in enumerate(names[:24]):
            cy = cyc.get(nm, {}); lab = f"{cy['t']:.1f} s" if cy.get('ok') else WHY.get(cy.get('why'), '—')
            d.text((60, 210 + i * 26), f"{nm[:28]:<30}{lab}", fill=BLUE if cy.get('ok') else MUT, font=_font(17, mono=True))
        ybot = 210 + min(len(names), 24) * 26 + 10
    c = res.get('cycle', {})
    if c.get('mean_s'): d.text((60, ybot), f"{c['n_full_cycle']} of {res['n_targets']} targets with a full collision-free cycle · mean {c['mean_s']:.1f} s", fill=INK, font=_font(20, True))
    u = res.get('uncertainty'); X1 = 880
    if u:
        d.text((X1, 170), f"{u['n']} PLAUSIBLE VARIANTS OF THE WORKSTATION (PHOTO UNCERTAINTY)", fill=BLUE, font=_font(18, mono=True))
        def bars(key, ytop, title):
            vals = [s[key] for s in u['samples']]; d.text((X1, ytop), title, fill=INK, font=_font(20, True))
            edges = [0, 0.01, 0.5, 0.7, 0.8, 0.9, 1.01]; labs = ['drop fails', '< 50 %', '50–70 %', '70–80 %', '80–90 %', '≥ 90 %']
            for i, lab in enumerate(labs):
                n = sum(1 for v in vals if edges[i] <= v < edges[i + 1]); Y = ytop + 40 + i * 34
                d.text((X1, Y + 12), lab, fill=MUT, font=_font(16), anchor='lm')
                d.rectangle([X1 + 150, Y, X1 + 150 + n * 13, Y + 24], fill=(BLUE if i == 5 else (INK if i == 0 else SIL)))
                d.text((X1 + 160 + n * 13, Y + 12), str(n), fill=INK, font=_font(16, True), anchor='lm')
        bars('fixed', 215, 'Fixed pedestal'); bars('lift', 490, 'With a vertical axis')
        d.text((X1, 770), 'Share of targets reached, drop included.', fill=MUT, font=_font(16))
        s = u['sensitivity'].get('fixed') or u['sensitivity'].get('lift')
        if s: d.text((X1, 800), 'Measure on site first: ' + ', '.join(k for k, _ in s[:2]), fill=INK, font=_font(18, True))
    im.save(path)

def video(sc, res, path, n_show=3, w=1280, h=720, fps=30):
    """Real-time playback of full cycles at joint speed limits (an upper bound on real speed)."""
    R0 = S.resolve(sc); bf = res['best_fixed']; robot = res['best_robot']; m, d, info = build(robot, R0, res['tool']); set_base(m, d, info, bf['x'], bf['y'], bf['pedestal'])
    good = [c for c in res['cycle']['per_target'] if c['ok']]
    if not good: return False
    idx = np.unique(np.linspace(0, len(good) - 1, n_show).astype(int)); sel = [good[i] for i in idx]
    view = scene_view(R0); ren = mujoco.Renderer(m, h, w); cam = mujoco.MjvCamera(); cam.lookat[:] = view['lookat']; cam.distance = view['dist']; cam.azimuth = 112; cam.elevation = -14
    fb, fs = _font(30, True), _font(20, mono=True); tmp = path.replace('.mp4', '.part.mp4'); site = info['tip_site']; vmax = ROBOTS[robot]['vmax']
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                           '-c:v', 'libx264', '-crf', '19', '-preset', 'medium', '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
    pl = R0['places'][0]; n_drop = 0
    def frame(label, sub):
        mujoco.mj_forward(m, d); ren.update_scene(d, cam); im = Image.fromarray(ren.render()); dr = ImageDraw.Draw(im)
        dr.text((40, 34), f"{sc['name']} · workstation diagnostic", fill=INK, font=fb)
        dr.text((40, 78), f"{robot}, {bf['pedestal']:.1f} m pedestal · scene rebuilt from photos · simulation, not a measurement", fill=MUT, font=fs)
        dr.text((40, h - 76), label, fill=BLUE, font=fb); dr.text((40, h - 38), sub, fill=MUT, font=fs); ff.stdin.write(im.tobytes())
    d.qpos[:6] = sel[0]['moves'][0][0]; d.mocap_pos[info['item']] = [0, 0, -1]
    for _ in range(15): frame('Pick and place', '')
    for c in sel:
        lab = f"{c['name']}  ·  cycle ≈ {c['t']:.1f} s"; carried = False
        d.mocap_pos[info['item']] = np.array(c['pick']) - np.array([0, 0, 0.05])
        for mi, mv in enumerate(c['moves']):
            T = max(move_time(mv, vmax), 0.2); nf = max(int(round(T * fps)), 2)
            for k in range(1, nf + 1):
                s = k / nf; s = s * s * (3 - 2 * s); d.qpos[:6] = along(mv, s); mujoco.mj_forward(m, d)
                if carried: d.mocap_pos[info['item']] = d.site_xpos[site] + d.site_xmat[site].reshape(3, 3)[:, 2] * 0.03
                frame(lab, 'collision-free path, joint speed limits')
            if mi == 1: carried = True; [frame(lab, 'grip') for _ in range(int(0.4 * fps))]
            if mi == 4:
                carried = False; d.mocap_pos[info['item']] = pl['drop'] - np.array([0.14 - 0.14 * n_drop, 0, 0.35]); n_drop += 1
                [frame(lab, 'release') for _ in range(int(0.3 * fps))]
    for _ in range(25): frame('Pick and place', '')
    ff.stdin.close(); rc = ff.wait(); ren.close()
    if rc == 0: os.replace(tmp, path)
    return rc == 0

VERDICT = dict(reach_ok='Reach is validated for a pilot.', reach_ok_if_measured='Reach works on the nominal geometry but depends on dimensions the photos do not pin down: measure before deciding.',
               reach_marginal='Reach is marginal: fewer than half of the targets get a full collision-free cycle.', no_feasible_layout='No tested robot and base position reaches both the pick targets and the drop.')

def _menagerie_licence(folder):
    p = os.path.join(assets.MENAGERIE, folder, 'LICENSE')
    t = open(p).read(4000) if os.path.exists(p) else ''
    if 'Redistribution and use in source and binary forms' in t: return 'BSD-3-Clause' if 'Neither the name' in t else 'BSD-2-Clause'
    for key, name in (('Apache License', 'Apache-2.0'), ('MIT License', 'MIT'), ('Creative Commons', 'CC-BY')):
        if key in t: return name
    return 'see LICENSE in the model folder'

def tooling(sc, res):
    """The end effector, why it was chosen, and whether tool + part fit each arm's payload and the gripper's stroke."""
    t = res['tool']; L = ['', '## Tool', '', f"**{t['type']}** ({t['kind']}): {TOOLS[t['type']]['note']}."]
    if t.get('why'): L += ['', f"Why this tool: {t['why']}"]
    items = [c for c in sc['components'] if c.get('type') == 'asset' and c.get('role') == 'item']
    metas = []
    if items:
        from . import simready; metas = [simready.get(c['source'].split(':', 1)[1]) for c in items]
    part = sc.get('task', {}).get('item_mass_kg')
    part = float(part) if part is not None else max((m['mass'] or 0 for m in metas), default=None)
    if part is not None:
        L += ['', f"Payload with the heaviest part ({part:.2f} kg) and the tool ({t['mass']:.2f} kg):", '']
        for r in res['robots']:
            need = part + t['mass']; cap = ROBOTS[r]['payload']
            L.append(f"- {r}: {need:.2f} of {cap:.1f} kg" + ('' if need <= cap else ' — **over payload**'))
        if t['kind'] == 'fingers' and part > t['payload']: L.append(f"- the gripper itself is rated {t['payload']:.1f} kg: **over the gripper's payload**")
    else: L += ['', 'Part mass not given (`task.item_mass_kg`): payload not checked.']
    if t['kind'] == 'fingers' and metas:
        wide = [m['name'] for m in metas if sorted(m['extent'])[1] > t['stroke']]
        L += ['', f"Parts wider than the {t['stroke'] * 1000:.0f} mm stroke on every side but one: {', '.join(wide)}." if wide else f"Every part fits the {t['stroke'] * 1000:.0f} mm stroke on at least one side."]
    return L

def provenance(sc, res):
    """Where every robot model and real object came from, its licence, and the attribution it requires."""
    from .robots import menagerie_dir
    L = ['', '## Models and assets used', '', '| Robot | Model | Licence | Reach, payload, joint speeds |', '|---|---|---|---|']
    for r in res['robots']:
        R = ROBOTS[r]; spec = R.get('datasheet', 'datasheet values kept in `ari_sim/robots.py`')
        if 'xml' in R: src, lic = f"MuJoCo Menagerie `{menagerie_dir(r)}` at `{assets.MENAGERIE_COMMIT[:7]}`", _menagerie_licence(menagerie_dir(r))
        else:
            from . import simready; a = simready.parse(R['simready']); src, lic = f"SimReady `{a['repo']}/{a['folder']}` at `{a['rev'][:7]}`", simready.license(R['simready'])
        L.append(f'| {r} | {src} | {lic} | {spec} |')
    objs = [c for c in sc['components'] if c.get('type') == 'asset']
    if objs:
        from . import simready
        L += ['', '| Object | Role | SimReady asset | Licence |', '|---|---|---|---|']
        metas = []
        for c in objs:
            m = simready.get(c['source'].split(':', 1)[1]); metas.append(m)
            rid = m['id'] if '@' in m['id'] else f"{simready.REPO}/{m['id']}@{simready.REVISION[:7]}"
            L.append(f"| {c.get('name', '')} | {c.get('role', 'fixture')} | `{rid}` | {m['license']} |")
        L += ['', 'Attribution:', ''] + [f"- {x}" for x in dict.fromkeys(m['attribution'] for m in metas)]
    return L

def markdown(sc, res, path):
    L = [f"# {sc['name']} — workstation diagnostic", '', sc.get('description', ''), '', f"**Verdict (reach only): {VERDICT[res['verdict']]}**", '',
         '> Grasp reliability is not assessed by this diagnostic. It checks that the robot can reach, carry and drop — not that it can grip a soft or cluttered item.', '',
         '## Robots', '', '| Robot | Targets reached (drop included) | Best base (x, y, pedestal) | Base positions where the drop is reachable |', '|---|---|---|---|']
    for r, s in res['robots'].items():
        b = s['best_base']; L.append(f"| {r} | {s['reached']} / {s['n']} | {b['x']:.2f}, {b['y']:.2f}, {b['pedestal']:.2f} m | {s['place_reachable_positions']} / {s['positions']} |")
    c = res.get('cycle')
    if c and c.get('mean_s'):
        L += ['', '## Cycle', '', f"- Full collision-free cycle for **{c['n_full_cycle']} of {res['n_targets']}** targets with {res['best_robot']}.",
              f"- Cycle time at joint speed limits: mean {c['mean_s']:.1f} s (min {c['min_s']:.1f}, max {c['max_s']:.1f}). This is a ceiling, not a production rate."]
    u = res.get('uncertainty')
    if u:
        f, l = u['fixed'], u['lift']
        L += ['', '## How sure are we', '', f"The scene was re-simulated on {u['n']} plausible variants drawn from the uncertainty on each dimension.", '',
              '| Layout | Drop unreachable | ≥ 80 % of targets | ≥ 90 % of targets |', '|---|---|---|---|',
              f"| Fixed pedestal | {f['place_fail']} / {u['n']} | {f['ge80']} | {f['ge90']} |", f"| Vertical axis | {l['place_fail']} / {u['n']} | {l['ge80']} | {l['ge90']} |"]
        s = u['sensitivity'].get('fixed') or u['sensitivity'].get('lift')
        if s:
            L += ['', '**Measure on site first** (dimensions that move the result most):', '']
            for k, r in s: L.append(f"- `{k}` — {sc['params'][k].get('note', '')} (currently {sc['params'][k]['value']} ± {sc['params'][k].get('sigma', 0)} m, correlation {r:+.2f})")
    L += tooling(sc, res)
    L += provenance(sc, res)
    if sc.get('scale_references'): L += ['', '## Scale references used', ''] + [f'- {x}' for x in sc['scale_references']]
    if sc.get('assumptions'): L += ['', '## Assumptions', ''] + [f'- {x}' for x in sc['assumptions']]
    open(path, 'w').write('\n'.join(L) + '\n')

def write_all(sc, res, outdir, with_video=True):
    os.makedirs(outdir, exist_ok=True)
    slim = json.loads(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    for c in slim.get('cycle', {}).get('per_target', []): c.pop('moves', None)
    json.dump(slim, open(os.path.join(outdir, 'diagnostic.json'), 'w'), indent=1)
    markdown(sc, res, os.path.join(outdir, 'report.md')); figure(sc, res, os.path.join(outdir, 'diagnostic.png'))
    R0 = S.resolve(sc); robot = res['best_robot']; m, d, info = build(robot, R0, res['tool']); bf = res['best_fixed']; set_base(m, d, info, bf['x'], bf['y'], bf['pedestal'])
    v = scene_view(R0); Image.fromarray(np.hstack([render(m, d, v, 960, 720, az=118), render(m, d, v, 960, 720, az=62)])).save(os.path.join(outdir, 'scene.png'))
    if with_video and res.get('cycle', {}).get('n_full_cycle'): video(sc, res, os.path.join(outdir, 'cycle.mp4'))
