"""The diagnostic: which robot, where to put it, what it reaches, how fast, and how sure we are.

Every candidate is a plan.Cell: Google's motion planner on the arm's own model, in the station rebuilt from the photos."""
import time, numpy as np
from . import spec as S
from .plan import Cell
from .robots import ROBOTS, DEFAULT_CANDIDATES, register, tool as tool_spec

def _evaluate(cell, resolved, seeds=None, restarts=5):
    """Per pick target: 'ok' | 'out_of_reach' | 'collision' (needs pre-pose and pick pose). Plus the place status.
    seeds: {target name: q} from a previous evaluation, which makes a nearby variant almost free to check."""
    places = resolved['places']; seeds = seeds or {}
    if not places: raise S.SceneError('scene has no place target (add a tote with "place": true or a place_point)')
    pl = places[0]; qdrop, s_pl = cell.reach(pl['drop'], pl['approach'], seeds.get('__drop'), restarts=max(restarts, 12)); qpp = None
    if qdrop is not None:
        qpp, s2 = cell.reach(pl['pre'], pl['approach'], qdrop, restarts=4); s_pl = 'ok' if qpp is not None else s2
    res = []; last = None; reachmax = ROBOTS[cell.robot]['reach'] + float(cell.tool['length']) + 0.25
    for t in resolved['picks']:
        if np.linalg.norm(t['pick'] - cell.base) > reachmax: res.append(dict(status='out_of_reach')); continue
        seed = seeds.get(t['name'], last)
        qpre, s1 = cell.reach(t['pre'], t['approach'], seed, restarts)
        if qpre is None: res.append(dict(status=s1)); continue
        qpick, s2 = cell.reach(t['pick'], t['approach'], qpre, restarts=3)
        if qpick is None: res.append(dict(status=s2)); continue
        last = qpre; res.append(dict(status='ok', qpre=qpre, qpick=qpick))
    return res, s_pl, (qpp, qdrop)

def _seeds(resolved, res, place):
    s = {t['name']: r['qpre'] for t, r in zip(resolved['picks'], res) if r['status'] == 'ok'}
    if place[1] is not None: s['__drop'] = place[1]
    return s

def _grid(spec_range):
    lo, hi, n = spec_range; return [float(v) for v in np.linspace(lo, hi, int(n))]

def diagnose(sc, n_variants=40, seed=1, log=print):
    rng = np.random.default_rng(seed); rob = sc.get('robot', {}); tool = tool_spec(rob.get('tool')); bs = rob.get('base_search')
    if not bs: raise S.SceneError('scene["robot"]["base_search"] is required: {"x": [lo, hi, n], "y": [lo, hi, n], "pedestal": [lo, hi, n]}')
    XS, YS, ZS = _grid(bs['x']), _grid(bs['y']), _grid(bs['pedestal']); cands = [register(c) for c in rob.get('candidates', DEFAULT_CANDIDATES)]
    R0 = S.resolve(sc); n = len(R0['picks']); t0 = time.time()
    if n == 0: raise S.SceneError('scene has no pick target (shelf_unit with "pick": true, or pick_point)')
    # 1. every robot at every base position, nominal geometry
    grid = {}; cells = {}
    for robot in cands:
        cell = cells[robot] = Cell(robot, tool); rows = []
        for x in XS:
            for y in YS:
                for z in ZS:
                    res, pl, _ = _evaluate(cell.place(R0, x, y, z), R0)
                    rows.append(dict(x=x, y=y, z=z, place=pl, status=[r['status'] for r in res]))
        grid[robot] = rows; log(f'  {robot}: {len(rows)} base positions tested ({time.time() - t0:.0f}s)')
    score = lambda g: sum(s == 'ok' for s in g['status']) if g['place'] == 'ok' else 0
    summary = {}
    for robot, rows in grid.items():
        b = max(rows, key=score)
        summary[robot] = dict(best_base=dict(x=b['x'], y=b['y'], pedestal=b['z']), reached=score(b), n=n,
                              place_reachable_positions=sum(r['place'] == 'ok' for r in rows), positions=len(rows),
                              best_ignoring_place=max(sum(s == 'ok' for s in r['status']) for r in rows))
    best_robot = max(summary, key=lambda r: summary[r]['reached']); rows = grid[best_robot]; bf = max(rows, key=score)
    lift = {}
    for g in rows:
        k = (g['x'], g['y']); lift.setdefault(k, set())
        if g['place'] == 'ok': lift[k] |= {i for i, s in enumerate(g['status']) if s == 'ok'}
    lxy = max(lift, key=lambda k: len(lift[k])); cell = cells[best_robot]
    out = dict(scene=sc['name'], n_targets=n, robots=summary, best_robot=best_robot, best_fixed=dict(x=bf['x'], y=bf['y'], pedestal=bf['z'], reached=score(bf), status=bf['status']),
               best_lift=dict(x=lxy[0], y=lxy[1], heights=ZS, reached=len(lift[lxy])), targets=[t['name'] for t in R0['picks']],
               cells=[(t.get('col'), t.get('row')) for t in R0['picks']], tool=cell.tool, dof=cell.pl.dof,
               planner='Google robotics plugin, motion-planning skill (motion_planner.py)')
    if out['best_fixed']['reached'] == 0:
        out['verdict'] = 'no_feasible_layout'; return out
    # 2. full collision-free cycle + time for every reachable target
    cell.place(R0, bf['x'], bf['y'], bf['z']); res, pl, (qpp, qdrop) = _evaluate(cell, R0, restarts=10); nominal = _seeds(R0, res, (qpp, qdrop)); cyc = []
    down = cell.stroke(qpp, R0['places'][0]['drop']) if pl == 'ok' else (None, 'place')
    prev = None
    for t, r in zip(R0['picks'], res):
        if r['status'] != 'ok' or pl != 'ok': cyc.append(dict(name=t['name'], ok=False, why=r['status'] if r['status'] != 'ok' else 'place')); continue
        done = None
        for qpre in cell.reach_near(t['pre'], t['approach'], qpp, (r['qpre'], prev))[:2]:      # the posture closest to the drop first
            ins = cell.stroke(qpre, t['pick'])
            if ins[0] is None: continue
            path = cell.move(qpp, qpre)
            if path[0] is not None: done = (path, ins, qpre); break
        if done is None or down[0] is None: cyc.append(dict(name=t['name'], ok=False, why='path_blocked')); continue
        path, ins, prev = done; moves = [path[0], ins[0], ins[0][::-1], path[0][::-1], down[0], down[0][::-1]]
        cyc.append(dict(name=t['name'], ok=True, t=2 * (path[1] + ins[1] + down[1]) + 0.7, moves=[mv.tolist() for mv in moves], pick=t['pick'].tolist()))
    ts = [c['t'] for c in cyc if c['ok']]
    out['cycle'] = dict(n_full_cycle=len(ts), mean_s=float(np.mean(ts)) if ts else None, min_s=float(min(ts)) if ts else None, max_s=float(max(ts)) if ts else None, per_target=cyc, dt=0.04)
    log(f'  cycle planning done: {len(ts)}/{n} targets with a full collision-free cycle ({time.time() - t0:.0f}s)')
    # 3. Monte Carlo over what the photos cannot pin down
    mc = []
    for i in range(n_variants):
        vals = S.sample(sc, rng); Ri = S.resolve(sc, vals); nb = max(len(Ri['picks']), 1)
        res, plf, _ = _evaluate(cell.place(Ri, bf['x'], bf['y'], bf['z']), Ri, nominal, restarts=3)
        fx = sum(r['status'] == 'ok' for r in res) / nb if plf == 'ok' else 0.0
        un = set(); anyplace = False
        for z in ZS:
            res, pl2, _ = _evaluate(cell.place(Ri, lxy[0], lxy[1], z), Ri, nominal, restarts=3)
            if pl2 == 'ok': anyplace = True; un |= {j for j, r in enumerate(res) if r['status'] == 'ok'}
        mc.append(dict(values=vals, fixed=fx, fixed_place=plf == 'ok', lift=len(un) / nb, lift_place=anyplace))
    keys = [k for k, p in sc['params'].items() if p.get('sigma', 0) > 0]
    sens = {}
    if keys and len(mc) > 5:
        X = np.array([[c['values'][k] for k in keys] for c in mc])
        for nm in ('fixed', 'lift'):
            y = np.array([c[nm] for c in mc])
            if y.std() > 1e-9: sens[nm] = sorted(((k, float(np.corrcoef(X[:, i], y)[0, 1])) for i, k in enumerate(keys)), key=lambda kv: -abs(kv[1]))[:4]
    fx = np.array([c['fixed'] for c in mc]); lf = np.array([c['lift'] for c in mc])
    out['uncertainty'] = dict(n=len(mc), fixed=dict(place_fail=int(sum(not c['fixed_place'] for c in mc)), ge90=int((fx >= 0.9).sum()), ge80=int((fx >= 0.8).sum()), median=float(np.median(fx)) if len(mc) else 0.0),
                              lift=dict(place_fail=int(sum(not c['lift_place'] for c in mc)), ge90=int((lf >= 0.9).sum()), ge80=int((lf >= 0.8).sum()), median=float(np.median(lf)) if len(mc) else 0.0),
                              sensitivity=sens, samples=[dict(fixed=c['fixed'], lift=c['lift']) for c in mc])
    log(f'  uncertainty run done: {len(mc)} variants ({time.time() - t0:.0f}s)')
    # 4. verdict on reach only — grasp reliability is NOT assessed here
    f = out['uncertainty']['fixed']; share = out['cycle']['n_full_cycle'] / n
    if share >= 0.75 and f['place_fail'] <= 0.1 * len(mc): v = 'reach_ok'
    elif share >= 0.5: v = 'reach_ok_if_measured'
    else: v = 'reach_marginal'
    out['verdict'] = v
    return out
