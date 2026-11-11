"""Scene description: a small parametric language for workstations.

A scene is JSON. Every number may be an expression over named `params`, and every param carries a
1-sigma uncertainty — that is what lets the diagnostic say which dimension to measure on site.
`components` come from a short library (box, cylinder, shelf_unit, tote, pick_point, place_point).

Frame convention: origin on the floor, z up, metres. The operator stands on the -y side and works towards +y.
"""
import ast, json, math, operator as op
import numpy as np

COLORS = dict(black=[0.07, 0.07, 0.07, 1], yellow=[0.95, 0.80, 0.15, 1], steel=[0.62, 0.64, 0.66, 1], galvanised=[0.5, 0.53, 0.55, 1],
              red=[0.85, 0.13, 0.10, 1], white=[0.9, 0.9, 0.95, 1], grey=[0.3, 0.3, 0.32, 1], blue=[0.18, 0.33, 1.0, 1],
              green=[0.2, 0.6, 0.3, 1], wood=[0.72, 0.56, 0.38, 1], orange=[0.95, 0.5, 0.1, 1])
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg, ast.UAdd: op.pos, ast.Pow: op.pow}
_FUNCS = dict(min=min, max=max, abs=abs, sqrt=math.sqrt)

class SceneError(ValueError): pass

def _eval(node, env):
    if isinstance(node, ast.Expression): return _eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env: raise SceneError(f"unknown param '{node.id}' (declare it under \"params\")")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS: return _OPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS: return _OPS[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS: return float(_FUNCS[node.func.id](*[_eval(a, env) for a in node.args]))
    raise SceneError('unsupported expression')

def num(x, env):
    """A number, or an arithmetic expression over params such as "-(open_w/2 + 0.08)"."""
    if isinstance(x, bool): raise SceneError('booleans are not numbers')
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        try: return _eval(ast.parse(x, mode='eval'), env)
        except SyntaxError as e: raise SceneError(f'bad expression {x!r}: {e.msg}')
    raise SceneError(f'expected a number or expression, got {x!r}')
def vec(v, env, n=3):
    if not isinstance(v, (list, tuple)) or len(v) != n: raise SceneError(f'expected a list of {n} numbers, got {v!r}')
    return np.array([num(a, env) for a in v])

def load(path_or_dict):
    sc = json.load(open(path_or_dict)) if isinstance(path_or_dict, str) else path_or_dict
    for k in ('name', 'components'):
        if k not in sc: raise SceneError(f'scene is missing "{k}"')
    sc.setdefault('params', {}); sc.setdefault('task', {}); sc.setdefault('robot', {})
    for k, p in sc['params'].items():
        if not isinstance(p, dict) or 'value' not in p: raise SceneError(f'param {k}: expected {{"value": .., "sigma": ..}}')
    resolve(sc)          # fail early on a bad scene
    return sc

def nominal(sc): return {k: float(p['value']) for k, p in sc['params'].items()}
def sample(sc, rng):
    """One plausible variant of the workstation, drawn from the per-param uncertainty."""
    out = {}
    for k, p in sc['params'].items():
        v = rng.normal(p['value'], p.get('sigma', 0.0))
        out[k] = float(np.clip(v, p.get('min', -np.inf), p.get('max', np.inf)))
    return out

def _rot(yaw_deg):
    a = math.radians(yaw_deg); return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

def resolve(sc, values=None):
    """Expand components into primitive geoms + pick/place targets for one set of param values."""
    env = nominal(sc) if values is None else values
    task = sc.get('task', {}); inset = float(task.get('pick_inset', 0.20)); pre_off = float(task.get('pre_offset', 0.12))
    geoms, picks, places, names = [], [], [], set()
    def add(name, shape, pos, size, color, collide=True, yaw=0.0):
        if name in names: raise SceneError(f'duplicate name {name!r}')
        names.add(name)
        rgba = COLORS.get(color, color) if isinstance(color, str) else color
        if not (isinstance(rgba, (list, tuple)) and len(rgba) == 4): raise SceneError(f'{name}: unknown color {color!r}; use one of {sorted(COLORS)} or [r,g,b,a]')
        if min(size) <= 0: raise SceneError(f'{name}: sizes must be positive, got {list(size)}')
        geoms.append(dict(name=name, shape=shape, pos=np.asarray(pos, float), size=np.asarray(size, float), rgba=list(rgba), collide=collide, yaw=float(yaw)))
    for i, c in enumerate(sc['components']):
        t = c.get('type'); name = c.get('name', f'{t}{i}'); col = c.get('color', 'grey'); yaw = num(c.get('yaw', 0), env); R = _rot(yaw)
        try:
            if t in ('box', 'cylinder'):
                size = vec(c['size'], env, 3 if t == 'box' else 2)
                add(name, t, vec(c['pos'], env), size, col, c.get('collide', True), yaw)
            elif t == 'shelf_unit':
                # origin = floor point at the centre of the OPEN face; cells open towards -y (rotated by yaw)
                o = vec(c['origin'], env); w = num(c['width'], env); h = num(c['height'], env); depth = num(c.get('depth', 1.0), env)
                bd = num(c.get('bin_depth', depth), env); ncol = int(c.get('ncol', 4)); pitch = num(c['pitch'], env); z0 = num(c.get('z0', 0.2), env)
                nrow = int((h - z0) // pitch)
                if nrow < 1 or ncol < 1: raise SceneError(f'{name}: no cells (height {h:.2f}, z0 {z0:.2f}, pitch {pitch:.2f})')
                L = lambda p: o + R @ np.array(p)
                if depth - bd > 0.01: add(f'{name}_back', 'box', L([0, bd + (depth - bd) / 2, h / 2]), [w, depth - bd, h], col, True, yaw)
                else: add(f'{name}_back', 'box', L([0, bd + 0.005, h / 2]), [w, 0.01, h], col, True, yaw)
                for s, nm in ((-1, 'L'), (1, 'R')): add(f'{name}_side{nm}', 'box', L([s * w / 2, bd / 2, h / 2]), [0.016, bd, h], col, True, yaw)
                for k in range(1, ncol): add(f'{name}_div{k}', 'box', L([w * (k / ncol - 0.5), bd / 2, z0 + nrow * pitch / 2]), [0.008, bd, nrow * pitch], col, True, yaw)
                for r in range(nrow + 1): add(f'{name}_shelf{r}', 'box', L([0, bd / 2, z0 + r * pitch]), [w, bd, 0.012], col, True, yaw)
                if c.get('pick', True):
                    a = R @ np.array([0, 1.0, 0])
                    for r in range(nrow):
                        for k in range(ncol):
                            x = w * (k - (ncol - 1) / 2) / ncol; z = z0 + (r + 0.5) * pitch
                            picks.append(dict(name=f'{name}_c{k + 1}r{r + 1}', col=k, row=r, ncol=ncol, nrow=nrow, approach=a, face=L([x, 0, z]),
                                              pick=L([x, inset, z]), pre=L([x, -pre_off, z]), support_z=o[2] + z0 + r * pitch + 0.006, top_z=o[2] + z0 + (r + 1) * pitch - 0.006))
            elif t == 'tote':
                cx, cy = vec(c['center'], env, 2); rim = num(c['rim'], env); sx, sy, sz = vec(c.get('size', [0.6, 0.4, 0.32]), env)
                L = lambda p: np.array([cx, cy, 0]) + R @ np.array(p)
                add(f'{name}_bottom', 'box', L([0, 0, rim - sz + 0.01]), [sx, sy, 0.02], col, True, yaw)
                for s in (-1, 1):
                    add(f'{name}_x{s}', 'box', L([s * sx / 2, 0, rim - sz / 2]), [0.016, sy, sz], col, True, yaw)
                    add(f'{name}_y{s}', 'box', L([0, s * sy / 2, rim - sz / 2]), [sx, 0.016, sz], col, True, yaw)
                if c.get('place', True):
                    places.append(dict(name=name, drop=np.array([cx, cy, rim + 0.08]), pre=np.array([cx, cy, rim + 0.20]), approach=np.array([0, 0, -1.0]),
                                       box=dict(center=[cx, cy], half=[sx / 2, sy / 2], rim=rim, depth=sz)))
            elif t == 'pick_point':
                p = vec(c['pos'], env); a = vec(c.get('approach', [0, 0, -1]), env); a = a / np.linalg.norm(a)
                picks.append(dict(name=name, approach=a, face=p - a * inset, pick=p, pre=p - a * (inset + pre_off), support_z=p[2] - 0.02, top_z=p[2] + 0.3))
            elif t == 'place_point':
                p = vec(c['pos'], env); a = vec(c.get('approach', [0, 0, -1]), env); a = a / np.linalg.norm(a)
                places.append(dict(name=name, drop=p, pre=p - a * 0.12, approach=a, box=None))
            else:
                raise SceneError(f'unknown component type {t!r}; available: box, cylinder, shelf_unit, tote, pick_point, place_point')
        except KeyError as e:
            raise SceneError(f'component {name!r} ({t}) is missing field {e}')
    return dict(geoms=geoms, picks=picks, places=places, values=env)
