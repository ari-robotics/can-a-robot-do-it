import numpy as np
# Reach/payload from the manufacturers' datasheets; joint speed limits in rad/s. Six-axis arms only: the planner
# assumes six joints. A model comes either from MuJoCo Menagerie (`xml`) or from a SimReady robot asset published by
# its manufacturer (`simready`, read by MuJoCo's OpenUSD importer); both are downloaded only when a scene uses them.
# SimReady files carry joint limits but not joint speeds, reach or payload: those always come from a datasheet.
ROBOTS = {
    'UR5e':    dict(xml='universal_robots_ur5e/ur5e.xml',   base='base',      tip='wrist_3_link', reach=0.85,  payload=5.0,  vmax=np.radians([180, 180, 180, 180, 180, 180])),
    'UR10e':   dict(xml='universal_robots_ur10e/ur10e.xml', base='base',      tip='wrist_3_link', reach=1.30,  payload=12.5, vmax=np.radians([120, 120, 180, 180, 180, 180])),
    'CRX-5iA': dict(xml='fanuc_crx5ia/crx5ia.xml',          base='base_link', tip='J6_link',      reach=0.994, payload=5.0,  vmax=np.radians([240, 240, 240, 250, 250, 250])),
    'Lite6':   dict(xml='ufactory_lite6/lite6.xml',         base='link_base', tip='link6',        reach=0.44,  payload=0.6,  vmax=np.radians([180, 180, 180, 180, 180, 180]),
                    datasheet='https://docs.lite6.ufactory.cc/9.technical_specifications.html'),
    'CRX-10iA': dict(simready='FANUC-CORPORATION/SimReady-Assets/crx10ia/simready_usd/crx10ia.usd@449b304714f21f81e552423fd4f6ea3a821c94c2',
                     reach=1.249, payload=10.0, vmax=np.radians([120, 120, 180, 180, 180, 180]),
                     datasheet='https://crx.fanucamerica.com/crx-10ia (reach, payload); https://flowntro.com/robots/crx-10ia (joint speeds)'),
}
DEFAULT_CANDIDATES = ['UR5e', 'UR10e', 'CRX-5iA']     # tried when a scene names none; fetched by `ari-sim setup`
GRIPPER_XML = 'robotiq_2f85/2f85.xml'

# End effectors the diagnostic can simulate. `length` is flange to contact point (reach and clearance), `mass` counts
# against the arm's payload (typical values for the suction tools, which are generic, not one product). The choice
# depends on what is picked: see the skill's step on choosing the tool.
TOOLS = {
    'suction_wand':  dict(kind='suction', length=0.38, mass=0.6, note='long vacuum wand: reaches deep into cells and bins; needs a flat, sealed face'),
    'suction_cup':   dict(kind='suction', length=0.12, mass=0.4, note='short vacuum cup on the flange: cartons, cases, panels picked from above or the side'),
    'gripper_2f85':  dict(kind='fingers', length=0.15, mass=0.93, stroke=0.085, payload=5.0, note='Robotiq 2F-85 parallel gripper: parts, tools, bottles up to 85 mm wide; needs room for the fingers',
                          datasheet='https://blog.robotiq.com/hubfs/Product-sheets/Adaptive%20Grippers/Product-sheet-Adaptive-Grippers-EN.pdf'),
}

def tool(spec):
    """The scene's robot.tool with catalog defaults filled in; rejects tools the simulator does not have."""
    spec = dict(spec or dict(type='suction_wand'))
    if spec.get('type') not in TOOLS: raise KeyError(f"unknown tool {spec.get('type')!r}; available: {sorted(TOOLS)}")
    return {**{k: v for k, v in TOOLS[spec['type']].items() if k != 'note'}, **spec}

def menagerie_dir(robot): return ROBOTS[robot]['xml'].split('/')[0] if 'xml' in ROBOTS[robot] else None

def register(entry):
    """A candidate written in the scene: a preset name, or {"name", "source": "simready:<id>", "reach", "payload",
    "vmax_deg": [6 values], "datasheet": url}. The agent finds the arm with `ari-sim assets search` and the numbers in
    the manufacturer's datasheet; nothing here is guessed."""
    if isinstance(entry, str):
        if entry not in ROBOTS: raise KeyError(f'unknown robot {entry!r}; presets: {sorted(ROBOTS)}, or describe one from SimReady (see robots.register)')
        return entry
    miss = [k for k in ('name', 'source', 'reach', 'payload', 'vmax_deg', 'datasheet') if k not in entry]
    if miss: raise KeyError(f'robot candidate {entry.get("name", entry)!r} is missing {miss}')
    if not str(entry['source']).startswith('simready:'): raise KeyError(f'robot source must be "simready:<id>", got {entry["source"]!r}')
    if len(entry['vmax_deg']) != 6: raise KeyError(f'{entry["name"]}: vmax_deg needs six joint speeds (deg/s); only six-axis arms are supported')
    ROBOTS[entry['name']] = dict(simready=entry['source'].split(':', 1)[1], reach=float(entry['reach']), payload=float(entry['payload']),
                                 vmax=np.radians(np.asarray(entry['vmax_deg'], float)), datasheet=entry['datasheet'])
    return entry['name']
