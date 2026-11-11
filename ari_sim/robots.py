import numpy as np
# Reach/payload from the manufacturers' datasheets; joint speed limits in rad/s. Models come from MuJoCo Menagerie
# and are downloaded only when a scene asks for them. Six-axis arms only: the planner assumes six joints.
ROBOTS = {
    'UR5e':    dict(xml='universal_robots_ur5e/ur5e.xml',   base='base',      tip='wrist_3_link', reach=0.85,  payload=5.0,  vmax=np.radians([180, 180, 180, 180, 180, 180])),
    'UR10e':   dict(xml='universal_robots_ur10e/ur10e.xml', base='base',      tip='wrist_3_link', reach=1.30,  payload=12.5, vmax=np.radians([120, 120, 180, 180, 180, 180])),
    'CRX-5iA': dict(xml='fanuc_crx5ia/crx5ia.xml',          base='base_link', tip='J6_link',      reach=0.994, payload=5.0,  vmax=np.radians([240, 240, 240, 250, 250, 250])),
    'Lite6':   dict(xml='ufactory_lite6/lite6.xml',         base='link_base', tip='link6',        reach=0.44,  payload=0.6,  vmax=np.radians([180, 180, 180, 180, 180, 180])),
}
DEFAULT_CANDIDATES = ['UR5e', 'UR10e', 'CRX-5iA']     # tried when a scene names none; fetched by `ari-sim setup`
GRIPPER_XML = 'robotiq_2f85/2f85.xml'

def menagerie_dir(robot): return ROBOTS[robot]['xml'].split('/')[0]
