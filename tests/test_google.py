"""The parts that run on Google's robotics plugin: found, and doing what the diagnostic relies on."""
import os, json, pytest
import numpy as np, mujoco
from ari_sim import spec as S, google, build, assets
from ari_sim.plan import frame
EX = os.path.join(os.path.dirname(__file__), '..', 'examples', 'goods_to_person_station', 'scene.json')
needs_plugin = pytest.mark.skipif(google.path('motion-planning', need=False) is None, reason="Google's robotics plugin not installed (ari-sim setup)")
needs_robot = pytest.mark.skipif(not os.path.isdir(os.path.join(assets.MENAGERIE, 'universal_robots_ur10e')), reason='robot models not downloaded (ari-sim setup)')

def test_every_tool_of_the_plugin_has_a_stage():
    assert {t['stage'] for t in google.TOOLS.values()} == {'perceive', 'rebuild', 'plan', 'execute', 'verify', 'report', 'real'}
    assert len(google.COMMIT) == 40 and all(t['use'] for t in google.TOOLS.values())
    with pytest.raises(KeyError): google.path('not-a-skill')

def test_frame_points_the_tool_along_the_approach():
    for a in ([0, 1, 0], [0, 0, -1], [0.3, 0.2, -0.9]):
        for spin in (0.0, np.pi / 2):
            R = frame(a, spin); n = np.asarray(a, float) / np.linalg.norm(a)
            assert np.allclose(R[:, 2], n) and np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1)
    assert abs(frame([0, 1, 0], np.pi / 2)[2, 1]) < 1e-9          # spun a quarter turn, a gripper closes horizontally in a shelf cell

def test_station_follows_the_asset_standards(tmp_path):
    sc = S.load(EX); p = build.station(S.resolve(sc), sc['name'], str(tmp_path)); m = mujoco.MjModel.from_xml_path(p)
    assert os.path.basename(os.path.dirname(p)) == 'goods_to_person_pick_station' and m.body(1).name == 'goods_to_person_pick_station'
    assert np.allclose(m.body_pos[1], 0) and m.njnt == 0 and m.nlight == 0          # one root body, no transform, no joint, no scene dressing
    assert set(m.geom_group) == {2, 3}
    vis = m.geom_group == 2; col = m.geom_group == 3
    assert (m.geom_contype[vis] == 0).all() and (m.geom_conaffinity[vis] == 0).all() and (m.geom_contype[col] == 1).all()
    assert ((m.geom_rgba[col, 3] >= 0.35) & (m.geom_rgba[col, 3] <= 0.6)).all() and (m.site_group == 4).all() and m.nsite == 33

@needs_plugin
def test_measure_maps_pixels_to_metres():
    from ari_sim.perceive import measure
    refs = [[100, 400, 0, 0], [500, 400, 0.6, 0], [500, 200, 0.6, 0.4], [100, 200, 0, 0.4]]          # a Euro tote seen square-on
    r = measure(refs, [[300, 300], [700, 400]]); a, b = r['points']
    assert (a['x'], a['y']) == pytest.approx((0.3, 0.2), abs=1e-6) and b['x'] == pytest.approx(0.9, abs=1e-6)
    assert r['reference_fit_m'] < 1e-6 and 0 < a['sigma'] < b['sigma']          # further from the reference, less sure

@needs_plugin
@needs_robot
def test_cell_plans_a_full_cycle_with_googles_planner():
    from ari_sim.plan import Cell
    from ari_sim.diagnose import _evaluate
    from ari_sim.robots import tool
    sc = S.load(EX); R = S.resolve(sc); c = Cell('UR10e', tool(sc['robot']['tool'])).place(R, -0.6, -0.8, 1.3)
    assert type(c.pl).__module__ == 'motion_planner' and c.pl.dof == 6
    res, place, (qpp, qdrop) = _evaluate(c, R); ok = [(t, r) for t, r in zip(R['picks'], res) if r['status'] == 'ok']
    assert place == 'ok' and len(ok) >= 20
    assert _evaluate(c.place(R, 5.0, 5.0, 0.5), R)[1] == 'out_of_reach'          # an arm across the room reaches nothing
    c.place(R, -0.6, -0.8, 1.3); t = ok[len(ok) // 2][0]; q = c.reach_near(t['pre'], t['approach'], qpp)[0]
    ins, ti = c.stroke(q, t['pick']); path, tp = c.move(qpp, q)
    assert ins is not None and path is not None and ti > 0 and tp > 0
    assert np.linalg.norm(c.tip(ins[-1])[0] - t['pick']) < 5e-3 and c.clearance(path[len(path) // 2])[0]

@needs_plugin
@needs_robot
def test_scene_holds_arm_station_and_tool():
    from ari_sim.robots import tool
    sc = S.load(EX); m, d, info = build.scene('UR10e', S.resolve(sc), tool(sc['robot']['tool']), (-0.6, -0.8, 1.3), name=sc['name'])
    assert len(info['qadr']) == 6 and m.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE and m.geom('pedestal').size[2] == pytest.approx(0.65)
