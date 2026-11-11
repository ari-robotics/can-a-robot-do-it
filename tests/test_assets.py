import json, os, pytest
import numpy as np
from ari_sim import spec as S, simready as SR, assets
from ari_sim.robots import ROBOTS, DEFAULT_CANDIDATES, menagerie_dir
EX = os.path.join(os.path.dirname(__file__), '..', 'examples', 'goods_to_person_station', 'scene.json')

NAMES = ['crate_blue_a01', 'crate_red_a19', 'bin_waste_a01', 'handTool_screwdriver_phillips_ph2_a01', 'furniture_workbench_standard_e01', 'food_apple_a01']

def test_name_fallback_ranks_catalog_words_and_synonyms(monkeypatch):
    monkeypatch.setattr(SR, 'names', lambda: NAMES)
    assert SR.tokens('handTool_screwdriver_phillips_ph2_a01') >= {'handtool', 'hand', 'tool', 'screwdriver', 'phillips'}
    assert 'a01' not in SR.tokens('crate_blue_a01')
    assert SR.search_names('blue crate')[0] == 'crate_blue_a01'
    assert set(SR.search_names('tote')) == {'crate_blue_a01', 'crate_red_a19', 'bin_waste_a01'}      # tote -> crate, bin
    assert SR.search_names('teapot') == []

def test_search_uses_the_api_and_falls_back_to_names(monkeypatch):
    import io
    hit = 'https://huggingface.co/datasets/FANUC-CORPORATION/SimReady-Assets/blob/' + 'a' * 40 + '/crx10ia/simready_usd/crx10ia.usd'
    monkeypatch.setattr(SR, '_open', lambda url, timeout=60: io.BytesIO(json.dumps([dict(url=hit), dict(url=hit)]).encode()))
    assert SR.search('fanuc arm') == [('FANUC-CORPORATION/SimReady-Assets/crx10ia/simready_usd/crx10ia.usd@' + 'a' * 40, 'central')]
    def down(url, timeout=60): raise OSError('unreachable')
    monkeypatch.setattr(SR, '_open', down); monkeypatch.setattr(SR, 'names', lambda: NAMES)
    assert SR.search('blue crate')[0] == ('crate_blue_a01', 'names')

def test_asset_ids():
    rev = 'b' * 40
    a = SR.parse(f'simready:Robotiq-Official/simready-assets/Robotiq_2F_85/simready_usd/Robotiq_2F_85.usda@{rev}')
    assert (a['repo'], a['rev'], a['folder']) == ('Robotiq-Official/simready-assets', rev, 'Robotiq_2F_85')
    assert SR.parse(f'https://huggingface.co/datasets/Robotiq-Official/simready-assets/blob/{rev}/Robotiq_2F_85/simready_usd/Robotiq_2F_85.usda')['id'] == a['id']
    assert SR.parse('crate_blue_a01') == dict(repo=SR.REPO, rev=SR.REVISION, path=None, folder='crate_blue_a01', id='crate_blue_a01')
    for bad in ('org/repo/x.usd@main', 'a/b'):
        with pytest.raises(KeyError): SR.parse(bad)

def test_download_skips_textures_and_thumbnails(monkeypatch):
    tree = [dict(path=f'crate_blue_a01/{p}', size=1) for p in ['simready_usd/sm_crate_blue_a01_01.usd', 'simready_usd/payloads/geometries.usd',
            'simready_usd/payloads/MuJoCo/enabled.usda', 'simready_usd/Textures/T_Albedo.png', 'simready_usd/source_assets/T_Normal.png',
            'simready_usd/.thumbs/256x256/sm_crate_blue_a01_01.usd.png', 'simready_usd/materials/OmniPBR.mdl', 'com.nvidia.simready.packaging.json', '.metadata/LICENSE.txt']]
    monkeypatch.setattr(SR, 'files', lambda ref: tree)
    got = {f['path'].split('/', 1)[1] for f in SR._wanted('crate_blue_a01')}
    assert got == {'simready_usd/sm_crate_blue_a01_01.usd', 'simready_usd/payloads/geometries.usd', 'simready_usd/payloads/MuJoCo/enabled.usda', 'com.nvidia.simready.packaging.json'}

@pytest.fixture
def fake_assets(tmp_path, monkeypatch):
    """A cached asset without network: a 0.6 x 0.4 x 0.25 m box mesh."""
    obj = tmp_path / 'mesh.obj'
    v = [(x, y, z) for x in (-.3, .3) for y in (-.2, .2) for z in (0, .25)]
    f = [(1, 2, 4), (1, 4, 3), (5, 7, 8), (5, 8, 6), (1, 5, 6), (1, 6, 2), (3, 4, 8), (3, 8, 7), (1, 3, 7), (1, 7, 5), (2, 6, 8), (2, 8, 4)]
    obj.write_text(''.join(f'v {a} {b} {c}\n' for a, b, c in v) + ''.join(f'f {a} {b} {c}\n' for a, b, c in f))
    meta = dict(name='crate_blue_a01', mesh=str(obj), extent=[0.6, 0.4, 0.25], mass=2.8, friction=0.4, warnings=[])
    monkeypatch.setattr(SR, 'get', lambda name, max_mb=200: dict(meta, name=name))
    return meta

def _scene(extra):
    sc = json.load(open(EX)); sc['components'] = [c for c in sc['components'] if c['name'] != 'tote'] + extra; return sc

def test_asset_roles(fake_assets):
    sc = _scene([dict(type='asset', name='crate', source='simready:crate_blue_a01', role='container', pos=[-1.2, 0.3, 0.87]),
                 dict(type='asset', name='bench', source='simready:crate_blue_a01', role='fixture', pos=[-1.2, -1.0, 0], yaw=90),
                 dict(type='asset', name='part', source='simready:crate_blue_a01', role='item')])
    R = S.resolve(S.load(sc)); g = {x['name']: x for x in R['geoms']}
    assert g['crate']['shape'] == 'mesh' and not g['crate']['collide'] and g['crate']['mesh'] == fake_assets['mesh']
    assert all(g[f'crate_wall_{k}']['rgba'][3] == 0 and g[f'crate_wall_{k}']['collide'] for k in ('bottom', 'x-1', 'y1'))
    place = [p for p in R['places'] if p['name'] == 'crate'][0]
    assert place['box']['rim'] == pytest.approx(0.87 + 0.25) and place['drop'][2] == pytest.approx(0.87 + 0.25 + 0.08)
    assert g['bench_solid']['size'].tolist() == [0.6, 0.4, 0.25] and g['bench_solid']['pos'][2] == pytest.approx(0.125)
    assert [it['source'] for it in R['items']] == ['simready:crate_blue_a01']

def test_asset_errors(fake_assets):
    with pytest.raises(S.SceneError): S.load(_scene([dict(type='asset', name='x', source='crate_blue_a01', pos=[0, 0, 0])]))
    with pytest.raises(S.SceneError): S.load(_scene([dict(type='asset', name='x', source='simready:crate_blue_a01', role='decor', pos=[0, 0, 0])]))

def test_asset_mesh_compiles(fake_assets):
    if not os.path.isdir(os.path.join(assets.MENAGERIE, menagerie_dir('UR10e'))): pytest.skip('robot models not downloaded (ari-sim setup)')
    from ari_sim.build import build
    sc = _scene([dict(type='asset', name='crate', source='simready:crate_blue_a01', role='container', pos=[-1.2, 0.3, 0.87])])
    m, d, info = build('UR10e', S.resolve(S.load(sc)))
    assert m.geom('env_crate').contype == 0 and m.nmesh >= 1

def test_robot_catalog_and_scene_robots():
    from ari_sim.robots import register
    assert set(DEFAULT_CANDIDATES) <= set(ROBOTS)
    for k, r in ROBOTS.items():
        assert {'reach', 'payload', 'vmax'} <= set(r) and len(r['vmax']) == 6, k
        assert ('xml' in r and {'base', 'tip'} <= set(r) and menagerie_dir(k) == r['xml'].split('/')[0]) or ('simready' in r and 'datasheet' in r), k
    arm = dict(name='Test arm', source='simready:FANUC-CORPORATION/SimReady-Assets/crx10ia_l/simready_usd/crx10ia_l.usd@' + 'c' * 40,
               reach=1.418, payload=10, vmax_deg=[120] * 6, datasheet='https://example.com/datasheet')
    assert register(arm) == 'Test arm' and ROBOTS['Test arm']['vmax'][0] == pytest.approx(np.radians(120))
    assert register('UR10e') == 'UR10e'
    with pytest.raises(KeyError): register(dict(arm, datasheet=None) | {'vmax_deg': [120] * 7})
    with pytest.raises(KeyError): register({k: v for k, v in arm.items() if k != 'datasheet'})
    with pytest.raises(KeyError): register('R2-D2')
    with pytest.raises(KeyError): assets.fetch_robots(['R2-D2'])
    ROBOTS.pop('Test arm')
