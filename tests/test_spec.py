import json, os, pytest
from ari_sim import spec as S
EX = os.path.join(os.path.dirname(__file__), '..', 'examples', 'goods_to_person_station', 'scene.json')

def test_example_resolves():
    sc = S.load(EX); R = S.resolve(sc)
    assert len(R['picks']) == 32 and len(R['places']) == 1
    assert abs(R['picks'][0]['approach'][1] - 1) < 1e-9

def test_expressions_and_sampling():
    sc = S.load(EX); import numpy as np
    v = S.sample(sc, np.random.default_rng(0)); assert set(v) == set(sc['params'])
    assert S.num('-(open_w/2 + 0.08)', {'open_w': 1.1}) == pytest.approx(-0.63)

def test_rejects_code_and_unknowns():
    with pytest.raises(S.SceneError): S.num('__import__("os").system("true")', {})
    with pytest.raises(S.SceneError): S.num('nope + 1', {})
    sc = json.load(open(EX)); sc['components'].append({'type': 'teapot'})
    with pytest.raises(S.SceneError): S.load(sc)
