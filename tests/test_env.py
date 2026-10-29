import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gymnasium.utils.env_checker import check_env
from rl_env.scene_env import SceneEnv, TokenSceneEnv, load_split

def test_bandit_gold_gets_full_reward():
    env = SceneEnv("test", shuffle=False)
    obs, info = env.reset(options={"index": 0})
    gold = load_split("test")[0]["gold"]
    assert obs == load_split("test")[0]["text"]
    _, r, term, trunc, inf = env.step(json.dumps(gold))
    assert term and not trunc and abs(r - 1.0) < 1e-9 and inf["exact"] == 1.0

def test_bandit_garbage_zero():
    env = SceneEnv("train"); env.reset(seed=0)
    _, r, term, _, inf = env.step("not json")
    assert term and r == 0.0 and inf["format"] == 0.0

def test_seed_reproducible():
    a, b = SceneEnv("train"), SceneEnv("train")
    assert a.reset(seed=7)[0] == b.reset(seed=7)[0]

def test_token_env_episode():
    env = TokenSceneEnv("test", shuffle=False)
    env.reset(options={"index": 0})
    ids = env.tok.encode(json.dumps(load_split("test")[0]["gold"]))
    for a in ids:
        _, r, d, t, _ = env.step(a); assert r == 0.0 and not d
    _, r, d, _, _ = env.step(env.tok.eos_token_id)
    assert d and abs(r - 1.0) < 1e-9

def test_token_env_truncates():
    env = TokenSceneEnv("test", max_tokens=5); env.reset(seed=0)
    for _ in range(5): out = env.step(0)
    assert out[3] is True

def test_spaces_contain_obs():
    env = TokenSceneEnv("train"); o, _ = env.reset(seed=1)
    assert isinstance(o["prompt"], str) and o["tokens"] == ()
    assert env.action_space.contains(env.tok.eos_token_id)
    b = SceneEnv("train"); ob, _ = b.reset(seed=1); assert isinstance(ob, str) and len(ob) > 0
