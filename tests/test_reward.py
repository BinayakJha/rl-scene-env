import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from reward import score, parse, reward_total

GOLD = {"objects": [{"name": "kid", "count": 3, "relation": {"type": "next_to", "target": "teacher"}},
                    {"name": "teacher", "count": 1, "relation": None}], "treatment": "object"}
PERFECT = json.dumps(GOLD)

def test_perfect():
    s = score(PERFECT, GOLD); assert s["total"] == 1.0 and s["exact"] == 1.0
def test_plural_names_ok():
    p = json.loads(PERFECT); p["objects"][0]["name"] = "kids"; assert score(json.dumps(p), GOLD)["count"] == 1.0
def test_invalid_json():
    s = score("three kids {oops", GOLD); assert s["format"] == 0 and s["total"] == 0
def test_wrong_count():
    p = json.loads(PERFECT); p["objects"][0]["count"] = 2; s = score(json.dumps(p), GOLD)
    assert s["count"] == 0.5 and s["relation"] == 1.0
def test_wrong_relation():
    p = json.loads(PERFECT); p["objects"][0]["relation"]["type"] = "behind"
    assert score(json.dumps(p), GOLD)["relation"] == 0.0
def test_wrong_treatment():
    p = json.loads(PERFECT); p["treatment"] = "chart"; assert score(json.dumps(p), GOLD)["treatment"] == 0
def test_bad_schema_partial_format():
    assert score('{"foo": 1}', GOLD)["format"] == 0.5
def test_object_spam_penalised():
    p = json.loads(PERFECT); p["objects"] += [{"name": f"x{i}", "count": 1, "relation": None} for i in range(8)]
    assert score(json.dumps(p), GOLD)["count"] < 0.5
def test_length_penalty():
    s = score(PERFECT + " " * 1000, GOLD); assert s["length_penalty"] > 0 and s["total"] < 1
def test_hold():
    g = {"objects": [], "treatment": "hold"}
    assert score('{"objects": [], "treatment": "hold"}', g)["total"] == 1.0
    assert score('{"objects": [{"name":"cat","count":1}], "treatment": "hold"}', g)["count"] == 0.0
def test_parse_with_prose():
    assert parse('Sure! ```json\n' + PERFECT + '\n```')["treatment"] == "object"
def test_trl_wrapper():
    out = reward_total(["p"], [[{"role": "assistant", "content": PERFECT}]], gold=[json.dumps(GOLD)])
    assert out == [1.0]
