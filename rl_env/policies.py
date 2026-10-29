"""Simple reference policies: random, heuristic (rule-based), and replay of saved model outputs."""
import json, os, random, re
from .scene_env import ROOT

TREAT = ["object", "comparison", "chart", "process", "hold"]
RELS = ["on","under","behind","in_front_of","next_to","inside","left_of","right_of"]
NOUNS = ["cat","box","chair","person","tree","car","dog","table"]

def random_policy(text, rng):
    """Random but schema-valid plan (uniform over treatments, 0-2 objects, random relations)."""
    objs = []
    for _ in range(rng.randint(0, 2)):
        rel = None if rng.random() < .5 else {"type": rng.choice(RELS), "target": rng.choice(NOUNS)}
        objs.append({"name": rng.choice(NOUNS), "count": rng.randint(1, 5), "relation": rel})
    return json.dumps({"objects": objs, "treatment": rng.choice(TREAT)})

NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
PHRASES = [("in front of","in_front_of"),("next to","next_to"),("beside","next_to"),("on top of","on"),
           ("left of","left_of"),("right of","right_of"),("inside","inside"),("in","inside"),
           ("under","under"),("below","under"),("behind","behind"),("on","on")]

def _sing(w): return w[:-1] if w.endswith("s") and len(w) > 3 and not w.endswith("ss") else w

def heuristic_policy(text, rng, eps=0.0):
    """Regex rule policy. With prob eps per decision it makes a random choice (stochastic sampling)."""
    t = text.lower()
    m = re.search(r"\b(\d+|" + "|".join(NUM) + r")\s+(\w+)", t)
    rel = None
    for ph, ty in PHRASES:
        mm = re.search(r"\b" + ph + r"\s+(?:the\s+|a\s+)?(\w+)", t)
        if mm: rel = (ty, _sing(mm.group(1))); break
    if rng.random() < eps and rel: rel = (rng.choice(RELS), rel[1])
    if m:
        n = m.group(1); n = int(n) if n.isdigit() else NUM[n]
        if rng.random() < eps: n = rng.randint(1, 6)
        objs = [{"name": _sing(m.group(2)), "count": n,
                 "relation": {"type": rel[0], "target": rel[1]} if rel else None}]
        if rel: objs.append({"name": rel[1], "count": 1, "relation": None})
        tr = "object"
    else:
        objs = []
        tr = ("chart" if re.search(r"\d|doubl|half|grew|rose|drop|percent|%", t) else
              "comparison" if re.search(r"compar|versus|vs|than", t) else
              "process" if re.search(r"first|then|step|after", t) else "hold")
    if rng.random() < eps: tr = rng.choice(TREAT)
    return json.dumps({"objects": objs, "treatment": tr})

def load_replay(path):
    d = json.load(open(os.path.join(ROOT, path)))
    return {p["id"]: p["output"] for p in d["predictions"]}, d["summary"]
