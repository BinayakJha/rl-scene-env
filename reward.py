"""Programmatic, verifiable reward for scene plans.
Components (each in [0,1]): format, count, relation, treatment; plus a length penalty.
Total = 0.15*format + 0.30*count + 0.30*relation + 0.25*treatment - length_penalty.
Anti-hacking: extra/hallucinated objects reduce count & relation scores (precision term),
and count credit requires exact match.
"""
import json, re

TREATMENTS = {"object", "comparison", "chart", "process", "hold"}
RELS = {"on","under","behind","in_front_of","next_to","inside","left_of","right_of"}
WEIGHTS = {"format": 0.15, "count": 0.30, "relation": 0.30, "treatment": 0.25}
MAX_CHARS = 400

def _norm(s):
    s = str(s).lower().strip().replace(" ", "_").replace("-", "_")
    return s[:-1] if s.endswith("s") and len(s) > 3 and not s.endswith("ss") else s

_NAME_ALIASES = {"people": "person", "children": "child", "women": "woman", "men": "man"}
def _name(s):
    s = str(s).lower().strip()
    return _NAME_ALIASES.get(s, _norm(s))

def parse(text):
    """Extract the first JSON object from model output; return dict or None."""
    if not isinstance(text, str): return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None

def format_score(plan):
    if plan is None: return 0.0
    ok = isinstance(plan.get("objects"), list) and plan.get("treatment") in TREATMENTS
    if ok:
        for o in plan["objects"]:
            if not (isinstance(o, dict) and "name" in o and isinstance(o.get("count"), int)): ok = False
    return 1.0 if ok else 0.5

def _objs(plan):
    out = {}
    for o in (plan or {}).get("objects", []) or []:
        if isinstance(o, dict) and "name" in o: out[_name(o["name"])] = o
    return out

def _f1(hits, n_pred, n_gold):
    if n_gold == 0 and n_pred == 0: return 1.0
    if n_gold == 0 or n_pred == 0: return 0.0
    p, r = hits / n_pred, hits / n_gold
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

def count_score(plan, gold):
    P, G = _objs(plan), _objs(gold)
    hits = sum(1 for k, g in G.items() if k in P and P[k].get("count") == g["count"])
    return _f1(hits, len(P), len(G))

def relation_score(plan, gold):
    P, G = _objs(plan), _objs(gold)
    gold_rels = {k: g["relation"] for k, g in G.items() if g.get("relation")}
    pred_rels = {k: p["relation"] for k, p in P.items() if isinstance(p.get("relation"), dict)}
    if not gold_rels and not pred_rels: return 1.0
    hits = 0
    for k, r in gold_rels.items():
        pr = pred_rels.get(k)
        if pr and _norm(pr.get("type", "")) == _norm(r["type"]) and _name(pr.get("target", "")) == _name(r["target"]):
            hits += 1
    return _f1(hits, len(pred_rels), len(gold_rels))

def treatment_score(plan, gold):
    return 1.0 if plan and plan.get("treatment") == gold["treatment"] else 0.0

def length_penalty(text):
    n = len(text or "")
    return 0.0 if n <= MAX_CHARS else min(0.2, (n - MAX_CHARS) / 2000)

def score(text, gold):
    plan = parse(text)
    c = {"format": format_score(plan)}
    if plan is None:
        c.update(count=0.0, relation=0.0, treatment=0.0)
    else:
        c.update(count=count_score(plan, gold), relation=relation_score(plan, gold),
                 treatment=treatment_score(plan, gold))
    c["length_penalty"] = length_penalty(text)
    c["total"] = sum(WEIGHTS[k] * c[k] for k in WEIGHTS) - c["length_penalty"]
    c["exact"] = float(all(c[k] == 1.0 for k in WEIGHTS))
    return c

# TRL GRPO reward functions: f(prompts, completions, **dataset_columns) -> list[float]
def _txt(c): return c if isinstance(c, str) else c[-1]["content"]
def _mk(key):
    def f(prompts, completions, gold, **kw):
        return [score(_txt(c), json.loads(g) if isinstance(g, str) else g)[key] for c, g in zip(completions, gold)]
    f.__name__ = f"reward_{key}"; return f
reward_total = _mk("total")
