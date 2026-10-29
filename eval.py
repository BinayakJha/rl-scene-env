"""Zero-shot (or fine-tuned) evaluation on the held-out test set.
python eval.py --model Qwen/Qwen2.5-0.5B-Instruct --n 50 --out results/base_0.5b.json
"""
import argparse, json, os, time, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from reward import score
from prompt import messages
from device import pick

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--data", default="data/test.jsonl")
ap.add_argument("--n", type=int, default=0, help="subset size (0=all)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_new_tokens", type=int, default=160)
ap.add_argument("--out", default="results/eval.json")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.data)]
if a.n: random.Random(a.seed).shuffle(rows); rows = rows[:a.n]
tok = AutoTokenizer.from_pretrained(a.model)
dev, dt = pick()
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt).to(dev).eval()
keys = ["format", "count", "relation", "treatment", "length_penalty", "total", "exact"]
agg = {k: 0.0 for k in keys}; per_kind = {}; preds = []; t0 = time.time()
for r in rows:
    prompt = tok.apply_chat_template(messages(r["text"]), add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    s = score(txt, r["gold"]); preds.append({"id": r["id"], "text": r["text"], "output": txt, **s})
    for k in keys: agg[k] += s[k]
    pk = per_kind.setdefault(r["kind"], {"n": 0, "total": 0.0, "exact": 0.0})
    pk["n"] += 1; pk["total"] += s["total"]; pk["exact"] += s["exact"]
n = len(rows)
summary = {"model": a.model, "n": n, "seconds": round(time.time() - t0, 1), **{k: round(v / n, 4) for k, v in agg.items()},
           "by_kind": {k: {"n": v["n"], "total": round(v["total"] / v["n"], 4), "exact": round(v["exact"] / v["n"], 4)} for k, v in per_kind.items()}}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"summary": summary, "predictions": preds}, open(a.out, "w"), indent=1)
print(json.dumps(summary, indent=1))
