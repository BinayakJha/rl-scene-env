"""Collect real statistics by running policies through SceneEnv. Writes results/env_stats.json + figures."""
import json, os, random, statistics as st, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_env.scene_env import SceneEnv, TokenSceneEnv, load_split, ROOT
from rl_env.policies import random_policy, heuristic_policy, load_replay

COMP = ["format", "count", "relation", "treatment", "length_penalty", "total", "exact"]

def run(env, act_fn, n):
    rows = []
    for i in range(n):
        obs, info = env.reset(options={"index": i})
        a = act_fn(obs, info)
        _, r, term, trunc, inf = env.step(a)
        rows.append(dict(inf, action=a, obs=obs))
    return rows

def summarize(rows):
    tot = np.array([r["total"] for r in rows])
    out = {"n": len(rows), "reward_mean": round(float(tot.mean()), 4), "reward_std": round(float(tot.std(ddof=1)), 4),
           "reward_min": round(float(tot.min()), 4), "reward_max": round(float(tot.max()), 4),
           "hist_counts": np.histogram(tot, bins=10, range=(0, 1))[0].tolist(),
           "components": {k: round(float(np.mean([r[k] for r in rows])), 4) for k in COMP},
           "action_chars_mean": round(float(np.mean([len(r["action"]) for r in rows])), 1),
           "action_chars_max": int(max(len(r["action"]) for r in rows))}
    for kind in ("spatial", "abstract"):
        k = [r["total"] for r in rows if r["kind"] == kind]
        out[f"{kind}_mean"] = round(float(np.mean(k)), 4); out[f"{kind}_n"] = len(k)
    return out

def main():
    S = {}
    tr, te = load_split("train"), load_split("test")
    S["dataset"] = {"train": len(tr), "test": len(te),
                    "train_kinds": Counter(e["kind"] for e in tr), "test_kinds": Counter(e["kind"] for e in te),
                    "test_treatments": Counter(e["gold"]["treatment"] for e in te),
                    "train_treatments": Counter(e["gold"]["treatment"] for e in tr)}
    for name, D in (("train", tr), ("test", te)):
        w = [len(e["text"].split()) for e in D]; c = [len(e["text"]) for e in D]
        S[f"obs_{name}"] = {"words_mean": round(st.mean(w), 2), "words_min": min(w), "words_max": max(w),
                            "chars_mean": round(st.mean(c), 1), "chars_max": max(c),
                            "unique_words": len({x for e in D for x in e["text"].lower().split()})}
    gw = [len(json.dumps(e["gold"], separators=(",", ":"))) for e in te]
    S["gold_action_chars_test"] = {"mean": round(st.mean(gw), 1), "min": min(gw), "max": max(gw)}
    trw = {x for e in tr for x in e["text"].lower().split()}; tew = {x for e in te for x in e["text"].lower().split()}
    S["test_words_unseen_in_train"] = len(tew - trw)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        gt = [len(tok.encode(json.dumps(e["gold"], separators=(",", ":")))) for e in te]
        S["gold_action_qwen_tokens"] = {"mean": round(st.mean(gt), 1), "max": max(gt), "vocab": len(tok)}
    except Exception as ex:
        S["gold_action_qwen_tokens"] = f"unavailable ({type(ex).__name__})"

    env = SceneEnv("test", shuffle=False)
    rng = random.Random(0)
    P = {}
    P["random"] = run(env, lambda o, i: random_policy(o, rng), len(te))
    P["heuristic"] = run(env, lambda o, i: heuristic_policy(o, rng), len(te))
    P["oracle_gold"] = run(env, lambda o, i: json.dumps(te[i["index"]]["gold"]), len(te))
    for tag, path in (("base_qwen0.5b", "results/base_qwen2.5-0.5b_full300.json"), ("sft_qwen0.5b", "results/sft_full300.json")):
        rep, summ = load_replay(path)
        P[tag] = run(env, lambda o, i: rep[i["id"]], len(te))
        S.setdefault("replay_saved_summary_total", {})[tag] = summ["total"]
    S["policies"] = {k: summarize(v) for k, v in P.items()}

    # per-prompt variance across K samples (cheap stochastic heuristic + random), the GRPO group-size question
    K = 8; pv = {}
    for name, fn in (("random", lambda o, r: random_policy(o, r)),
                     ("heuristic_eps0.3", lambda o, r: heuristic_policy(o, r, eps=0.3))):
        r2 = random.Random(1); within, zero_var, means = [], 0, []
        for i in range(len(te)):
            rs = []
            for _ in range(K):
                o, _ = env.reset(options={"index": i}); rs.append(env.step(fn(o, r2))[1])
            within.append(np.var(rs, ddof=1)); means.append(np.mean(rs)); zero_var += np.ptp(rs) == 0
        pv[name] = {"K": K, "mean_within_prompt_var": round(float(np.mean(within)), 4),
                    "between_prompt_var_of_means": round(float(np.var(means, ddof=1)), 4),
                    "frac_groups_zero_variance": round(zero_var / len(te), 4)}
    S["per_prompt_variance"] = pv

    # token-level env smoke stats (char tokenizer): episode length of gold completions
    tenv = TokenSceneEnv("test", shuffle=False, max_tokens=512); L = []; R = []
    for i in range(len(te)):
        tenv.reset(options={"index": i}); ids = tenv.tok.encode(json.dumps(te[i]["gold"])) + [tenv.tok.eos_token_id]
        for t, a in enumerate(ids):
            _, r, d, tr_, _ = tenv.step(a)
            if d or tr_: break
        L.append(t + 1); R.append(r)
    S["token_env_gold_char_episode"] = {"len_mean": round(st.mean(L), 1), "len_max": max(L), "reward_mean": round(st.mean(R), 4)}

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(S, open(os.path.join(ROOT, "results", "env_stats.json"), "w"), indent=1, default=dict)
    figs(P, pv)
    print(json.dumps(S, indent=1, default=dict))

def figs(P, pv):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = os.path.join(ROOT, "figures"); os.makedirs(d, exist_ok=True)
    order = ["random", "heuristic", "base_qwen0.5b", "sft_qwen0.5b", "oracle_gold"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].boxplot([[r["total"] for r in P[k]] for k in order], showmeans=True)
    ax[0].set_xticks(range(1, 6), ["random", "heuristic", "base 0.5B", "SFT 0.5B", "oracle"], fontsize=8)
    ax[0].set_ylabel("episode reward"); ax[0].set_title("Reward per policy (300 test episodes)")
    comps = ["format", "count", "relation", "treatment"]; w = .16
    for j, k in enumerate(order):
        ax[1].bar(np.arange(4) + (j - 2) * w, [np.mean([r[c] for r in P[k]]) for c in comps], w, label=k)
    ax[1].set_xticks(range(4), comps); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=7); ax[1].set_title("Mean component score")
    fig.tight_layout(); fig.savefig(os.path.join(d, "rewards.png"), dpi=200)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k in ("base_qwen0.5b", "heuristic", "random"):
        ax.hist([r["total"] for r in P[k]], bins=20, range=(0, 1), alpha=.55, label=k)
    ax.set_xlabel("episode reward"); ax.set_ylabel("count"); ax.legend(); ax.set_title("Reward distribution, test split")
    fig.tight_layout(); fig.savefig(os.path.join(d, "reward_hist.png"), dpi=200)

if __name__ == "__main__":
    main()
