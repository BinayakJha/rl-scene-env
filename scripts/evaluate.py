"""Final evaluation on the frozen TEST set (run once, after all training and tuning).

Every policy plays the same test scenes with the same rollout seeds. For BC and PPO we
evaluate each of the 3 training seeds and also pool them. We also play the frozen train-check
and validation sets with the learned policies, to compare train vs val vs test (overfitting check).
"""
import argparse
import json
import platform
import time

import torch

import _setup  # noqa: F401
from scene_repair import settings
from scene_repair.encoding import OBS_SIZE
from scene_repair.env import NUM_ACTIONS, SceneRepairEnv
from scene_repair.evaluation import evaluate, summarize
from scene_repair.networks import NetPolicy, PolicyValueNet
from scene_repair.policies.planner import MAX_EXPANSIONS, PlannerPolicy
from scene_repair.policies.random_policy import RandomPolicy
from scene_repair.scenes import load_scenes

TEST_SEED = 5_000_000  # rollout seeds for test episodes (shared by all policies)


def load_net(path):
    net = PolicyValueNet(OBS_SIZE, NUM_ACTIONS, settings.HIDDEN_SIZE)
    net.load_state_dict(torch.load(path))
    return net


def hardware():
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    import os
    return {"cpu": cpu, "cores": os.cpu_count(), "python": platform.python_version(),
            "torch": torch.__version__, "os": platform.platform(), "gpu": "none (CPU only)"}


def main(quick, variants):
    cfg = settings.get(quick)
    out = settings.results_dir(quick)
    sets = {name: load_scenes(out / "data" / f"{name}.json") for name in ("test", "val", "train_check")}
    report = {"hardware": hardware(), "test_scenes": len(sets["test"]),
              "rollouts_per_scene": cfg["test_rollouts"], "variants": {}}

    for variant in variants:
        env = SceneRepairEnv(**settings.VARIANTS[variant])
        rows = {}

        def run(name, policy, scenes="test", rollouts=cfg["test_rollouts"]):
            start = time.time()
            records = evaluate(env, policy, sets[scenes], rollouts, base_seed=TEST_SEED)
            seconds = time.time() - start
            return records, seconds

        # random and planner: no training, so no training seeds
        records, seconds = run("random", RandomPolicy(seed=0))
        rows["random"] = {"pooled": summarize(records), "seconds": seconds}
        planner = PlannerPolicy(allow_undo=env.allow_undo)
        records, seconds = run("planner", planner)
        decisions = sum(r["steps"] for r in records)
        rows["planner"] = {"pooled": summarize(records), "seconds": seconds,
                           "decisions": decisions,
                           "ms_per_decision": 1000 * seconds / decisions,
                           "expansions_total": planner.total_expansions,
                           "expansion_limit": MAX_EXPANSIONS,
                           "decisions_hitting_limit": int(planner.limit_hits)}

        for kind in ("bc", "ppo"):
            paths = [out / "models" / f"{kind}_{variant}_seed{s}.pt" for s in cfg["seeds"]]
            if not all(p.exists() for p in paths):
                continue
            pooled, per_seed, gen = [], [], {}
            for seed, path in zip(cfg["seeds"], paths):
                policy = NetPolicy(load_net(path), kind)
                records, _ = run(kind, policy)
                for r in records:
                    r["scene_index"] = (seed, r["scene_index"])  # keep scenes apart per seed
                pooled += records
                per_seed.append({"seed": seed, **summarize(records)})
                for split in ("train_check", "val"):
                    rec, _ = run(kind, policy, scenes=split, rollouts=1)
                    gen.setdefault(split, []).append(summarize(rec)["success"]["mean"])
            # pooled CI resamples (seed, scene) pairs, so it includes seed-to-seed variation
            rows[kind] = {"pooled": summarize(pooled), "per_seed": per_seed,
                          "train_vs_val_success": gen}
        report["variants"][variant] = rows
        print(f"== {variant} ==")
        for name, row in rows.items():
            p = row["pooled"]
            print(f"  {name:8s} success {p['success']['mean']:.3f} "
                  f"[{p['success']['ci_low']:.3f}, {p['success']['ci_high']:.3f}]  "
                  f"R {p['reward']['mean']:.3f}  actions {p['actions_used']['mean']:.2f}", flush=True)

    with open(out / "test_results.json", "w") as f:
        json.dump(report, f, indent=1)
    write_table(report, out / "results_table.md")


def write_table(report, path):
    lines = []
    for variant, rows in report["variants"].items():
        lines += [f"### Test results: {variant}", "",
                  "| Policy | Full success | Mean R | Constraints met (s) | Actions used | UNDOs/episode |",
                  "|---|---|---|---|---|---|"]
        for name, row in rows.items():
            p = row["pooled"]
            fmt = lambda k, d=3: f"{p[k]['mean']:.{d}f} [{p[k]['ci_low']:.{d}f}, {p[k]['ci_high']:.{d}f}]"
            lines.append(f"| {name} | {fmt('success')} | {fmt('reward')} | {fmt('s')} | "
                         f"{fmt('actions_used', 2)} | {p['undos']['mean']:.2f} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--variants", nargs="+", default=["main", "deterministic", "no_undo"])
    a = parser.parse_args()
    main(a.quick, a.variants)
