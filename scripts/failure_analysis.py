"""Why do episodes fail? For each policy on the main test set, sort failed episodes into:
  stopped early   the policy chose STOP while a goal was still unmet
  out of budget   the budget hit 0 before every goal was met
and count wasted actions (invalid moves, empty UNDOs) per episode. Writes results/failure_analysis.json.
"""
import argparse
import json

import _setup  # noqa: F401
from scene_repair import settings
from scene_repair.env import SceneRepairEnv
from scene_repair.evaluation import evaluate
from scene_repair.networks import NetPolicy
from scene_repair.policies.planner import PlannerPolicy
from scene_repair.policies.random_policy import RandomPolicy
from scene_repair.scenes import load_scenes
from evaluate import TEST_SEED, load_net


def analyse(records):
    failed = [r for r in records if not r["success"]]
    stopped = sum(1 for r in failed if r["trajectory"][-1]["outcome"] == "stop")
    wasted = [sum(1 for t in r["trajectory"][1:] if t["outcome"] in ("invalid", "undo_empty"))
              for r in records]
    return {"episodes": len(records), "failures": len(failed), "stopped_early": stopped,
            "out_of_budget": len(failed) - stopped,
            "wasted_actions_per_episode": sum(wasted) / len(records)}


def main(quick):
    cfg = settings.get(quick)
    out = settings.results_dir(quick)
    scenes = load_scenes(out / "data" / "test.json")
    env = SceneRepairEnv()
    policies = {"random": [RandomPolicy(0)], "planner": [PlannerPolicy()]}
    for kind in ("bc", "ppo"):
        policies[kind] = [NetPolicy(load_net(out / "models" / f"{kind}_main_seed{s}.pt"), kind)
                          for s in cfg["seeds"]]
    report = {}
    for name, plist in policies.items():
        records = []
        for p in plist:
            records += evaluate(env, p, scenes, cfg["test_rollouts"], TEST_SEED, keep_trajectories=True)
        report[name] = analyse(records)
        print(name, report[name])
    with open(out / "failure_analysis.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    main(parser.parse_args().quick)
