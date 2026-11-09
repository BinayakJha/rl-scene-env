"""Train PPO for one seed. Every few updates we check VALIDATION success and keep the best
network so far (model selection uses validation only, never test)."""
import argparse
import json
import time

import torch

import _setup  # noqa: F401
from scene_repair import settings
from scene_repair.encoding import OBS_SIZE
from scene_repair.env import NUM_ACTIONS, SceneRepairEnv
from scene_repair.evaluation import evaluate, summarize
from scene_repair.networks import NetPolicy, PolicyValueNet
from scene_repair.policies.ppo import PPOConfig, train_ppo
from scene_repair.scenes import load_scenes


def main(seed, quick, variant="main"):
    torch.set_num_threads(1)  # we run several seeds side by side, one CPU core each
    cfg = settings.get(quick)
    out = settings.results_dir(quick)
    env_kwargs = settings.VARIANTS[variant]
    steps = cfg["ppo_steps"] if variant == "main" else cfg["ablation_ppo_steps"]
    ppo_cfg = PPOConfig(total_steps=steps, **settings.PPO_HYPERPARAMETERS)
    val_scenes = load_scenes(out / "data" / "val.json")
    val_env = SceneRepairEnv(**env_kwargs)
    net = PolicyValueNet(OBS_SIZE, NUM_ACTIONS, settings.HIDDEN_SIZE)
    (out / "models").mkdir(parents=True, exist_ok=True)
    model_path = out / "models" / f"ppo_{variant}_seed{seed}.pt"

    curve, best, updates, start = [], [(-1.0, -1.0)], [0], time.time()

    def on_update(steps_done, log):
        updates[0] += 1
        point = {"steps": steps_done, **{k: float(v) for k, v in log.items()}}
        if updates[0] % cfg["ppo_eval_every"] == 0 or steps_done >= steps:
            val = summarize(evaluate(val_env, NetPolicy(net, "ppo"), val_scenes, 1, base_seed=77))
            point["val_success"] = val["success"]["mean"]
            point["val_reward"] = val["reward"]["mean"]
            if (point["val_success"], point["val_reward"]) > best[0]:
                best[0] = (point["val_success"], point["val_reward"])
                point["saved"] = True
                torch.save(net.state_dict(), model_path)
            print(f"[ppo {variant} seed {seed}] {steps_done:,} steps  "
                  f"train success {log['train_success']:.3f}  val success {point['val_success']:.3f}  "
                  f"entropy {log['entropy']:.2f}  ({time.time() - start:.0f}s)", flush=True)
        curve.append(point)

    total = train_ppo(net, ppo_cfg, seed, env_kwargs, on_update)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    with open(out / "curves" / f"ppo_{variant}_seed{seed}.json", "w") as f:
        json.dump({"curve": curve, "env_steps": total, "best_val": best[0],
                   "train_seconds": time.time() - start, "config": vars(ppo_cfg)}, f, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="main")
    parser.add_argument("--quick", action="store_true")
    a = parser.parse_args()
    main(a.seed, a.quick, a.variant)
