"""Train behavior cloning for one seed: collect planner demos, then fit the network.

The epoch with the best VALIDATION success is kept (early stopping on validation only).
"""
import argparse
import json
import time

import numpy as np
import torch

import _setup  # noqa: F401
from scene_repair import settings
from scene_repair.encoding import OBS_SIZE
from scene_repair.env import NUM_ACTIONS, SceneRepairEnv
from scene_repair.evaluation import evaluate, summarize
from scene_repair.networks import NetPolicy, PolicyValueNet
from scene_repair.policies.bc import collect_demos, train_bc_epoch
from scene_repair.scenes import load_scenes


def main(seed, quick, variant="main"):
    torch.set_num_threads(1)
    cfg = settings.get(quick)
    out = settings.results_dir(quick)
    env_kwargs = settings.VARIANTS[variant]
    val_scenes = load_scenes(out / "data" / "val.json")
    val_env = SceneRepairEnv(**env_kwargs)

    # 1. demos: each seed gets its own demo episodes (different training scenes and noise)
    start = time.time()
    obs, masks, actions, expansions = collect_demos(
        cfg["bc_demo_episodes"], env_kwargs, seed_offset=seed * 10_000_000)
    demo_seconds = time.time() - start

    # 2. supervised training, checking validation after every epoch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = PolicyValueNet(OBS_SIZE, NUM_ACTIONS, settings.HIDDEN_SIZE)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.BC_LEARNING_RATE)
    curve, best = [], (-1.0, -1.0)
    for epoch in range(1, cfg["bc_epochs"] + 1):
        loss = train_bc_epoch(net, optimizer, obs, masks, actions, settings.BC_BATCH_SIZE, rng)
        val = summarize(evaluate(val_env, NetPolicy(net, "bc"), val_scenes, 1, base_seed=77))
        point = {"epoch": epoch, "loss": loss, "val_success": val["success"]["mean"],
                 "val_reward": val["reward"]["mean"]}
        curve.append(point)
        score = (point["val_success"], point["val_reward"])
        if score > best:
            best = score
            (out / "models").mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), out / "models" / f"bc_{variant}_seed{seed}.pt")
        print(f"[bc seed {seed}] epoch {epoch} loss {loss:.3f} val success {point['val_success']:.3f}",
              flush=True)

    (out / "curves").mkdir(parents=True, exist_ok=True)
    with open(out / "curves" / f"bc_{variant}_seed{seed}.json", "w") as f:
        json.dump({"curve": curve, "demo_episodes": cfg["bc_demo_episodes"],
                   "demo_pairs": int(len(actions)), "demo_planner_expansions": int(expansions),
                   "demo_seconds": demo_seconds, "best_val": best,
                   "train_seconds": time.time() - start}, f, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="main")
    parser.add_argument("--quick", action="store_true")
    a = parser.parse_args()
    main(a.seed, a.quick, a.variant)
