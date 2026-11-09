"""Freeze the validation, test and train-check scene sets as JSON files in data/.

Training scenes are NOT stored: PPO and BC draw fresh training scenes from the training
seed range forever. Validation and test are frozen so every policy sees the same scenes.
"""
import argparse
import json

import _setup  # noqa: F401
from scene_repair import settings
from scene_repair.scenes import make_fixed_set, save_scenes, template_key


def main(quick):
    cfg = settings.get(quick)
    out = settings.results_dir(quick) / "data"
    out.mkdir(parents=True, exist_ok=True)
    sets = {
        "val": make_fixed_set("val", cfg["val_scenes"]),
        "test": make_fixed_set("test", cfg["test_scenes"]),
        "train_check": make_fixed_set("train", cfg["train_check_scenes"]),
    }
    for name, scenes in sets.items():
        save_scenes(scenes, out / f"{name}.json")
    summary = {name: {"scenes": len(s), "templates": len({template_key(x.constraints) for x in s})}
               for name, s in sets.items()}
    with open(out / "splits_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("froze splits:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    main(parser.parse_args().quick)
