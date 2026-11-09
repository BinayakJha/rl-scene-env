"""Describe the frozen test scenes: objects, goals, and the shortest noise-free fix length."""
import argparse
import json
from collections import Counter

import _setup  # noqa: F401
from scene_repair import settings
from scene_repair.policies.planner import plan
from scene_repair.scenes import load_scenes


def main(quick):
    out = settings.results_dir(quick)
    scenes = load_scenes(out / "data" / "test.json")
    shortest = []
    for s in scenes:
        state = {"constraints": s.constraints, "layout": s.start_layout, "budget": 8,
                 "undo_buffer": None, "start_budget": 8}
        shortest.append(len(plan(state)[0]))
    stats = {
        "scenes": len(scenes),
        "objects": dict(sorted(Counter(s.num_objects for s in scenes).items())),
        "goals": dict(sorted(Counter(len(s.constraints) for s in scenes).items())),
        "goal_kinds": dict(Counter(c.kind for s in scenes for c in s.constraints)),
        "shortest_noise_free_fix": dict(sorted(Counter(shortest).items())),
        "mean_shortest_fix": sum(shortest) / len(shortest),
    }
    with open(out / "test_scene_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    main(parser.parse_args().quick)
