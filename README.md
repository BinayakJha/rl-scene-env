# Scene Plan RL Environment

A small reinforcement learning environment for my CPS 379 final project.

**The idea:** a language model reads one short spoken sentence (for example, "three chairs around a table") and writes a simple JSON plan of the scene: which objects, how many, where they are, and how to show it. A scorer gives it a reward from 0 to 1.

## How it works (Gymnasium style)

```python
from rl_env.scene_env import SceneEnv

env = SceneEnv(split="test")
obs, info = env.reset()          # obs = one sentence
action = '{"objects": [...]}'    # the model's JSON plan
obs, reward, done, truncated, info = env.step(action)
print(reward, info)              # total reward + score for each part
```

- **One step per episode:** sentence in, plan out, reward, done.
- **Actions are discrete:** the model picks tokens from its vocabulary.
- **Reward:** 15% valid format + 30% correct counts + 30% correct positions + 25% correct display type.

## Files

| File | What it does |
|---|---|
| `rl_env/scene_env.py` | The environment |
| `reward.py` | The scoring function |
| `data/` | 2,000 training and 300 test sentences |
| `rl_env/collect_stats.py` | Runs simple policies and saves stats |
| `sft.py`, `train_grpo.py` | Training scripts (SFT, then GRPO) |
| `tests/` | Tests |

## Run it

```bash
pip install -r requirements.txt
python -m rl_env.collect_stats   # environment statistics
python -m pytest tests           # run tests
```
