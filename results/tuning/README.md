# How PPO was tuned (validation only)

All tuning runs used training seed 0 and the first 200 scenes of the frozen **validation**
set (1 rollout each). The test set was never looked at during tuning.
Tuning runs were stopped early once the trend was clear, so they did not all reach their
step budget. Numbers below are copied from the log files in this folder (rounds 2 and 3)
or from the console (round 1, whose logs were overwritten).

## Round 1: the first observation (233 numbers, no "pull" hint)
| Setting | Best val success seen | at steps |
|---|---|---|
| lr 3e-4, entropy 0.01, hidden 128 | 0.30 | 563,200 |
| lr 1e-3, entropy 0.01, hidden 128 | 0.385 | 716,800 |
| lr 1e-3, entropy 0.01, hidden 256 | 0.37 | 716,800 |
| lr 1e-3, entropy 0.003, hidden 128 | 0.355 | 614,400 |
| lr 3e-4, hidden 256, 8 epochs | 0.365 | 614,400 |

Learning was slow everywhere. Diagnosis: the network had to learn "object k's goal is to the
east, so pick action MOVE(k, E)" from raw one-hot positions. Fix: add a per-object **pull**
hint (sum of the gaps of its unmet goals), computed only from C and L (no new information).

## Round 2: with the pull hint (243 numbers), constant learning rate
| Setting | Best val success |
|---|---|
| lr 1e-3, entropy 0.01, hidden 128 | 0.915 |
| lr 3e-4, entropy 0.01, hidden 128 | 0.87 |
| lr 1e-3, entropy 0.03, hidden 128 | 0.915 |
| lr 1e-3, entropy 0.01, hidden 256 | 0.865 |

## Round 3: learning rate shrinks linearly to 0 (3M-step schedule, stopped near 2M)
| Setting | Best val success | Mean of all val checks |
|---|---|---|
| lr 1e-3, entropy 0.01 | 0.91 | 0.801 |
| **lr 1e-3, entropy 0.03 (chosen)** | 0.905 | **0.835** |
| lr 2e-3, entropy 0.01 | 0.90 | 0.830 |
| no annealing, lr 1e-3, 64 envs | 0.895 | 0.816 |

The best scores are all within about 2 points of each other, which is inside the noise of a
200-scene check. We picked lr 1e-3, entropy 0.03, annealed, hidden 128 because it was
steadiest across checks. Final runs: 3,000,000 steps, 3 seeds, best checkpoint by validation.
