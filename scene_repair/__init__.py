"""Budgeted Stochastic Scene Repair: a small RL project for CPS 379.

The package is split into small files, each with one job:
    grid.py         the 5x5 board, directions and simple position helpers
    constraints.py  the three goal types and how to check them
    scenes.py       making random puzzles (scenes) and the train/val/test splits
    env.py          the Gymnasium environment (the rules of the game)
    encoding.py     turning the game state into a list of numbers for a neural net
    networks.py     the small neural networks used by BC and PPO
    evaluation.py   playing many episodes and summarising the results
    policies/       the four players: random, planner, behavior cloning, PPO
"""
