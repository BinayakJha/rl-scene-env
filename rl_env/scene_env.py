"""Gymnasium interface for the scene-plan task (CPS 379).

SceneEnv: contextual bandit, horizon 1. obs = spoken line (prompt text),
action = generated completion string (JSON plan). reward = reward.score total,
info = component scores.

TokenSceneEnv: multi-step token-level MDP view of the same task. State = prompt +
tokens so far; action = next token id (Discrete(vocab)); episode ends on EOS or
max_tokens; sparse terminal reward = reward.score(decoded completion). This is the
MDP that GRPO/PPO actually optimise over (per-token log-probs, terminal reward).
"""
import json, os, sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from reward import score  # noqa: E402

MAX_TEXT = 4096


def load_split(split="train"):
    with open(os.path.join(ROOT, "data", f"{split}.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


class SceneEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, split="train", shuffle=True, render_mode=None):
        self.data = load_split(split)
        self.shuffle = shuffle
        self.render_mode = render_mode
        self.observation_space = spaces.Text(max_length=MAX_TEXT)
        self.action_space = spaces.Text(max_length=MAX_TEXT)
        self._i, self._ex = -1, None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "index" in options:
            self._i = int(options["index"])
        elif self.shuffle:
            self._i = int(self.np_random.integers(len(self.data)))
        else:
            self._i = (self._i + 1) % len(self.data)
        self._ex = self.data[self._i]
        return self._ex["text"], {"id": self._ex["id"], "kind": self._ex["kind"], "index": self._i}

    def step(self, action):
        assert self._ex is not None, "call reset() first"
        c = score(action, self._ex["gold"])
        info = dict(c, id=self._ex["id"], kind=self._ex["kind"])
        if self.render_mode == "human":
            print(f"[{self._ex['id']}] {self._ex['text']!r} -> {action!r}  r={c['total']:.3f}")
        self._ex = None
        return "", float(c["total"]), True, False, info  # terminated after 1 step


class CharTokenizer:
    """Fallback tokenizer (printable ASCII + EOS) so TokenSceneEnv runs without HF."""
    def __init__(self):
        self.chars = [chr(i) for i in range(32, 127)]
        self.eos_token_id = len(self.chars)
        self.vocab_size = len(self.chars) + 1
    def encode(self, s): return [self.chars.index(ch) for ch in s if ch in self.chars]
    def decode(self, ids): return "".join(self.chars[i] for i in ids if i < len(self.chars))


class TokenSceneEnv(gym.Env):
    """Token-level MDP. Pass a HF tokenizer (Qwen vocab ~151k) or use CharTokenizer."""
    def __init__(self, split="train", tokenizer=None, max_tokens=96, shuffle=True):
        self.inner = SceneEnv(split, shuffle=shuffle)
        self.tok = tokenizer or CharTokenizer()
        self.max_tokens = max_tokens
        V = getattr(self.tok, "vocab_size", None) or len(self.tok)
        self.action_space = spaces.Discrete(V)
        self.observation_space = spaces.Dict({
            "prompt": spaces.Text(max_length=MAX_TEXT),
            "tokens": spaces.Sequence(spaces.Discrete(V)),
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        prompt, info = self.inner.reset(seed=seed, options=options)
        self.prompt, self.tokens = prompt, []
        return {"prompt": prompt, "tokens": tuple()}, info

    def step(self, action):
        a = int(action)
        done = a == self.tok.eos_token_id
        if not done:
            self.tokens.append(a)
        truncated = (not done) and len(self.tokens) >= self.max_tokens
        obs = {"prompt": self.prompt, "tokens": tuple(self.tokens)}
        if done or truncated:
            _, r, _, _, info = self.inner.step(self.tok.decode(self.tokens))
            return obs, r, done, truncated, info
        return obs, 0.0, False, False, {}
