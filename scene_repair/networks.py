"""The small neural network shared by behavior cloning and PPO.

    observation (233 numbers)
        -> Linear -> Tanh -> Linear -> Tanh      (the "body": 2 hidden layers of 128)
        -> policy head: 22 scores, one per action ("logits")
        -> value head:  1 number, a guess of the final reward from here (only PPO uses it)

Action masking: scores for switched-off actions are set to a huge negative number, so after
softmax their probability is exactly 0. That is how "pad to 5 objects" stays harmless.
"""
import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size, num_actions, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs, action_mask):
        features = self.body(obs)
        logits = self.policy_head(features)
        logits = logits.masked_fill(~action_mask, -1e9)  # forbidden actions get probability 0
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def distribution(self, obs, action_mask):
        """A categorical (pick-one-of-22) probability distribution, plus the value guess."""
        logits, value = self.forward(obs, action_mask)
        return torch.distributions.Categorical(logits=logits), value


class NetPolicy:
    """Wraps a trained network so it plays like every other policy: act(obs, info) -> action.

    greedy=True takes the most likely action (used for evaluation);
    greedy=False samples from the probabilities.
    """

    def __init__(self, net, name, greedy=True, seed=0):
        self.net = net.eval()
        self.name = name
        self.greedy = greedy
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def act(self, obs, info):
        obs_t = torch.as_tensor(obs).unsqueeze(0)
        mask_t = torch.as_tensor(info["action_mask"]).unsqueeze(0)
        logits, _ = self.net(obs_t, mask_t)
        if self.greedy:
            return int(logits.argmax(dim=-1))
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=self.generator))
