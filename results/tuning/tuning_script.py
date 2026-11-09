import time, torch, sys, json
torch.set_num_threads(1)
from scene_repair.networks import PolicyValueNet, NetPolicy
from scene_repair.policies.ppo import PPOConfig, train_ppo
from scene_repair.env import SceneRepairEnv, NUM_ACTIONS
from scene_repair.encoding import OBS_SIZE
from scene_repair.scenes import make_fixed_set
from scene_repair.evaluation import evaluate, summarize
args = json.loads(sys.argv[1])
hidden = args.pop("hidden", 128)
val = make_fixed_set("val", 200)
net = PolicyValueNet(OBS_SIZE, NUM_ACTIONS, hidden=hidden)
cfg = PPOConfig(**args)
t=time.time(); env=SceneRepairEnv()
def cb(steps, log):
    if (steps // 2048) % 50 == 0:
        s = summarize(evaluate(env, NetPolicy(net,"ppo"), val, 1, base_seed=7))
        print(steps, round(time.time()-t), round(log['entropy'],2), round(log['train_success'],3), "VAL", round(s["success"]["mean"],3), round(s["reward"]["mean"],3), flush=True)
train_ppo(net, cfg, seed=0, on_update=cb)
