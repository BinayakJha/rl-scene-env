"""GRPO with verifiable rewards (TRL GRPOTrainer). Target: single Apple Silicon Mac (MPS), no vLLM/bitsandbytes; also runs on CUDA.
python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --out ckpt/grpo-0.5b   # from base
python train_grpo.py --model ckpt/sft-0.5b-merged --out ckpt/sft-grpo-0.5b       # SFT -> GRPO
"""
import argparse, json, torch
from device import pick
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from prompt import messages
import reward

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--out", default="ckpt/grpo")
ap.add_argument("--steps", type=int, default=500); ap.add_argument("--num_generations", type=int, default=4)
ap.add_argument("--beta", type=float, default=0.04); a = ap.parse_args()
rows = [json.loads(l) for l in open("data/train.jsonl")]
ds = Dataset.from_list([{"prompt": messages(r["text"]), "gold": json.dumps(r["gold"])} for r in rows])
cfg = GRPOConfig(output_dir=a.out, max_steps=a.steps, learning_rate=1e-5, per_device_train_batch_size=4, gradient_accumulation_steps=2,
                 num_generations=a.num_generations, max_completion_length=96, use_vllm=False, beta=a.beta,
                 logging_steps=5, save_steps=100, bf16=(pick()[0]=='cuda'), use_cpu=(pick()[0]=='cpu'), report_to="none", log_completions=True)
tr = GRPOTrainer(model=a.model, reward_funcs=[reward.reward_total], args=cfg, train_dataset=ds,
                 peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
tr.train(); tr.save_model(a.out)
