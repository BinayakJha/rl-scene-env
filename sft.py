"""Supervised fine-tuning baseline on gold scene plans (TRL SFTTrainer, LoRA). Runs on Apple Silicon (MPS), CUDA, or CPU.
python sft.py --model Qwen/Qwen2.5-0.5B-Instruct --out ckpt/sft-0.5b
"""
import argparse, json, torch
from device import pick
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from prompt import messages

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--out", default="ckpt/sft")
ap.add_argument("--epochs", type=float, default=2); a = ap.parse_args()
rows = [json.loads(l) for l in open("data/train.jsonl")]
ds = Dataset.from_list([{"messages": messages(r["text"]) + [{"role": "assistant", "content": json.dumps(r["gold"], separators=(",", ":"))}]} for r in rows])
cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4,
                logging_steps=10, save_strategy="epoch", bf16=(pick()[0]=='cuda'), use_cpu=(pick()[0]=='cpu'), report_to="none")
tr = SFTTrainer(model=a.model, args=cfg, train_dataset=ds,
                peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
tr.train(); tr.save_model(a.out)
