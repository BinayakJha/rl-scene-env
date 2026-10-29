import torch
def pick():
    if torch.cuda.is_available(): return "cuda", torch.bfloat16
    if torch.backends.mps.is_available(): return "mps", torch.float32  # fp32 is the safe choice on MPS
    return "cpu", torch.float32
