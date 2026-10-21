import math
import torch
import torch.nn.functional as F

def get_realistic_inputs(device="cuda"):
    torch.manual_seed(42)
    d_in = 1024
    d_out = 2048
    std = 1.0 / math.sqrt(d_in)
    
    x = torch.randn(4096, d_in, device=device, dtype=torch.float16) * std
    w1 = torch.randn(d_out, d_in, device=device, dtype=torch.float16) * std
    b1 = torch.zeros(d_out, device=device, dtype=torch.float16)
    w2 = torch.randn(d_out, d_in, device=device, dtype=torch.float16) * std
    b2 = torch.zeros(d_out, device=device, dtype=torch.float16)
    return x, w1, b1, w2, b2

def swiglu_reference(x, w1, b1, w2, b2):
    return F.linear(x, w1, b1) * F.silu(F.linear(x, w2, b2))
