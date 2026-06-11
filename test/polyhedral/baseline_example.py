import torch

# Standard Llama-3 70B Hidden Dim
HIDDEN_DIM = 8192 
SEQ_LEN = 2048

def rms_norm_residual_block(x, residual, weight):
    # 1. Residual Add (Pointwise)
    # Inductor often fuses this with the previous kernel,
    # but the RMSNorm below usually acts as a "hard stop."
    x = x + residual

    # 2. RMSNorm (Reduction)
    # This is the "Barrier." Inductor creates a reduction kernel here.
    eps = 1e-6
    # Compute variance across the hidden dimension
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps) * weight

    # 3. Post-Reduction Gating (Pointwise)
    # On H200, writing 'x_normed' to HBM just to read it back
    # for this Silu is a massive waste of 4.8 TB/s bandwidth.
    # The .chunk() operation is what torch.compile is NOT optimized for!
    gate, up = x_normed.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up

# Compilation Target
compiled_fn = torch.compile(rms_norm_residual_block, fullgraph=True)

# H200 Setup (using float16 or bfloat16)
device = "cuda"
dtype = torch.bfloat16
x = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
res = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
w = torch.randn(HIDDEN_DIM, device=device, dtype=dtype)

# Warmup
# for _ in range(5):
_ = compiled_fn(x, res, w)
