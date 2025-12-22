import torch
from triton.testing import do_bench

def sample(logits, temperature: float = 0.8):
    logits = logits / max(temperature, 1e-5)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    q = torch.empty_like(logits).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

B = 256
V = 128256
logits = torch.randn(B, V, dtype=torch.bfloat16, device="cuda")
tokens = sample(logits)
compiled_sample = torch.compile(sample)

for _ in range(5):
    compiled_sample(logits)

ms_eager = do_bench(lambda: sample(logits))
ms_compled = do_bench(lambda: compiled_sample(logits))

print(f"{ms_eager=}")
print(f"{ms_compled=}")

# print(tokens)
