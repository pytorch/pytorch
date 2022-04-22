import torch
from torch.cuda.jiterator import create_jit_fn


code_string = "template <typename T> T python_jitted(T x, T y) { return  - std::abs(x) * std::abs(y); }"

fn = create_jit_fn(code_string, "python_jitted", "elementwise", alpha=1, beta=2)

a = torch.rand(3, dtype=torch.float, device='cuda')
b = torch.rand(3, dtype=torch.float, device='cuda')

c = fn(a, b, beta=3, alpha=-1)
expected =-torch.abs(a) * torch.abs(b)

print(c)
print(expected)

# assert torch.allclose(expected, c)