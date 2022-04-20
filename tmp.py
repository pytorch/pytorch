import torch
from torch.cuda.jiterator import create_jit_fn


code_string = "template <typename T> T abs_add_kernel(T x, T y) { return std::abs(x) + std::abs(y); }"

fn = create_jit_fn(code_string, "abs_add", "elementwise", alpha=1, beta=2)

a = torch.rand(3, dtype=torch.float)
b = torch.rand(3, dtype=torch.float)

fn(a, b, beta=3, alpha=-1)