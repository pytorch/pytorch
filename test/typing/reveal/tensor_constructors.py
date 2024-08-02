# mypy: disable-error-code="possibly-undefined"
# flake8: noqa
import torch
from torch.testing._internal.common_utils import TEST_NUMPY

if TEST_NUMPY:
    import numpy as np

# From the docs, there are quite a few ways to create a tensor:
# https://pytorch.org/docs/stable/tensors.html

# torch.tensor()
reveal_type(torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]))  # E: {Tensor}
reveal_type(torch.tensor([0, 1]))  # E: {Tensor}
reveal_type(
    torch.tensor(
        [[0.11111, 0.222222, 0.3333333]],
        dtype=torch.float64,
        device=torch.device("cuda:0"),
    )
)  # E: {Tensor}
reveal_type(torch.tensor(3.14159))  # E: {Tensor}

# torch.sparse_coo_tensor
i = torch.tensor([[0, 1, 1], [2, 0, 2]])  # E: {Tensor}
v = torch.tensor([3, 4, 5], dtype=torch.float32)  # E: {Tensor}
reveal_type(torch.sparse_coo_tensor(i, v, [2, 4]))  # E: {Tensor}
reveal_type(torch.sparse_coo_tensor(i, v))  # E: {Tensor}
reveal_type(
    torch.sparse_coo_tensor(
        i, v, [2, 4], dtype=torch.float64, device=torch.device("cuda:0")
    )
)  # E: {Tensor}
reveal_type(torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1]))  # E: {Tensor}
reveal_type(
    torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])
)  # E: {Tensor}

# torch.as_tensor
if TEST_NUMPY:
    a = np.array([1, 2, 3])
    reveal_type(torch.as_tensor(a))  # E: {Tensor}
    reveal_type(torch.as_tensor(a, device=torch.device("cuda")))  # E: {Tensor}

# torch.as_strided
x = torch.randn(3, 3)
reveal_type(torch.as_strided(x, (2, 2), (1, 2)))  # E: {Tensor}
reveal_type(torch.as_strided(x, (2, 2), (1, 2), 1))  # E: {Tensor}

# torch.from_numpy
if TEST_NUMPY:
    a = np.array([1, 2, 3])
    reveal_type(torch.from_numpy(a))  # E: {Tensor}

# torch.zeros/zeros_like
reveal_type(torch.zeros(2, 3))  # E: {Tensor}
reveal_type(torch.zeros(5))  # E: {Tensor}
reveal_type(torch.zeros_like(torch.empty(2, 3)))  # E: {Tensor}

# torch.ones/ones_like
reveal_type(torch.ones(2, 3))  # E: {Tensor}
reveal_type(torch.ones(5))  # E: {Tensor}
reveal_type(torch.ones_like(torch.empty(2, 3)))  # E: {Tensor}

# torch.arange
reveal_type(torch.arange(5))  # E: {Tensor}
reveal_type(torch.arange(1, 4))  # E: {Tensor}
reveal_type(torch.arange(1, 2.5, 0.5))  # E: {Tensor}

# torch.range
reveal_type(torch.range(1, 4))  # E: {Tensor}
reveal_type(torch.range(1, 4, 0.5))  # E: {Tensor}

# torch.linspace
reveal_type(torch.linspace(3, 10, steps=5))  # E: {Tensor}
reveal_type(torch.linspace(-10, 10, steps=5))  # E: {Tensor}
reveal_type(torch.linspace(start=-10, end=10, steps=5))  # E: {Tensor}
reveal_type(torch.linspace(start=-10, end=10, steps=1))  # E: {Tensor}

# torch.logspace
reveal_type(torch.logspace(start=-10, end=10, steps=5))  # E: {Tensor}
reveal_type(torch.logspace(start=0.1, end=1.0, steps=5))  # E: {Tensor}
reveal_type(torch.logspace(start=0.1, end=1.0, steps=1))  # E: {Tensor}
reveal_type(torch.logspace(start=2, end=2, steps=1, base=2))  # E: {Tensor}

# torch.eye
reveal_type(torch.eye(3))  # E: {Tensor}

# torch.empty/empty_like/empty_strided
reveal_type(torch.empty(2, 3))  # E: {Tensor}
reveal_type(torch.empty_like(torch.empty(2, 3), dtype=torch.int64))  # E: {Tensor}
reveal_type(torch.empty_strided((2, 3), (1, 2)))  # E: {Tensor}

# torch.full/full_like
reveal_type(torch.full((2, 3), 3.141592))  # E: {Tensor}
reveal_type(torch.full_like(torch.full((2, 3), 3.141592), 2.71828))  # E: {Tensor}

# torch.quantize_per_tensor
reveal_type(
    torch.quantize_per_tensor(
        torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8
    )
)  # E: {Tensor}

# torch.quantize_per_channel
x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
quant = torch.quantize_per_channel(
    x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8
)
reveal_type(x)  # E: {Tensor}

# torch.dequantize
reveal_type(torch.dequantize(x))  # E: {Tensor}

# torch.complex
real = torch.tensor([1, 2], dtype=torch.float32)
imag = torch.tensor([3, 4], dtype=torch.float32)
reveal_type(torch.complex(real, imag))  # E: {Tensor}

# torch.polar
abs = torch.tensor([1, 2], dtype=torch.float64)
pi = torch.acos(torch.zeros(1)).item() * 2
angle = torch.tensor([pi / 2, 5 * pi / 4], dtype=torch.float64)
reveal_type(torch.polar(abs, angle))  # E: {Tensor}

# torch.heaviside
inp = torch.tensor([-1.5, 0, 2.0])
values = torch.tensor([0.5])
reveal_type(torch.heaviside(inp, values))  # E: {Tensor}

# contains
inp = torch.tensor([1, 2, 3])
reveal_type(inp.__contains__(2))  # E: bool
