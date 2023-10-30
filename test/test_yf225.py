import torch
import torch._dynamo.config
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch._dynamo import disable

# TORCH_LOGS="+dynamo,aot,inductor" TORCH_COMPILE_DEBUG=1 python test/test_yf225.py

device = "cuda" if torch.cuda.is_available() else "cpu"

@disable()
def g1_mutation_tuple(d, e):
    d.relu_()
    return d, e

@disable()
def g1_mutation_tensor(d, e):
    d.relu_()
    return d + e

@disable()
def g2(a, b):
    return torch.cat(torch.chunk(a * b, 2))

global_a = torch.randn(4, 4, device=device)

@disable()
def g2_read_global_var(a, b):
    return torch.cat(torch.chunk(a * b.div(torch.selu(global_a)), 2))

def global3(a, b):
    return a + b


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))  # torch.randn(4, 4))
        self.register_buffer('buf', torch.randn(1))  # torch.randn(4, 4))

    @disable()
    def f_read_param_mutate_param(self, c):
        self.buf.relu_()
        return c * c * self.weight

    def f2(self, x, y):
        return x + y

    def subfunc(self, x, y):
        z = torch.relu(x) + g1_mutation_tuple(x, y)[0]
        z = z + g1_mutation_tensor(x, x)
        z = z + g2(x, y)
        z = x + y
        z = z + g2_read_global_var(x, y)
        z = z + self.f_read_param_mutate_param(x)
        z = z + torch.tanh(self.weight)
        z = z + self.buf
        z = z + global_a
        z = z + self.f2(x, y)
        z = z + global3(x, y)
        return z

    def forward(self, x, y):
        x.relu_()
        self.buf.relu_()
        y = torch.cat(torch.chunk(y, 2))
        z = self.subfunc(x, y)
        return z


with (
    torch._dynamo.config.patch(
        dynamic_shapes=False,
        capture_dynamic_output_shape_ops=False,
        capture_scalar_outputs=False,
    ),
):
    torch._dynamo.reset()
    m = TestModule()
    m = m.to(device)
    compiled_m = torch.compile(m, backend="aot_eager" if device == "cpu" else "inductor", fullgraph=False, dynamic=False)
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 4, device=device)

    # ref = m(x, y)
    actual = compiled_m(x, y)
    # assert torch.allclose(ref, actual)
