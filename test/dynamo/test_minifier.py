# Owner(s): ["module: dynamo"]
import unittest

import torch._dynamo
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import skipIfNNModuleInlined


requires_gpu = unittest.skipUnless(
    torch.cuda.is_available() or torch.xpu.is_available(), "requires cuda or xpu"
)


class MinifierTests(MinifierTestBase):
    # Test that compile, runtime, and accuracy errors after dynamo can be repro'd (both CPU and CUDA/XPU)
    def _test_after_dynamo(self, device, backend, expected_error):
        run_code = f"""\
@torch.compile(backend={backend!r})
def inner(x):
    for _ in range(10):
        x = torch.sin(x)
    x = torch.relu(x)
    for _ in range(10):
        x = torch.cos(x)
    return x

inner(torch.randn(20, 20).to("{device}"))
"""
        self._run_full_test(run_code, "dynamo", expected_error, isolate=False)

    def test_after_dynamo_cpu_compile_error(self):
        self._test_after_dynamo(
            "cpu", "relu_compile_error_TESTING_ONLY", "ReluCompileError"
        )

    def test_after_dynamo_cpu_runtime_error(self):
        self._test_after_dynamo(
            "cpu", "relu_runtime_error_TESTING_ONLY", "ReluRuntimeError"
        )

    def test_after_dynamo_cpu_accuracy_error(self):
        self._test_after_dynamo(
            "cpu", "relu_accuracy_error_TESTING_ONLY", "AccuracyError"
        )

    @requires_gpu
    def test_after_dynamo_cuda_compile_error(self, device):
        self._test_after_dynamo(
            device, "relu_compile_error_TESTING_ONLY", "ReluCompileError"
        )

    @requires_gpu
    def test_after_dynamo_cuda_runtime_error(self, device):
        self._test_after_dynamo(
            device, "relu_runtime_error_TESTING_ONLY", "ReluRuntimeError"
        )

    @requires_gpu
    def test_after_dynamo_cuda_accuracy_error(self, device):
        self._test_after_dynamo(
            device, "relu_accuracy_error_TESTING_ONLY", "AccuracyError"
        )

    def test_after_dynamo_non_leaf_compile_error(self):
        run_code = """\
@torch.compile(backend="non_leaf_compile_error_TESTING_ONLY")
def inner(x):
    return x + 1

inner(torch.randn(20, 20, requires_grad=True) + 1)
"""
        self._run_full_test(
            run_code, "dynamo", "TestingOnlyCompileError", isolate=False
        )

    # Ensure that the testing backends pass when relu is not present.
    def _test_after_dynamo_backend_passes(self, device, backend):
        @torch.compile(backend=backend)
        def inner(x):
            for _ in range(10):
                x = torch.sin(x)
            for _ in range(10):
                x = torch.cos(x)
            return x

        inner(torch.randn(20, 20).to(device))

    def test_after_dynamo_cpu_compile_backend_passes(self):
        self._test_after_dynamo_backend_passes("cpu", "relu_compile_error_TESTING_ONLY")

    def test_after_dynamo_cpu_runtime_backend_passes(self):
        self._test_after_dynamo_backend_passes("cpu", "relu_runtime_error_TESTING_ONLY")

    def test_after_dynamo_cpu_accuracy_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cpu", "relu_accuracy_error_TESTING_ONLY"
        )

    @requires_gpu
    def test_after_dynamo_cuda_compile_backend_passes(self, device):
        self._test_after_dynamo_backend_passes(
            device, "relu_compile_error_TESTING_ONLY"
        )

    @requires_gpu
    def test_after_dynamo_cuda_runtime_backend_passes(self, device):
        self._test_after_dynamo_backend_passes(
            device, "relu_runtime_error_TESTING_ONLY"
        )

    @requires_gpu
    def test_after_dynamo_cuda_accuracy_backend_passes(self, device):
        self._test_after_dynamo_backend_passes(
            device, "relu_accuracy_error_TESTING_ONLY"
        )

    # Test that a module with mixed cpu/(cuda|xpu) parts with an error after dynamo can be repro'd
    @skipIfNNModuleInlined()
    @requires_gpu
    def test_cpu_cuda_module_after_dynamo(self, device):
        backend_name = "relu_compile_error_TESTING_ONLY"
        run_code = f"""\
class CpuCudaModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m_x = torch.nn.Linear(20, 20).to(device)
        self.m_y = torch.nn.Linear(20, 20)
        self.p_x = torch.nn.Parameter(torch.randn(20, 20).to(device))
        self.p_y = torch.nn.Parameter(torch.randn(20, 20))
        self.b_x = torch.nn.Buffer(torch.ones(20, 20).to(device))
        self.b_y = torch.nn.Buffer(torch.ones(20, 20))

    def forward(self, x, y):
        return self.m_x(x) + self.p_x + self.b_x, self.m_y(y) + self.p_y + self.b_y

mod = CpuCudaModule()

@torch.compile(backend={backend_name!r})
def inner(x1, y1):
    x2 = torch.randn(20, 20).to(device)
    y2 = torch.randn(20, 20)
    x3, y3 = mod(x1 + x2, y1 + y2)
    return torch.relu(x3.cpu() + y3)

inner(torch.randn(20, 20).to(device), torch.randn(20, 20))
"""

        res = self._run_full_test(run_code, "dynamo", "ReluCompileError", isolate=False)

        self.assertExpectedInline(
            res.minifier_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.G__mod___m_x = Linear(in_features=20, out_features=20, bias=True).to(device)
        self.G__mod___m_y = Linear(in_features=20, out_features=20, bias=True)
        self.register_buffer('G__mod___b_x', torch.randn([20, 20], dtype=torch.float32).to(device))
        self.register_buffer('G__mod___b_y', torch.randn([20, 20], dtype=torch.float32))
        self.G__mod___p_x = torch.nn.Parameter(torch.randn([20, 20], dtype=torch.float32, device=device))
        self.G__mod___p_y = torch.nn.Parameter(torch.randn([20, 20], dtype=torch.float32))

    def forward(self, L_x1_ : torch.Tensor, L_y1_ : torch.Tensor):
        l_x1_ = L_x1_
        l_y1_ = L_y1_
        randn = torch.randn(20, 20)
        x2 = randn.to(device);  randn = None
        y2 = torch.randn(20, 20)
        add = l_x1_ + x2;  l_x1_ = x2 = None
        add_1 = l_y1_ + y2;  l_y1_ = y2 = None
        g__mod___m_x = self.G__mod___m_x(add);  add = None
        g__mod___p_x = self.G__mod___p_x
        add_2 = g__mod___m_x + g__mod___p_x;  g__mod___m_x = g__mod___p_x = None
        g__mod___b_x = self.G__mod___b_x
        x3 = add_2 + g__mod___b_x;  add_2 = g__mod___b_x = None
        g__mod___m_y = self.G__mod___m_y(add_1);  add_1 = None
        g__mod___p_y = self.G__mod___p_y
        add_4 = g__mod___m_y + g__mod___p_y;  g__mod___m_y = g__mod___p_y = None
        g__mod___b_y = self.G__mod___b_y
        y3 = add_4 + g__mod___b_y;  add_4 = g__mod___b_y = None
        cpu = x3.cpu();  x3 = None
        add_6 = cpu + y3;  cpu = y3 = None
        relu = torch.relu(add_6);  add_6 = None
        return (relu,)""",
        )

    # Test if we can actually get a minified graph
    def test_if_graph_minified(self):
        backend_name = "relu_compile_error_TESTING_ONLY"
        run_code = f"""\
@torch.compile(backend={backend_name!r})
def inner(x):
    for _ in range(20):
        x = torch.sin(x)
    x = torch.relu(x)
    for _ in range(20):
        x = torch.cos(x)
    return x

inner(torch.randn(20, 20))
"""

        res = self._run_full_test(run_code, "dynamo", "ReluCompileError", isolate=False)

        self.assertExpectedInline(
            res.repro_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_19):
        x_20 = torch.relu(x_19);  x_19 = None
        return (x_20,)""",
        )


devices = ["cuda", "xpu", "cpu"]
instantiate_device_type_tests(
    MinifierTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
