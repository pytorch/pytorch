# Owner(s): ["module: inductor"]
import copy
import os
import sys
import tempfile
import unittest

import torch
import torch._export
import torch._inductor
import torch.fx._pytree as fx_pytree
from torch._dynamo.testing import same
from torch._inductor import config
from torch._inductor.utils import aot_inductor_launcher

from torch.testing import FileCheck

from torch.testing._internal.common_utils import (
    IS_CI,
    IS_FBCODE,
    IS_WINDOWS,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils import _pytree as pytree

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_torchinductor import (
            copy_tests,
            requires_cuda,
            requires_multigpu,
            TestFailure,
        )
    except ImportError:
        from test_torchinductor import (
            copy_tests,
            requires_cuda,
            requires_multigpu,
            TestFailure,
        )
except (unittest.SkipTest, ImportError) as e:
    if __name__ == "__main__":
        sys.exit(0)
    raise


class AOTInductorModelRunner:
    @classmethod
    def compile(cls, model, example_inputs, options=None, constraints=None):
        # The exact API is subject to change
        so_path, exported = torch._export.aot_compile(
            model,
            example_inputs,
            options=options,
            constraints=constraints,
            remove_runtime_assertions=True,
        )
        return so_path, exported

    @classmethod
    def load(cls, so_path, example_inputs):
        is_cpu = all(x.device.type == "cpu" for x in example_inputs)
        if IS_FBCODE:
            from .fb import test_aot_inductor_model_runner_pybind

            optimized = test_aot_inductor_model_runner_pybind.Runner(
                so_path, is_cpu
            ).run
        else:
            launcher = aot_inductor_launcher
            if is_cpu:
                launcher = launcher.replace("false /*is_cpu*/", "true /*is_cpu*/")

            optimized = torch.utils.cpp_extension.load_inline(
                name="aot_inductor",
                cpp_sources=[launcher],
                # use a unique build directory to avoid test interference
                build_directory=tempfile.mkdtemp(),
                functions=["run"],
                extra_ldflags=[so_path],
                with_cuda=not is_cpu,
            ).run

        return optimized

    @classmethod
    def run_compiled(cls, optimized, exported, example_inputs):
        flat_example_inputs = fx_pytree.tree_flatten_spec(
            (example_inputs, {}), exported.call_spec.in_spec
        )
        output_tensors = optimized(flat_example_inputs)
        return pytree.tree_unflatten(output_tensors, exported.call_spec.out_spec)

    @classmethod
    def run(cls, model, example_inputs, options=None, constraints=None):
        so_path, exported = AOTInductorModelRunner.compile(
            model, example_inputs, options=options, constraints=constraints
        )
        optimized = AOTInductorModelRunner.load(so_path, example_inputs)
        return AOTInductorModelRunner.run_compiled(optimized, exported, example_inputs)

    @classmethod
    def run_multiple(
        cls,
        model,
        list_example_inputs,
        options=None,
        constraints=None,
    ):
        so_path, exported = AOTInductorModelRunner.compile(
            model,
            list_example_inputs[0],
            options=options,
            constraints=constraints,
        )
        optimized = AOTInductorModelRunner.load(so_path, list_example_inputs[0])
        list_output_tensors = []
        for example_inputs in list_example_inputs:
            list_output_tensors.append(
                AOTInductorModelRunner.run_compiled(optimized, exported, example_inputs)
            )
        return list_output_tensors


def check_model(
    self: TestCase,
    model,
    example_inputs,
    options=None,
    constraints=None,
):
    with torch.no_grad(), config.patch(
        "aot_inductor.abi_compatible", self.abi_compatible
    ):
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(example_inputs)
        expected = ref_model(*ref_inputs)
        actual = AOTInductorModelRunner.run(model, example_inputs, options, constraints)

    self.assertTrue(same(actual, expected))


def check_model_with_multiple_inputs(
    self: TestCase,
    model,
    list_example_inputs,
    options=None,
    constraints=None,
):
    with torch.no_grad(), config.patch(
        "aot_inductor.abi_compatible", self.abi_compatible
    ):
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(list_example_inputs)
        list_expected = [ref_model(*inputs) for inputs in ref_inputs]
        list_actual = AOTInductorModelRunner.run_multiple(
            model, list_example_inputs, options, constraints
        )

    self.assertTrue(same(list_actual, list_expected))


class AOTInductorTestsTemplate:
    def test_simple(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_output_path(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch("aot_inductor.output_path", "tmp_output_"):
            self.check_model(Model(), example_inputs)

    @requires_cuda()
    def test_multi_device(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                x = x.cpu()
                x = x + 2
                x = x.cuda()
                return x

        example_inputs = (torch.randn(32, 64, device=self.device),)
        self.check_model(Model(), example_inputs)

    def test_large(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 250112)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 250112, device=self.device),
            torch.randn(1, 512, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_with_offset(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.orig_tensor = torch.randn(2, 15, 10, device=device)[0]
                self.tensor = self.orig_tensor[5:, :]

            def forward(self, x, y):
                return (
                    x
                    + torch.nn.functional.linear(y, self.orig_tensor[:10, :])
                    + self.tensor
                )

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(self.device), example_inputs)

    def test_freezing(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(9, 10, device=device)
                self.padding = torch.randn(1, 10, device=device)

            def forward(self, x, y):
                padded_weight = torch.cat((self.weight, self.padding), dim=0)
                return x + torch.nn.functional.linear(y, padded_weight)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )

        with config.patch({"freezing": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_missing_output(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.cos(b)
                return c

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_output_misaligned(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                x_getitem = cat[0]
                y_getitem = cat[1]
                x_sigmoid = torch.sigmoid(x_getitem)
                return x_sigmoid, y_getitem

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_dynamic_smem_above_default_limit(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x @ y

        model = Model().to(self.device)
        # on A100, the generated Triton kernel for this MM
        # requires 55296 bytes of dynamic SMEM which is above
        # the A100's default dynamic SMEM limit of 49152 bytes.
        example_inputs = (
            torch.randn(10285, 96, device=self.device),
            torch.randn(96, 1, device=self.device),
        )
        self.check_model(
            model,
            example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    def test_seq(self):
        layernorm = torch.nn.LayerNorm(10)
        net = torch.nn.Sequential(
            layernorm,
            torch.nn.ReLU(),
            layernorm,
            torch.nn.ReLU(),
        )

        example_inputs = (torch.randn(10, device=self.device),)
        self.check_model(net.eval(), example_inputs)

    def test_addmm(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        example_inputs = (a,)
        self.check_model(model, example_inputs)

    def test_aliased_buffer_reuse(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = 2 * x
                y = 2 * y
                c = torch.cat([x, y], dim=-1)
                d = 1 + c
                m = torch.mm(d, d)
                return m[:, :2] + x

        example_inputs = (
            torch.randn(4, 2, device=self.device),
            torch.randn(4, 2, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_buffer_reuse(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.cos(y)
                c = torch.mm(a, b)
                d = torch.relu(c)
                e = torch.sigmoid(d)
                f = torch.mm(x, y)
                g = e + f
                return g

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_duplicated_params(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.rand(6))
                self.q = self.p

            def forward(self, x):
                return self.p * x + self.q

        example_inputs = (torch.rand(6, device=self.device),)
        self.check_model(Model(), example_inputs)

    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    def test_inf(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("Inf")
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    def test_nan(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("nan")
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    def test_simple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        a = torch.randn(128, 2048, device=self.device)
        b = torch.randn(128, 2048, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        example_inputs = (a, b)
        self.check_model(Model(), example_inputs, constraints=constraints)

    def test_poi_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        a = torch.randn(128, 2048, device=self.device)
        b = torch.randn(128, 2048, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        list_example_inputs = [(a, b)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),
                torch.randn(64, 2048, device=self.device),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),
                torch.randn(211, 2048, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            Model(), list_example_inputs, constraints=constraints
        )

    def test_addmm_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
        ]
        list_example_inputs = [(a,)]
        batch = 2048
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        batch = 128
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            constraints=constraints,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    def test_bmm_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)

        M = 8
        N = 6
        K = 16
        model = Model()
        batch = 1024
        a = torch.randn(batch, M, K, device=self.device)
        b = torch.randn(batch, K, N, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        list_example_inputs = [(a, b)]
        batch = 2048
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        batch = 128
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
            constraints=constraints,
        )

    def test_foreach_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                return cat

        model = Model()
        a = torch.randn(128, 2048, device=self.device)
        b = torch.randn(128, 2048, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        list_example_inputs = [(a, b)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),
                torch.randn(64, 2048, device=self.device),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),
                torch.randn(211, 2048, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            constraints=constraints,
        )

    # scaled_dot_product_flash_attention
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    def test_sdpa(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)[0]

        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    def test_sdpa_2(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q, k, v, x):
                t = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )[0]
                return x + t

        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_zero_grid_with_unbacked_symbols(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                nz = torch.nonzero(x)
                b = torch.ones_like(nz, dtype=torch.float16)
                c = torch.zeros_like(nz, dtype=torch.float16)
                d = (b + c) @ y
                return d.sum()

        example_inputs = (
            torch.tensor([1, 1, 1], device="cuda"),
            torch.randn((1, 32), dtype=torch.float16, device="cuda"),
        )
        self.check_model(Repro(), example_inputs)

    def test_dynamic_cat(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                return torch.cat([x1, x2], dim=0)

        a = torch.randn(2, 4, device=self.device)
        b = torch.randn(3, 4, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 10,
            torch._export.dynamic_dim(b, 0) >= 1,
            torch._export.dynamic_dim(b, 0) <= 20,
        ]
        example_inputs = (a, b)
        self.check_model(Model(), example_inputs, constraints=constraints)

    @requires_multigpu()
    def test_replicate_on_devices(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self, w1, w2):
                super().__init__()
                self.w1 = w1
                self.w2 = w2

            def forward(self, x, y):
                a = x * self.w1
                b = y * self.w2
                return a + b

        w1 = torch.randn(10, 10)
        w2 = torch.randn(10, 10)
        inputs = (torch.randn(10, 10), torch.randn(10, 10))
        result_cpu = Model(w1, w2)(*inputs)

        # Compile model with AOTInductor
        with torch.cuda.device(0), config.patch(
            "aot_inductor.abi_compatible", self.abi_compatible
        ):
            so_path, exported = AOTInductorModelRunner.compile(
                model=Model(w1.cuda(0), w2.cuda(0)),
                example_inputs=tuple(t.cuda(0) for t in inputs),
            )

        # Run model on cuda:N
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                example_inputs = tuple(t.cuda(i) for t in inputs)
                optimized = AOTInductorModelRunner.load(so_path, example_inputs)
                result_cuda = AOTInductorModelRunner.run_compiled(
                    optimized, exported, example_inputs
                )
            self.assertTrue(same(result_cpu, result_cuda.cpu()))

    @requires_multigpu()
    def test_non_default_cuda_device(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x, y):
                return x + torch.nn.functional.linear(y, self.weight)

        weight = torch.randn(10, 10)
        inputs = (torch.randn(10, 10), torch.randn(10, 10))
        result_cpu = Model(weight)(*inputs)

        with torch.cuda.device(0), torch.no_grad(), config.patch(
            "aot_inductor.abi_compatible", self.abi_compatible
        ):
            result_cuda_0 = AOTInductorModelRunner.run(
                Model(weight.cuda(0)), tuple(t.cuda(0) for t in inputs)
            )

        with torch.cuda.device(1), torch.no_grad(), config.patch(
            "aot_inductor.abi_compatible", self.abi_compatible
        ):
            result_cuda_1 = AOTInductorModelRunner.run(
                Model(weight.cuda(1)), tuple(t.cuda(1) for t in inputs)
            )

        self.assertTrue(same(result_cpu, result_cuda_0.cpu()))
        self.assertTrue(same(result_cpu, result_cuda_1.cpu()))

    def test_reuse_kernel(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.sin(b)
                d = torch.mm(b, c)
                return d

        example_inputs = (
            torch.randn(87, 87, device=self.device),
            torch.randn(87, 87, device=self.device),
        )
        self.check_model(Model(), example_inputs)

        if self.device == "cuda":
            so_path, _ = torch._export.aot_compile(Model(), example_inputs)
            with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
                src_code = cpp.read()
                FileCheck().check_count(
                    "triton_poi_fused_sin_0 = loadKernel(",
                    1,
                    exactly=True,
                ).run(src_code)


class AOTInductorTestABICompatibleCpu(TestCase):
    device = "cpu"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpu,
    "abi_compatible_cpu",
    # test_failures, xfail by default, set is_skip=True to skip
    {
        "test_addmm_multiple_dynamic": TestFailure(("abi_compatible_cpu",)),
        "test_bmm_multiple_dynamic": TestFailure(("abi_compatible_cpu",)),
        "test_dynamic_cat": TestFailure(("abi_compatible_cpu",)),
        "test_dynamic_smem_above_default_limit": TestFailure(("abi_compatible_cpu",)),
        "test_foreach_multiple_dynamic": TestFailure(("abi_compatible_cpu",)),
        # TODO: test_freezing_abi_compatible_cpu somehow fails on CI but not locally,
        #   NotImplementedError: Cannot access storage of OpaqueTensorImpl
        "test_freezing": TestFailure(("abi_compatible_cpu",), is_skip=True),
        "test_poi_multiple_dynamic": TestFailure(("abi_compatible_cpu",)),
        "test_sdpa": TestFailure(("abi_compatible_cpu",)),
        "test_sdpa_2": TestFailure(("abi_compatible_cpu",)),
        "test_simple_dynamic": TestFailure(("abi_compatible_cpu",)),
        "test_zero_grid_with_unbacked_symbols": TestFailure(
            ("abi_compatible_cpu",), is_skip=True
        ),
    },
)


class AOTInductorTestABICompatibleCuda(TestCase):
    device = "cuda"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCuda,
    "abi_compatible_cuda",
    # test_failures, xfail by default, set is_skip=True to skip
    {
        "test_zero_grid_with_unbacked_symbols": TestFailure(
            ("abi_compatible_cuda",), is_skip=True
        ),
    },
)


class AOTInductorTestNonABICompatibleCpu(TestCase):
    device = "cpu"
    abi_compatible = False
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestNonABICompatibleCpu,
    "non_abi_compatible_cpu",
    # test_failures, xfail by default, set is_skip=True to skip
    {
        "test_addmm_multiple_dynamic": TestFailure(("non_abi_compatible_cpu",)),
        "test_bmm_multiple_dynamic": TestFailure(("non_abi_compatible_cpu",)),
        "test_dynamic_smem_above_default_limit": TestFailure(
            ("non_abi_compatible_cpu",)
        ),
        # TODO: test_freezing_non_abi_compatible_cpu somehow fails on CI but not locally,
        #   NotImplementedError: Cannot access storage of OpaqueTensorImpl
        "test_freezing": TestFailure(("non_abi_compatible_cpu",), is_skip=True),
    },
)


class AOTInductorTestNonABICompatibleCuda(TestCase):
    device = "cuda"
    abi_compatible = False
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestNonABICompatibleCuda,
    "non_abi_compatible_cuda",
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_CUDA and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
