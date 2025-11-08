# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import contextlib
import dataclasses
import functools
import io
import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from importlib.machinery import SourceFileLoader
from pathlib import Path
from string import Template
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd, config
from torch._dynamo.backends.debugging import aot_eager
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.cpp_builder import is_msvc_cl
from torch._inductor.test_case import run_tests, TestCase
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.overrides import BaseTorchFunctionMode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_S390X,
    IS_WINDOWS,
    parametrize,
    scoped_load_inline,
    skipIfWindows,
)
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.utils._python_dispatch import TorchDispatchMode


# note: these tests are not run on windows due to inductor_utils.HAS_CPU


def make_compiler_fn(
    fullgraph=True, dynamic=True, backend="inductor", gm_hook=lambda gm: None
):
    assert backend in ["inductor", "aot_eager", "eager", "ca_eager"]

    def _compiler_fn(gm):
        """Same as torch.compile() but counts number of compiles"""
        gm_hook(gm)

        _backend = backend
        if backend == "ca_eager":
            return gm
        elif backend != "eager":

            def _inner_compiler(gm_, example_inputs_):
                counters["compiled_autograd"]["compiles"] += 1
                if backend == "inductor":
                    return inductor.compile(gm_, example_inputs_)
                elif backend == "aot_eager":
                    return aot_eager(gm_, example_inputs_)

            _backend = _inner_compiler

        return torch.compile(gm, backend=_backend, fullgraph=fullgraph, dynamic=dynamic)

    return _compiler_fn


compiler_fn = make_compiler_fn()


# TODO(jansel): hooks as lambdas creates recompiles in dynamo, we should fix that
def hook1(grad):
    return grad * 2


def hook2(grads):
    return (grads[0] + 1,)


def hook3(gI, gO):
    return (torch.sin(gI[0]) + gO[0],)


def reset():
    torch._logging.set_logs(compiled_autograd_verbose=False)
    config.compiled_autograd = False
    compiled_autograd.reset()
    torch._dynamo.utils.counters.clear()


class BaseCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("must override")


class TestCompiledAutograd(TestCase):
    def setUp(self) -> None:
        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(config.patch("record_runtime_overhead", False))
        super().setUp()
        reset()

    def tearDown(self) -> None:
        self.exit_stack.close()
        super().tearDown()
        reset()

    def check_output_and_recompiles(
        self, fn, count=1, compiler_fn=compiler_fn, compile_fn=False
    ):
        if isinstance(count, list):
            captures, compiles = count
        else:
            captures, compiles = count, count
        with torch.autograd.set_multithreading_enabled(False):
            torch._dynamo.reset()
            counters["compiled_autograd"].clear()
            torch.manual_seed(123)
            expected = list(fn())
            torch.manual_seed(123)
            with (
                compiled_autograd._enable(compiler_fn),
                mock.patch(
                    "torch._functorch.aot_autograd.AOT_COUNTER",
                    new_callable=itertools.count,
                ),
            ):
                opt_fn = torch.compile(fn) if compile_fn else fn
                actual = list(opt_fn())
            self.assertEqual(expected, actual)
            self.assertEqual(counters["compiled_autograd"]["captures"], captures)
            self.assertEqual(counters["compiled_autograd"]["compiles"], compiles)

    def run_as_subprocess(self, script) -> bytes:
        try:
            return subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            self.fail(f"Subprocess exited with return code: {e.returncode}")

    def test_hipify_not_loaded_with_import_torch(self):
        script = """
import torch
assert globals().get("hipify", False) is False
"""
        self.run_as_subprocess(script)

    def test_hipify_not_loaded_with_import_cpp_extension(self):
        script = """
import torch.utils.cpp_extension
assert globals().get("hipify", False) is False
"""
        self.run_as_subprocess(script)

    def test_dynamo_flaky_segfault(self):
        script = """
import torch

def main():
    def compiler_fn(gm):
        return torch.compile(gm, backend="eager")

    def inner():
        x = torch.randn(1000, 3000)
        w = torch.randn(1000, 3000, requires_grad=True)
        def model(i):
            return torch.nn.functional.linear(i, w)
        out = model(x)
        loss = out.sum()
        with torch._dynamo.compiled_autograd._enable(compiler_fn):
            loss.backward()
        assert(w.grad is not None)

    inner()
    torch._dynamo.reset()
    inner()

main()
        """
        # Run it three times to catch bad dynamo state resets
        for _ in range(3):
            self.run_as_subprocess(script)

    def gen_cache_miss_log_prefix(self):
        if IS_WINDOWS:
            if is_msvc_cl():
                return "Cache miss due to new autograd node: struct "
            else:
                self.fail(
                    "Compilers other than msvc have not yet been verified on Windows."
                )
                return ""
        else:
            return "Cache miss due to new autograd node: "

    def test_reset(self):
        compiled_autograd.compiled_autograd_enabled = True
        torch._C._dynamo.compiled_autograd.set_autograd_compiler(lambda: None, True)
        # TODO: return prior verbose logger
        # torch._C._dynamo.compiled_autograd.set_verbose_logger(dummy)
        compiled_autograd.COMPILE_COUNTER = None

        # state should be clean after reset
        compiled_autograd.reset()

        assert compiled_autograd.compiled_autograd_enabled is False
        (
            prior_compiler,
            prior_dynamic,
        ) = torch._C._dynamo.compiled_autograd.set_autograd_compiler(None, False)
        assert prior_compiler is None
        assert prior_dynamic is False
        assert (
            compiled_autograd.COMPILE_COUNTER is not None
            and next(compiled_autograd.COMPILE_COUNTER) == 0
        )

    def test_basic(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])
            result = model(x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        self.check_output_and_recompiles(fn)

    def test_cache_hit(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])
                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad

        self.check_output_and_recompiles(fn)

    def test_graph_break_custom_op(self):
        @torch.library.custom_op("mylib::sin", mutates_args={})
        def sin(x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        def setup_context(ctx, inputs, output):
            (x,) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * x.cos()

        sin.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        y = sin(x.clone()).sum()
        with compiled_autograd._enable(compiler_fn):
            y.backward()

    def test_tensor_grad_hook1(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])

                model[0].weight.register_hook(hook1)

                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook2(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.grad_fn.register_prehook(hook2)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook3(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.grad_fn.register_hook(hook3)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_reorder_acc_grad(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, bias=True),
            torch.nn.Conv2d(4, 4, 3, bias=True),
        )
        compiled_model = torch.compile(model)
        x = torch.randn([1, 4, 32, 32])

        model(x).sum().backward()
        ref_res = [
            model[0].weight.grad,
            model[0].bias.grad,
            model[1].weight.grad,
            model[1].bias.grad,
        ]

        model[0].weight.grad = None
        model[0].bias.grad = None
        model[1].weight.grad = None
        model[1].bias.grad = None
        with compiled_autograd._enable(compiler_fn):
            compiled_model(x).sum().backward(retain_graph=True)
        res = [
            model[0].weight.grad,
            model[0].bias.grad,
            model[1].weight.grad,
            model[1].bias.grad,
        ]

        self.assertEqual(res[0], ref_res[0])
        self.assertEqual(res[1], ref_res[1])
        self.assertEqual(res[2], ref_res[2])
        self.assertEqual(res[3], ref_res[3])

    def test_reorder_post_hook1(self):
        def grad_div(param):
            param.grad = param.grad / 4.0

        class Module(torch.nn.Module):
            def __init__(self, ioc):
                super().__init__()
                self.fc1 = torch.nn.Linear(ioc, ioc, bias=False)
                self.fc2 = torch.nn.Linear(ioc, ioc, bias=False)

                self.grad_acc_hooks = []
                self.grad_acc = []
                self.params = [self.fc1.weight, self.fc2.weight]
                for param in self.params:

                    def wrapper(param):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def grad_acc_hook(*notneeded):
                            grad_div(param)

                        self.grad_acc.append(grad_acc)
                        self.grad_acc_hooks.append(
                            grad_acc.register_hook(grad_acc_hook)
                        )

                    wrapper(param)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x.sum()

        bs = 8
        ioc = 16
        model = Module(ioc)
        input = torch.randn([bs, ioc])

        # eager ref
        model(input).backward()
        ref_res = [model.fc1.weight.grad, model.fc2.weight.grad]

        # cag
        model.fc1.weight.grad = None
        model.fc2.weight.grad = None
        model_to_train = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            model_to_train(input).backward()
        res = [model_to_train.fc1.weight.grad, model_to_train.fc2.weight.grad]

        self.assertEqual(res[0], ref_res[0])
        self.assertEqual(res[1], ref_res[1])

    def test_reorder_post_hook2(self):
        x = torch.randn([1, 4, 32, 32], requires_grad=True)
        y = torch.sigmoid(x)
        z = torch.tanh(y)

        assert isinstance(z.grad_fn, torch.autograd.graph.Node)
        assert isinstance(y.grad_fn, torch.autograd.graph.Node)
        handle_z = z.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
        handle_y = y.grad_fn.register_hook(lambda gI, gO: (gI[0] * 2,))
        z.sum().backward(retain_graph=True)
        ref_res = x.grad

        x.grad = None
        with compiled_autograd._enable(compiler_fn):
            z.sum().backward(retain_graph=True)
        res = x.grad

        self.assertEqual(res, ref_res)

    def test_reorder_post_hook3(self):
        conv = torch.nn.Conv2d(4, 4, 3, bias=False)
        x = torch.randn([1, 4, 32, 32])
        y = conv(x)

        assert isinstance(y.grad_fn, torch.autograd.graph.Node)
        # this hook will mul 2.0 to the conv weight gradient
        handle_y = y.grad_fn.register_hook(lambda gI, gO: (gI[0], gI[1] * 2, gI[2]))
        y.sum().backward(retain_graph=True)
        ref_res = x.grad

        x.grad = None
        with compiled_autograd._enable(compiler_fn):
            y.sum().backward(retain_graph=True)
        res = x.grad

        self.assertEqual(res, ref_res)

    def test_reorder_all_bwd_hooks(self):
        def tensor_hook(grad):
            return grad.sub(2.0)

        def acc_grad_node_pre_hook(grad_out):
            return (grad_out[0].div(5.0),)

        def post_acc_grad_hook(tensor):
            tensor.grad.add_(3.0)

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]
                self.conv1.weight.register_hook(tensor_hook)
                self.conv1.weight.register_post_accumulate_grad_hook(post_acc_grad_hook)
                self.acc_grad1.register_prehook(acc_grad_node_pre_hook)

                def acc_grad_node_post_hook1(grad_in, grad_out):
                    self.conv1.weight.grad.mul_(0.5)

                self.acc_grad1.register_hook(acc_grad_node_post_hook1)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]
                self.conv2.weight.register_hook(tensor_hook)
                self.conv2.weight.register_post_accumulate_grad_hook(post_acc_grad_hook)
                self.acc_grad2.register_prehook(acc_grad_node_pre_hook)

                def acc_grad_node_post_hook2(grad_in, grad_out):
                    self.conv2.weight.grad.mul_(0.5)

                self.acc_grad2.register_hook(acc_grad_node_post_hook2)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_reorder_multi_post_hooks(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]

                def acc_grad_node1_post_hook1(grad_in, grad_out):
                    self.conv1.weight.grad.mul_(0.5)

                def acc_grad_node1_post_hook2(grad_in, grad_out):
                    self.conv1.weight.grad.sub_(0.3)

                self.acc_grad1.register_hook(acc_grad_node1_post_hook1)
                self.acc_grad1.register_hook(acc_grad_node1_post_hook2)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]

                def acc_grad_node2_post_hook1(grad_in, grad_out):
                    self.conv2.weight.grad.mul_(0.3)

                def acc_grad_node2_post_hook2(grad_in, grad_out):
                    self.conv2.weight.grad.sub_(0.5)

                self.acc_grad2.register_hook(acc_grad_node2_post_hook1)
                self.acc_grad2.register_hook(acc_grad_node2_post_hook2)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_reorder_multi_pre_hooks(self):
        def acc_grad_node_pre_hook1(grad_out):
            return (grad_out[0].div(5.0),)

        def acc_grad_node_pre_hook2(grad_out):
            return (grad_out[0].sub(0.3),)

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]
                self.acc_grad1.register_prehook(acc_grad_node_pre_hook1)
                self.acc_grad1.register_prehook(acc_grad_node_pre_hook2)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]
                self.acc_grad2.register_prehook(acc_grad_node_pre_hook1)
                self.acc_grad2.register_prehook(acc_grad_node_pre_hook2)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_reorder_multi_tensor_pre_hooks(self):
        def tensor_hook1(grad):
            return grad.sub(2.0)

        def tensor_hook2(grad):
            return grad.mul(0.5)

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]
                self.conv1.weight.register_hook(tensor_hook1)
                self.conv1.weight.register_hook(tensor_hook2)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]
                self.conv2.weight.register_hook(tensor_hook1)
                self.conv2.weight.register_hook(tensor_hook2)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_torch_compile(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Sigmoid(),
            )
            opt_model = torch.compile(model, fullgraph=True)

            for _ in range(3):
                x = torch.randn([1, 4])

                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    @parametrize("api", ("compile", "optimize"))
    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_compile_api(self, api, backend):
        def wrap(fn, backend):
            if api == "compile":
                return torch.compile(fn, backend=backend)
            elif api == "optimize":
                return torch._dynamo.optimize(backend)(fn)

        def fn(model, inputs):
            res = []
            for inp in inputs:
                result = model(inp).sum()
                result.backward()
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                model.zero_grad()
            return res

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        inputs = [
            torch.randn([1, 4]),
            torch.randn([2, 4]),
            torch.randn([3, 4]),
        ]

        expected = fn(model, inputs)
        with config.patch(compiled_autograd=True):
            compiled_fn = wrap(fn, backend)
        actual = compiled_fn(model, inputs)
        self.assertEqual(expected, actual)
        self.assertEqual(counters["compiled_autograd"]["captures"], 2)

    @parametrize("api", ("compile", "optimize"))
    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_compile_api_disable(self, api, backend):
        def wrap(fn, backend):
            if api == "compile":
                return torch.compile(fn, backend=backend)
            elif api == "optimize":
                return torch._dynamo.optimize(backend)(fn)

        def fn(model, inputs):
            res = []
            for inp in inputs:
                result = model(inp).sum()
                result.backward()
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                model.zero_grad()
            return res

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        inputs = [
            torch.randn([1, 4]),
            torch.randn([2, 4]),
            torch.randn([3, 4]),
        ]

        expected = fn(model, inputs)
        with config.patch(compiled_autograd=True):
            compiled_fn = wrap(fn, backend)
        with torch._dynamo.compiled_autograd._disable():
            actual = compiled_fn(model, inputs)
        self.assertEqual(expected, actual)
        self.assertTrue("compiled_autograd" not in counters)

    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_optimize_assert(self, backend):
        # can be merged into the test above once we support
        # no graph break on .backward

        def fn(model, inp):
            # NOTE: not calling .backward in the compiled fn
            return model(inp).sum()

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        inp = torch.randn([1, 4])

        out = fn(model, inp)
        out.backward()
        expected = [p.grad for p in model.parameters()]
        model.zero_grad()
        with config.patch(compiled_autograd=True):
            compiled_fn = torch._dynamo.optimize_assert(backend)(fn)

        # should not error due to undefined `rebuild_ctx`
        out = compiled_fn(model, inp)
        out.backward()
        actual = [p.grad for p in model.parameters()]
        self.assertEqual(expected, actual)
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)

    @config.patch(compiled_autograd=True)
    def test_nested_context_manager(self):
        def ctx():
            return compiled_autograd._enable(torch.compile)

        # ok
        outer = ctx()
        inner = ctx()
        outer.__enter__()
        inner.__enter__()
        inner.__exit__(None, None, None)
        outer.__exit__(None, None, None)

        # not ok
        outer = ctx()
        inner = ctx()
        outer.__enter__()
        inner.__enter__()
        with self.assertRaisesRegex(
            AssertionError,
            "Nested Compiled Autograd Contexts must return before their parent context",
        ):
            outer.__exit__(None, None, None)

    @config.patch(compiled_autograd=True)
    def test_nested_compile(self):
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            lib.define("square(Tensor x) -> Tensor")

            @torch.library.impl("testlib::square", "CPU")
            def square_impl(x: torch.Tensor) -> torch.Tensor:
                # nested inference graph compile
                @torch.compile(backend="eager")
                def fn(x):
                    return x**2

                return fn(x)

            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, x):
                    return torch.ops.testlib.square(x)

            x = torch.tensor([2.0, 3.0], requires_grad=True)

            @torch.compile
            def fn(x):
                return MyFn.apply(x)

            fn(x).sum().backward()

    @config.patch(compiled_autograd=True)
    def test_no_nested_compiled_autograd(self):
        # We disable CA before entering the CA graph
        # So re-entrants should be running with the eager autograd engine

        def unrelated_autograd_call():
            x = torch.randn(20, 20, requires_grad=True)
            y = torch.randn(20, 20, requires_grad=True)
            loss = torch.matmul(x, y).sum()
            loss.backward()

        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                unrelated_autograd_call()
                return gO

        x = torch.randn(10, 10, requires_grad=True)
        loss = MyFn.apply(x).sum()

        torch.compile(lambda: loss.backward(create_graph=True))()
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_multiple_torch_compile(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        def fn():
            result = model(x).sum()
            result.backward()

        model2 = torch.nn.Linear(4, 4)
        x2 = torch.randn([1, 4])

        def fn2():
            result = model2(x2).sum()
            result.backward()

        no_ca1 = torch.compile(fn)
        no_ca1()
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        counters.clear()

        with config.patch(compiled_autograd=True):
            with_ca = torch.compile(fn2)
            with_ca()
            self.assertEqual(counters["compiled_autograd"]["captures"], 1)
            counters.clear()

        no_ca2 = torch.compile(fn)
        no_ca2()
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)

    def test_torch_compile_graph_break(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        @torch._dynamo.disable()
        def fn():
            result = model(x).sum()
            result.backward()

        with config.patch(compiled_autograd=True):
            opt_fn = torch.compile(fn)
            opt_fn()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_graph_break2(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        @torch._dynamo.disable()
        def inner_fn(loss):
            loss.backward()

        def fn():
            result = model(x).sum()
            inner_fn(result)

        with config.patch(compiled_autograd=True):
            opt_fn = torch.compile(fn)
            opt_fn()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_only_backward_call(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        result = model(x).sum()
        with config.patch(compiled_autograd=True):
            opt_bwd = torch.compile(lambda: result.backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_dynamo_boxed(self):
        def get_placeholders(gm_):
            placeholders = []
            for node in gm_.graph.nodes:
                if node.op == "placeholder":
                    placeholders.append(node)
            return placeholders

        def eager_with_check(gm, is_bwd):
            def inner_compiler(gm_, example_inputs_):
                placeholders = get_placeholders(gm_)
                if is_bwd:
                    # boxed inputs
                    assert isinstance(placeholders[0].meta["example_value"], list)
                else:
                    # not boxed inputs
                    assert not isinstance(placeholders[0].meta["example_value"], list)

                return gm_

            return torch.compile(gm, backend=inner_compiler)

        bwd_compiler_fn = functools.partial(eager_with_check, is_bwd=True)

        def fn(inputs):
            args_0, args_1, args_2 = inputs
            out = torch.mm(args_0, args_1)
            out = torch.mm(out, args_2)
            loss = out.sum()
            with compiled_autograd._enable(bwd_compiler_fn):
                loss.backward()
            yield args_0.grad
            yield args_1.grad
            yield args_2.grad

        inputs = [
            torch.randn([1, 2], requires_grad=True),
            torch.randn([2, 3], requires_grad=True),
            torch.randn([3, 4], requires_grad=True),
        ]

        compiled_fn = eager_with_check(fn, is_bwd=False)
        grads = list(compiled_fn(inputs))
        self.assertEqual(len(grads), 3)
        self.assertNotEqual(grads[0], None)
        self.assertNotEqual(grads[1], None)
        self.assertNotEqual(grads[2], None)

    def test_inputs_aliasing_bytecode_attr_mutations(self):
        # Freeze compiled autograd graph
        compiler = torch._dynamo.compiled_autograd.AutogradCompilerInstance(compiler_fn)
        param = torch.ones(100)
        active = torch.ones(100) * 2
        inputs = [param, active]
        _, proxies, _, _ = compiler.begin_capture(
            inputs=inputs,
            sizes=[],
            scalars=[],
            origins=[[], [], []],
            accumulate_grad=False,
            check_nans=False,
        )
        param_proxy, activ_proxy = proxies
        buf = activ_proxy * 2
        torch.ops.inductor.accumulate_grad_.default(param_proxy, buf)
        runtime_wrapper, compiled_fn = compiler.end_capture(buf)

        def bytecode_hook(code, out_code):
            import dis
            import sys

            if sys.version_info < (3, 11):
                call_op = "CALL_FUNCTION"
            else:
                call_op = "CALL"

            insts = list(dis.get_instructions(out_code))
            call_graph_idx = next(
                i for i, inst in enumerate(insts) if inst.opname == call_op
            )
            # pre-graph should alias: inputs_ref_0 = inputs[0]
            matches = [
                inst
                for inst in insts[:call_graph_idx]
                if inst.opname == "STORE_FAST" and inst.argval == "inputs_ref_0"
            ]
            self.assertTrue(len(matches) == 1)
            # post-graph should access inputs_ref_0 instead of inputs
            matches = [
                inst for inst in insts[call_graph_idx:] if inst.argval == "inputs"
            ]
            self.assertTrue(len(matches) == 0)
            matches = [
                inst
                for inst in insts[call_graph_idx:]
                if inst.opname == "LOAD_FAST" and inst.argval == "inputs_ref_0"
            ]
            self.assertTrue(len(matches) == 1)

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)
        try:
            runtime_wrapper(
                compiled_fn=compiled_fn,
                inputs=[param, active],
                sizes=(),
                scalars=(),
                hooks=[],
                packed_inputs=[],
            )
        finally:
            handle.remove()

    def test_inputs_aliasing_bytecode_stack_restore(self):
        logging.getLogger().setLevel(logging.WARNING)
        from torch.testing._internal.logging_tensor import LoggingTensor

        # Create a graph that allows inputs stealing
        def forward(inputs):
            add = inputs[0] + 1
            add_1 = add + inputs[1]  # handled in suffix for tensor subclass
            out = add_1.cpu()
            return (out,)

        gm = torch.fx.symbolic_trace(forward)
        torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
        compiled_fn = torch.compile(gm)

        inputs = [
            torch.ones(1000000, dtype=torch.float32),
            LoggingTensor(torch.ones(1)),
        ]
        match_done = False

        def bytecode_hook(code, out_code):
            import dis
            import sys

            nonlocal match_done

            # test is sensitive to what Dynamo traces. So as soon as the main
            # graph is tested, we skip the bytecode hook checks for future
            # frames.
            if not match_done:
                if sys.version_info < (3, 11):
                    call_op = "CALL_FUNCTION"
                else:
                    call_op = "CALL"

                insts = list(dis.get_instructions(out_code))
                call_graph_idx = next(
                    i for i, inst in enumerate(insts) if inst.opname == call_op
                )
                # pre-graph should alias: inputs_ref_0 = inputs[0]
                matches = [
                    inst
                    for inst in insts[:call_graph_idx]
                    if inst.opname == "STORE_FAST" and inst.argval == "inputs_ref_0"
                ]
                self.assertTrue(len(matches) == 1)
                # post-graph should access inputs_ref_0 instead of inputs
                matches = [
                    inst for inst in insts[call_graph_idx:] if inst.argval == "inputs"
                ]
                self.assertTrue(len(matches) == 0)
                matches = [
                    inst
                    for inst in insts[call_graph_idx:]
                    if inst.opname == "LOAD_FAST" and inst.argval == "inputs_ref_0"
                ]
                self.assertTrue(len(matches) == 1)
                match_done = True

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)
        try:
            compiled_fn(inputs)
            self.assertTrue(len(inputs) == 0)
        finally:
            handle.remove()

    def test_implicit_add(self):
        def fn():
            y = torch.randn(1, 4, requires_grad=True)

            def model(x):
                # y is used multiple times, gradients get added
                return torch.sigmoid(x * y + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.backward()
                yield result
                yield y.grad
                y.grad = None

        self.check_output_and_recompiles(fn)

    def test_output_nodes_all_leaves(self):
        def fn():
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                gy, gz = torch.autograd.grad(result, inputs=[y, z])
                assert y.grad is None
                assert z.grad is None
                yield gy
                yield gz

        self.check_output_and_recompiles(fn)

    def test_output_nodes_some_leaves(self):
        def fn():
            class UnreachableBwd(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    raise RuntimeError

            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                return torch.sigmoid(UnreachableBwd.apply(y) * z)

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                gz = torch.autograd.grad(result, inputs=[z])
                assert y.grad is None
                assert z.grad is None
                yield gz

        self.check_output_and_recompiles(fn)

    def test_no_output_nodes_all_leaves(self):
        def fn():
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])
                result = model(x).sum()
                out = result.backward()
                assert out is None
                assert y.grad is not None
                assert z.grad is not None
                yield y.grad
                yield z.grad
                y.grad = None
                z.grad = None

        self.check_output_and_recompiles(fn)

    def test_no_output_nodes_some_leaves(self):
        def fn():
            class UnreachableBwd(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    raise RuntimeError

            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)
            a = torch.randn(1, 4, requires_grad=True)

            def model(x):
                return torch.sigmoid(x * y * z * UnreachableBwd.apply(a))

            for _ in range(3):
                x = torch.randn([1, 4])
                result = model(x).sum()
                out = result.backward(inputs=[y, z])
                assert out is None
                assert y.grad is not None
                assert z.grad is not None
                assert a.grad is None
                yield y.grad
                yield z.grad
                y.grad = None
                z.grad = None

        self.check_output_and_recompiles(fn)

    def test_no_output_nodes_different_leaves_will_recompile(self):
        def fn():
            def fwd(x, y, z):
                out = x * y  # MulBackward0
                out2 = out * z  # MulBackward0
                return out2.sum()  # SumBackward0

            x = torch.randn(5, requires_grad=True)
            y = torch.randn(5, requires_grad=True)
            z = torch.randn(5, requires_grad=True)
            loss = fwd(x, y, z)
            torch.compile(lambda: torch.autograd.backward(loss, inputs=[x]))()
            yield x.grad
            x.grad = None

            loss = fwd(x, y, z)
            torch.compile(lambda: torch.autograd.backward(loss, inputs=[y]))()
            yield y.grad

        # Guarded by TensorArg id, mismatch on last MulBackward0
        self.check_output_and_recompiles(fn, 2)

    def test_dynamic_shapes(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for b in range(10, 100, 10):
                x = torch.randn([b, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    def test_dynamic_shapes_from_forward(self):
        class ToyModel(nn.Module):
            def __init__(self, in_feat=10, hidden_feat=50, out_feat=5):
                super().__init__()
                self.linear1 = nn.Linear(in_feat, hidden_feat)
                self.linear2 = nn.Linear(hidden_feat, hidden_feat)
                self.linear3 = nn.Linear(hidden_feat, out_feat)
                self.mse_loss = torch.nn.MSELoss()

            def forward(self, inputs, output):
                out1 = self.linear1(inputs)
                out2 = self.linear2(out1)
                out3 = self.linear3(out2)
                return self.mse_loss(out3, output)

        m = ToyModel()
        m = torch.compile(m)

        def run(i):
            torch._dynamo.utils.counters.clear()
            inp = torch.randn(i, 10)
            target = torch.randn(i, 5)
            loss = m(inp, target)
            with compiled_autograd._enable(make_compiler_fn(dynamic=None)):
                loss.backward()

        counters = torch._dynamo.utils.counters
        run(3)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 1)
        run(4)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 1)
        run(5)
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 0)
        run(6)
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 0)

    def test_dynamic_shapes_eager_node(self):
        # Here, we have no way of marking the symbolic sizes using in SumBackward as dynamic
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for b, s in zip([10, 20, 30], [2, 4, 8]):
                x = torch.randn([b, 4])
                result = opt_model(x)
                view = result.view(s, -1)
                # sum will save dynamic sizes
                loss = view.sum()
                loss.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    def test_dynamic_shapes_annotations(self):
        @torch.compile
        def f(x):
            return x.sin().sin()

        with torch._dynamo.compiled_autograd._enable(torch.compile):
            x = torch.randn(2, 3, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            out = f(x)
            out.sum().backward()

            x = torch.randn(4, 3, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            out = f(x)
            out.sum().backward()

        # mark_dynamic should not cause ConstraintViolationError
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_api_dynamic_shapes(self):
        # Here, we have no way of marking the symbolic sizes using in SumBackward as dynamic
        def fn(call_backward):
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )

            for b, s in zip([10, 20, 30], [2, 4, 8]):
                x = torch.randn([b, 4])
                result = model(x)
                view = result.view(s, -1)
                # sum will save dynamic sizes
                loss = view.sum()
                call_backward(loss)
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        def call_backward(loss):
            loss.backward()

        eager_out = list(fn(call_backward))
        with config.patch(compiled_autograd=True):
            compiled_out = list(fn(torch.compile(call_backward, dynamic=True)))
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_accumulate_without_zero(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for _ in range(10):
                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad.clone()
                yield model[0].bias.grad.clone()
                yield model[2].weight.grad.clone()
                yield model[2].bias.grad.clone()

        self.check_output_and_recompiles(fn, count=2)

    def test_inplace_grad_update(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for _ in range(10):
                w_grad = torch.rand_like(model[0].weight)
                b_grad = torch.rand_like(model[0].bias)
                model[0].weight.grad = w_grad
                model[0].bias.grad = b_grad

                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                assert model[0].weight.grad is w_grad
                assert model[0].bias.grad is b_grad
                yield w_grad.clone()
                yield b_grad.clone()

        self.check_output_and_recompiles(fn, count=1)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_issue106555(self):
        DEVICE = torch.device(GPU_TYPE, 0)
        NUM_FEATURES = 256

        def bias_sigmoid_mul(x1, x2, bias):
            x2 = torch.sigmoid(x2 + bias)
            y = x1 * x2
            return y

        bias_sigmoid_mul_jit = torch.compile(bias_sigmoid_mul)

        class ModuleWithJit(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear_1 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=True)
                self.linear_2 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=False)
                self.linear_2_bias = nn.Parameter(torch.zeros(NUM_FEATURES))

            def forward(self, input_tensor):
                x1 = self.linear_1(input_tensor)
                x2 = self.linear_2(input_tensor)
                output = bias_sigmoid_mul_jit(x1, x2, self.linear_2_bias)
                return output

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.module_with_jit_1 = ModuleWithJit()
                self.module_with_jit_2 = ModuleWithJit()

            def forward(self, x, gradient_checkpointing: bool):
                if gradient_checkpointing:
                    y = torch.utils.checkpoint.checkpoint(
                        self._forward, x, use_reentrant=True
                    )
                else:
                    y = self._forward(x)
                return y

            def _forward(self, x):
                x = x + self.module_with_jit_1(x)
                x = x + self.module_with_jit_2(x.transpose(-2, -3)).transpose(-2, -3)
                return x

        device_interface = get_interface_for_device(GPU_TYPE)
        device_interface.set_device(device=DEVICE)
        torch.manual_seed(1234567890)
        model = Model()
        model.train()
        model.to(device=DEVICE)
        model_parameters = list(model.parameters())

        torch.manual_seed(1234567890)
        input_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(device=DEVICE)
        input_tensor.requires_grad = True
        target_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(
            dtype=input_tensor.dtype, device=DEVICE
        )

        for _ in range(10):
            for param in model_parameters:
                param.grad = None
            output_tensor = model(
                x=input_tensor.clone(),
                gradient_checkpointing=True,
            )
            loss = torch.mean(torch.abs(target_tensor - output_tensor))
            loss.backward()

    def test_keep_graph_simple(self):
        x = torch.tensor([2.0], requires_grad=True)
        y = x**2

        # First backward pass; keep the computation graph
        y.backward(retain_graph=True)
        self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4

        # Note - this will run under both the eager and compiled regime.
        def fn():
            # Reset the gradients
            x.grad = torch.tensor([0.0])
            # Second and Third backward pass; keep the computation graph
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            return x.grad

        self.check_output_and_recompiles(fn, count=1)

    def test_keep_graph_usage_after_compiled(self):
        x = torch.tensor([2.0], requires_grad=True)
        y = x**2

        # First backward pass; keep the computation graph
        def eager_check():
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            x.grad = torch.tensor([0.0])

        eager_check()

        for _ in range(5):
            with compiled_autograd._enable(compiler_fn):
                eager_check()

            eager_check()

    def test_custom_fn_saved_tensors(self):
        def fn():
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MySin.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_multiple_tensors(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    ctx.save_for_backward(x, y)
                    return torch.sin(x), torch.sin(y)

                @staticmethod
                def backward(ctx, gO_x, gO_y):
                    (x, y) = ctx.saved_tensors
                    return gO_x * torch.cos(x), gO_y * torch.cos(y)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                y = torch.arange(0.0, i, requires_grad=True)
                out1, out2 = MyFn.apply(x, y)
                loss = (out1 * out2).sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_multiple_tensors_dedup(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x, x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    (x1, x2) = ctx.saved_tensors
                    return gO * torch.cos(x1) * torch.cos(x2)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MyFn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_shape_tensor(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    (x,) = ctx.saved_tensors
                    return gO * x.shape[0]

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MyFn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_attr(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.shape = x.shape
                    return x

                @staticmethod
                def backward(ctx, gO):
                    x_shape = ctx.shape[0]
                    return gO * x_shape

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MyFn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_custom_fn_multiple_grads(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    return x + y, y

                @staticmethod
                def backward(ctx, gO_1, gO_2):
                    return gO_1, gO_2

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                y = torch.arange(0.0, i, requires_grad=True)
                out1, out2 = MyFn.apply(x, y)
                loss = (out1 + out2).sum()
                loss.backward()
                yield x.grad
                yield y.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_non_variable_input(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y, z):
                    return x * 2, y * 3, z * 4

                @staticmethod
                def backward(ctx, gO_1, gO_2, gO_3):
                    return gO_1, gO_2, gO_3

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                y = 1
                z = torch.arange(0.0, i, requires_grad=True)
                out1, out2, out3 = MyFn.apply(x, y, z)
                loss = (out1 + out2 + out3).sum()
                loss.backward()
                yield x
                yield y
                yield z

        self.check_output_and_recompiles(fn)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_logging_tensor_flaky(self) -> None:
        # when you first run some test using triton and then run test_inputs_aliasing_bytecode_stack_restore
        # resulting in:
        #   - pytest: `TypeError: unsupported operand type(s) for +: 'Tensor' and 'LoggingTensor'`
        #   - python: `TypeError: not all arguments converted during string formatting`

        # 1. some triton involving test
        def fn():
            def _fn(x):
                return x

            x = torch.arange(
                1, 10, requires_grad=True, dtype=torch.float16, device=GPU_TYPE
            )
            out = _fn(x)
            loss = out.sum()
            loss.backward()

        with compiled_autograd._enable(compiler_fn):
            fn()

        logging.getLogger().setLevel(
            logging.WARNING
        )  # triton setup overwrote it to INFO
        # 2. test_inputs_aliasing_bytecode_stack_restore
        from torch.testing._internal.logging_tensor import LoggingTensor

        def forward(inputs):
            add = inputs[0] + 1
            add_1 = add + inputs[1]
            out = add_1.cpu()
            return (out,)

        gm = torch.fx.symbolic_trace(forward)
        print(gm.print_readable())
        torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
        compiled_fn = torch.compile(gm)

        inputs = [
            torch.ones(1000000, dtype=torch.float32),
            LoggingTensor(torch.ones(1)),
        ]

        compiled_fn(inputs)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_custom_fn_output_metadata(self):
        def my_compiler_fn(gm):
            for node in gm.graph.nodes:
                if isinstance(node.target, torch._ops.OpOverload):
                    assert node.target._name != "aten::_to_copy", (
                        "there should be no implicit copies (e.g. dtype casting)"
                    )

            def inner_compiler(gm_, example_inputs_):
                counters["compiled_autograd"]["compiles"] += 1
                return inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO

            x = torch.arange(
                1, 10, requires_grad=True, dtype=torch.float16, device=GPU_TYPE
            )
            x_view = x.view(3, 3)
            out = MyFn.apply(x_view)
            loss = out.sum()
            loss.backward()
            yield x.dtype
            yield x.device
            yield x.grad

        self.check_output_and_recompiles(fn, count=1)

    def test_custom_fn_with_same_graph(self):
        def fn():
            class MyFn1(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO

            # same as MyFn1, but different autograd function id
            # should not be using same graph as MyFn1
            class MyFn2(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO

            for myfn in [MyFn1, MyFn2, MyFn1, MyFn2]:
                x = torch.arange(0.0, 10, requires_grad=True)
                out = myfn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=2
        )  # should compile once for MyFn1 and once for MyFn2

    def test_custom_fn_dynamically_defined_class(self):
        def fn():
            def create_class(multiplier: int):
                class DynamicFn(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x):
                        return x * multiplier

                    @staticmethod
                    def backward(ctx, gO):
                        return gO * multiplier

                return DynamicFn

            for multiplier in [10, 20, 30]:
                x = torch.arange(0.0, 10, requires_grad=True)
                out = create_class(multiplier).apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn, count=3)

    def test_custom_fn_bw_graph_break(self):
        def fn():
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    print("graph break")
                    return gO * torch.cos(x)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MySin.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=[1, 3], compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_custom_fn_compiled_fw_graph_break(self):
        def fn():
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    print("graph break")
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            opt_model = torch.compile(MySin.apply)
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = opt_model(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=1, compiler_fn=make_compiler_fn(fullgraph=False)
        )
        self.assertEqual(counters["stats"]["unique_graphs"], 4)  # 3 fw, 1 bw

    def test_custom_fn_compiled_fw_bw_graph_break(self):
        def fn():
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    print("graph break")
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            opt_model = torch.compile(MySin.apply)
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = opt_model(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=[1, 3], compiler_fn=make_compiler_fn(fullgraph=False)
        )
        self.assertEqual(counters["stats"]["unique_graphs"], 6)  # 3 fw, 3 bw

    def test_mismatch_fake_tensor_mode(self, dynamic_shape=False):
        """
        Repro the failure of training nanogpt with both compiled-autograd
        and _LazyGraphModule. Check https://github.com/pytorch/pytorch/pull/118981
        for more context.
        """
        B = 8
        x = torch.rand(B, 16)
        y = torch.rand(B, 16, requires_grad=True)

        if dynamic_shape:
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(y, 0)

        def f():
            y.grad = None
            out = x + y

            # make sure the backward call does not trigger any error when
            # compiling the backward graph
            out.sum().backward()
            return out, y.grad

        self.check_output_and_recompiles(f, compile_fn=True)

    def test_mismatch_fake_tensor_mode_dynamic_shape(self):
        self.test_mismatch_fake_tensor_mode(dynamic_shape=True)

    def test_accumulate_grad_accuracy(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 1, bias=False),
                torch.nn.Linear(1, 2, bias=False),
            )
            x = torch.randn(2, 2)

            out = model(x)
            loss = out.sum()
            torch.manual_seed(0)
            loss.backward()

            yield model[0].weight.grad
            yield model[1].weight.grad

        self.check_output_and_recompiles(fn, 1)

    def test_trace_run_with_rng_state(self):
        def sdpa(xq, xk):
            return F.scaled_dot_product_attention(xq, xk, xk, is_causal=True)

        def g(xq_1, xk_1, xq_2, xk_2):
            # xq: (bs, n_local_heads, seqlen, head_dim)
            # xk: (bs, n_local_heads, cache_len + seqlen, head_dim)
            y1 = sdpa(xq_1, xk_1)
            y2 = torch.utils.checkpoint.checkpoint(
                sdpa, xq_2, xk_2, use_reentrant=False
            )
            y = torch.mul(y1, y2)
            z = torch.matmul(y, y)
            return z

        def f():
            bs = 1
            n_local_heads = 1
            seqlen = 2
            head_dim = 2
            cache_len = 2
            xq_list = [
                torch.ones(
                    (bs, n_local_heads, seqlen, head_dim),
                    requires_grad=True,
                    device="cpu",
                )
                for _ in range(2)
            ]
            xk_list = [
                torch.ones(
                    (bs, n_local_heads, cache_len + seqlen, head_dim),
                    requires_grad=True,
                    device="cpu",
                )
                for _ in range(2)
            ]
            out = torch.compile(g, fullgraph=True)(
                xq_list[0], xk_list[0], xq_list[1], xk_list[1]
            )
            out.sum().backward()
            return out, *[x.grad for x in xq_list + xk_list]

        """
        Walkthrough of what happens with `run_with_rng_state`:
        1. `run_with_rng_state` only shows up in the backward graph (this op is inserted by the partitioner).
        2. The Dynamo graph captured by Compiled Autograd looks like:
        ```
        ===== __compiled_fn_3 =====
        torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
            def forward(self, L_inputs_ : list):
                ...
                run_with_rng_state = torch.ops.higher_order.run_with_rng_state(
                    getitem_8,
                    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
                    getitem_3, getitem_4, getitem_4, 0.0, True,
                )
                ...
        ```
        3. We want to preserve this `run_with_rng_state` op when going through AOTAutograd. We do it by having special handling
        in `run_with_rng_state` op's py_functionalize_impl.
        """

        def _run_with_rng_state_op_check(inductor_post_grad_graph):
            # Checks that `run_with_rng_state` op exists in Compiled Autograd's Inductor post-grad graph.
            op_set = {node.target for node in inductor_post_grad_graph.nodes}
            if torch.ops.higher_order.run_and_save_rng_state not in op_set:
                # This is backward graph, so check existence of `run_with_rng_state` op
                self.assertTrue(torch.ops.higher_order.run_with_rng_state in op_set)

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=_run_with_rng_state_op_check
        ):
            compiler_fn = make_compiler_fn(fullgraph=True)

            def make_compiler_fn_with_op_check():
                def _compiler_fn(gm):
                    # Checks that `run_with_rng_state` op exists in Compiled Autograd's Dynamo graph.
                    self.assertTrue(
                        any(
                            node.target is torch.ops.higher_order.run_with_rng_state
                            for node in gm.graph.nodes
                        )
                    )
                    return compiler_fn(gm)

                return _compiler_fn

            compiler_fn_with_op_check = make_compiler_fn_with_op_check()
            self.check_output_and_recompiles(
                f, compiler_fn=compiler_fn_with_op_check, compile_fn=False
            )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_trace_auto_functionalized_v2(self):
        self.trace_auto_functionalized_base()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_trace_auto_functionalized(self):
        self.trace_auto_functionalized_base()

    def trace_auto_functionalized_base(self):
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            torch.library.define(
                "testlib::foo",
                "(Tensor(a!) x) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "testlib::foo_mutated",
                "(Tensor(a!) x) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("testlib::foo", "cpu", lib=lib)
            def foo(x):
                x.add_(5)
                return x

            @torch.library.impl("testlib::foo", "Meta", lib=lib)
            def foo_meta(x):
                return x

            @torch.library.impl(
                "testlib::foo_mutated", "CompositeImplicitAutograd", lib=lib
            )
            def foo_mutated(x):
                return torch.ops.testlib.foo(x)

            def _get_custom_policy(must_recompute_list=None):
                def _custom_policy(ctx, func, *args, **kwargs):
                    if must_recompute_list is not None and func in must_recompute_list:
                        return torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE
                    else:
                        return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

                return _custom_policy

            def context_fn():
                must_recompute_list = [
                    torch.ops.higher_order.auto_functionalized,
                ]
                return torch.utils.checkpoint.create_selective_checkpoint_contexts(
                    _get_custom_policy(
                        must_recompute_list=must_recompute_list,
                    ),
                )

            def g(x):
                x = torch.matmul(x, x)
                torch.ops.testlib.foo_mutated(x)
                return torch.matmul(x, x)

            def g_cp(x):
                return torch.utils.checkpoint.checkpoint(
                    g, x, use_reentrant=False, context_fn=context_fn
                )

            def f():
                inps = (torch.randn(4, 4, requires_grad=True),)
                output = torch.compile(g_cp, backend="aot_eager", fullgraph=True)(*inps)
                output.sum().backward()
                return output, inps[0].grad

            """
            Walkthrough of what happens with `auto_functionalized`:
            1. `auto_functionalized` op is inserted into the graph during AOTAutograd functionalization.
            We force the op to be recomputed (by using SAC), so it appears in the backward graph.
            2. The AOT backward graph looks like:
            ```
            ===== Backward graph 0 =====
            def forward(self, primals_1: "f32[4, 4][4, 1]cpu", tangents_1: "f32[4, 4][4, 1]cpu"):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = mm)
                ...
                return (add_1,)
            ```
            3. The Compiled Autograd graph looks like:
            ```
            ===== Compiled autograd graph =====
            def forward(self, inputs, sizes, scalars, hooks):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = aot0_mm)
                ...
                return []
            ```
            4. The Dynamo graph captured by Compiled Autograd looks like:
            ```
            ===== __compiled_fn_3 =====
            def forward(self, L_inputs_ : list):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = aot0_mm)
                ...
                return (new_grad,)
            ```
            5. The Compiled Autograd's AOT "forward-only" graph looks like:
            ```
            ===== Forward graph 1 =====
            def forward(self, arg0_1: "f32[][]cpu", arg1_1: "f32[4, 4][4, 1]cpu"):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = mm)
                ...
                return (clone_1,)
            ```
            6. The `auto_functionalized` op should then be lowered using the normal lowering path in Inductor.
            """

            compiler_fn = make_compiler_fn(fullgraph=True, backend="aot_eager")

            def make_compiler_fn_with_op_check():
                def _compiler_fn(gm):
                    auto_functionalize_func = (
                        torch.ops.higher_order.auto_functionalized
                        if not torch._inductor.config.enable_auto_functionalized_v2
                        else torch.ops.higher_order.auto_functionalized_v2
                    )

                    # Checks that `auto_functionalized` op exists in Compiled Autograd's Dynamo graph.
                    self.assertTrue(
                        any(
                            node.target is auto_functionalize_func
                            for node in gm.graph.nodes
                        ),
                        f"{auto_functionalize_func} op not found in {gm.graph}",
                    )
                    return compiler_fn(gm)

                return _compiler_fn

            compiler_fn_with_op_check = make_compiler_fn_with_op_check()
            self.check_output_and_recompiles(
                f, compiler_fn=compiler_fn_with_op_check, compile_fn=False
            )

    @scoped_load_inline
    def test_autograd_cpp_node_non_traceable(self, load_inline):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = false;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_non_traceable_autograd_cpp_node, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = load_inline(
            name="test_non_traceable_autograd_cpp_node",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            x = torch.ones(10, 10, requires_grad=True)
            out = module.custom_op_backed_by_autograd_fn(x)
            loss = out.sum()
            loss.backward()
            yield x.grad

        # should not raise
        self.check_output_and_recompiles(
            fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
        )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_basic(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_basic_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_basic",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn, 1)
        else:
            # compiles for 10 (static) and 100 (dynamic), each with a graph break
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_id(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

struct CustomOpAutogradFunction2 : public torch::autograd::Function<CustomOpAutogradFunction2> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

torch::Tensor custom_op_backed_by_autograd_fn2(torch::Tensor x) {
  return CustomOpAutogradFunction2::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_id_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("custom_op_backed_by_autograd_fn2", custom_op_backed_by_autograd_fn2);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_id",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions=[
                "custom_op_backed_by_autograd_fn",
                "custom_op_backed_by_autograd_fn2",
            ],
            verbose=True,
        )

        def same_autograd_fn():
            def fn():
                x = torch.ones(10, 10, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

            yield from fn()  # compile
            yield from fn()  # reuse
            yield from fn()  # reuse
            yield from fn()  # reuse

        if is_traceable:
            self.check_output_and_recompiles(same_autograd_fn, 1)
        else:
            self.check_output_and_recompiles(
                same_autograd_fn,
                count=[1, 2],
                compiler_fn=make_compiler_fn(fullgraph=False),
            )

        def different_autograd_fn():
            def fn(op):
                x = torch.ones(10, 10, requires_grad=True)
                out = op(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

            op1 = module.custom_op_backed_by_autograd_fn
            op2 = module.custom_op_backed_by_autograd_fn2
            yield from fn(op1)  # compile
            yield from fn(op2)  # compile
            yield from fn(op1)  # reuse
            yield from fn(op2)  # reuse

        if is_traceable:
            self.check_output_and_recompiles(different_autograd_fn, 2)
        else:
            # ????
            self.check_output_and_recompiles(
                same_autograd_fn,
                count=[1, 2],
                compiler_fn=make_compiler_fn(fullgraph=False),
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_basic(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y,
      const torch::Tensor& fixed) {
    ctx->save_for_backward({x, y});
    ctx->saved_data["fixed_tensor"] = fixed;
    ctx->saved_data["bool"] = true;
    ctx->saved_data["int"] = 1;
    c10::List<std::string> list({"string"});
    ctx->saved_data["list"] = std::move(list);
    c10::Dict<std::string, double> dict;
    dict.insert("string", 1.0);
    ctx->saved_data["dict"] = std::move(dict);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor y = saved_variables[1];
    torch::Tensor fixed = ctx->saved_data["fixed_tensor"].toTensor();
    assert(ctx->saved_data["bool"].isBool());
    c10::SymInt i = ctx->saved_data["int"].toSymInt();
    c10::List<c10::IValue> list = ctx->saved_data["list"].toList();
    assert(list.size() == 1);
    assert(list.get(0).toStringRef() == "string");
    c10::Dict<c10::IValue, c10::IValue> dict = ctx->saved_data["dict"].toGenericDict();
    assert(dict.size() == 1);
    assert(dict.at("string") == 1.0);

    torch::autograd::variable_list grad_inputs(3);
    grad_inputs[0] = x + y + torch::sum(fixed) + i;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& fixed) {
  return CustomOpAutogradFunction::apply(x, y, fixed);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_basic_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_saved_basic",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            fixed = torch.ones(2, 2)
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                y = torch.randn(i, i)
                out = module.custom_op_backed_by_autograd_fn(x, y, fixed)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn, 1)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_dynamic(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    ctx->save_for_backward({x});
    ctx->saved_data["dynamic"] = x.view(-1);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    torch::Tensor z = ctx->saved_data["dynamic"].toTensor();

    torch::autograd::variable_list grad_inputs(1);
    grad_inputs[0] = x + torch::sum(z);
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_dynamic_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_saved_dynamic",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        # compiles for 10 (static) and 100 (dynamic)
        if is_traceable:
            self.check_output_and_recompiles(fn, 1)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_int(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      int64_t y) {
    ctx->save_for_backward({x});
    ctx->saved_data["int"] = y;
    ctx->saved_data["symint"] = c10::SymInt(y);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    c10::SymInt y = ctx->saved_data["int"].toSymInt();
    c10::SymInt ys = ctx->saved_data["symint"].toSymInt();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + y + ys;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, int64_t y) {
  return CustomOpAutogradFunction::apply(x, y);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_int_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_saved_int",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for y in [1, 2, 3, 1]:
                x = torch.ones(10, 10, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x, y)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_float(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      double z) {
    ctx->save_for_backward({x});
    ctx->saved_data["float"] = z;
    ctx->saved_data["symfloat"] = c10::SymFloat(z);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    c10::SymFloat z = ctx->saved_data["float"].toSymFloat();
    c10::SymFloat zs = ctx->saved_data["symfloat"].toSymFloat();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + z + zs;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, double z) {
  return CustomOpAutogradFunction::apply(x, z);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_float_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_saved_float",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for z in [1.1, 2.2, 3.3, 1.1]:
                x = torch.ones(10, 10, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x, z)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            # compiled autograd and dynamo both support symfloat, but not backend
            self.check_output_and_recompiles(fn, [1, 4])
            # 1 restart analysis due to specialize_float=False
            self.assertEqual(counters["stats"]["unique_graphs"], 3)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 3], compiler_fn=make_compiler_fn(fullgraph=False)
            )
            self.assertEqual(counters["stats"]["unique_graphs"], 2)

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_data_dependent(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;
  static int iteration;

  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y) {
    ctx->save_for_backward({x, y});
    ctx->saved_data["bool"] = true;
    ctx->saved_data["int"] = 1;

    switch (iteration) {
        case 0: {
            break;
        }
        case 1: {
            // recompile
            ctx->saved_data["forces_recompile"] = iteration;
            break;
        }
        case 2: {
            // recompile
            ctx->set_materialize_grads(false);
            break;
        }
        case 3: {
            // reuse
            break;
        }
        default: {
            throw std::runtime_error("unexpected iteration");
        }
    }
    iteration++;
    return {x, y};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor y = saved_variables[1];
    c10::SymInt i = ctx->saved_data["int"].toSymInt();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + y + i;
    return grad_inputs;
  }
};

int CustomOpAutogradFunction::iteration = 0;

torch::autograd::variable_list custom_op_backed_by_autograd_fn(const torch::Tensor& x, const torch::Tensor& y) {
  return CustomOpAutogradFunction::apply(x, y);
}

void reset() {
    CustomOpAutogradFunction::iteration = 0;
}

TORCH_LIBRARY(test_autograd_cpp_node_data_dependent_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("reset", reset);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_data_dependent",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions=["custom_op_backed_by_autograd_fn", "reset"],
            verbose=True,
        )

        def fn():
            module.reset()
            for i in [10, 10, 10, 10]:
                x = torch.ones(i, i, requires_grad=True)
                y = torch.randn(i, i)
                (
                    out1,
                    out2,
                ) = module.custom_op_backed_by_autograd_fn(x, y)
                loss = (out1 + out2).sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn, 3)
        else:
            self.check_output_and_recompiles(
                fn, count=[3, 6], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_free_activation_memory(self):
        script = """
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.testing._internal.inductor_utils import GPU_TYPE

def main():
    device_interface = get_interface_for_device(GPU_TYPE)
    assert(device_interface.memory_allocated() == 0)

    # Use an op to check that the memory is freed by the time the op is executed
    def assertion_impl(to_clone):
        mem_allocated = device_interface.memory_allocated()
        assert mem_allocated < 4000000  # some activations should be freed
        return to_clone.clone()

    with torch.library._scoped_library("test_compiled_autograd", "FRAGMENT") as lib:
        lib.define(
            "assertion_op(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,)
        )
        lib.impl("assertion_op", assertion_impl, "CPU")
        lib.impl("assertion_op", lambda x: x.clone(), "Meta")

        # Create a graph that allows inputs stealing
        def forward(activations):
            add = activations[0] + 1
            out = add.cpu()
            cloned_out = torch.ops.test_compiled_autograd.assertion_op(out)
            return (cloned_out,)

        gm = torch.fx.symbolic_trace(forward)
        torch._dynamo.utils.set_locals_to_steal(gm, ["activations"])
        compiled_fn = torch.compile(gm)

        # allocate at least 4,000,000 bytes (1,000,000 * 4 bytes)
        activations = [torch.ones(1000000, dtype=torch.float32, device=GPU_TYPE)]
        assert device_interface.memory_allocated() > 4000000

        out = compiled_fn(activations)
        assert len(activations) == 0

main()
        """
        self.run_as_subprocess(script)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_free_activation_memory_subclass(self):
        # cover the case when aot inputs have subclasses, resulting in a different runtime wrapper

        script = """
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.testing._internal.inductor_utils import GPU_TYPE

def main():
    device_interface = get_interface_for_device(GPU_TYPE)
    assert device_interface.memory_allocated() == 0

    # Use an op to check that the memory is freed by the time the op is executed
    def assertion_impl(to_clone):
        mem_allocated = device_interface.memory_allocated()
        assert mem_allocated < 1200000  # some activations should be freed
        assert mem_allocated > 800000  # currently subclasses don't seem to be freed in inductor
        return to_clone.clone()

    with torch.library._scoped_library("test_compiled_autograd", "FRAGMENT") as lib:
        lib.define(
            "assertion_op(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,)
        )
        lib.impl("assertion_op", assertion_impl, "CPU")
        lib.impl("assertion_op", lambda x: x.clone(), "Meta")
        lib.impl("assertion_op", lambda x: x.clone(), "NestedTensor")

        def fn(inputs):
            _, y = inputs
            out = y.cpu()
            cloned_out = torch.ops.test_compiled_autograd.assertion_op(out)
            return cloned_out

        gm = torch.fx.symbolic_trace(fn)
        torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
        compiled_fn = torch.compile(gm)

        from torch.nested._internal.nested_tensor import jagged_from_list

        activations = [
            jagged_from_list(
                [
                    torch.ones((1, 100000), device=GPU_TYPE),  # 400,000 bytes
                    torch.ones((1, 100000), device=GPU_TYPE),  # 400,000 bytes
                ],
                None,
            )[
                0
            ],  # NestedTensor
            torch.ones((1, 100000), device=GPU_TYPE),  # 400,000 bytes
        ]
        # 1,200,000 bytes (3 * 4 * 100,000 bytes)
        assert device_interface.memory_allocated() > 1200000

        out = compiled_fn(activations)
        assert len(activations) == 0

main()
        """
        self.run_as_subprocess(script)

    def test_callback_graph_break_throws_error(self):
        called = [0]

        def callback_final():
            called[0] += 1

        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad):
                torch.autograd.Variable._execution_engine.queue_callback(callback_final)
                torch._dynamo.graph_break()
                return grad

        a = torch.rand((3, 3), requires_grad=True)
        with self.assertRaisesRegex(
            AssertionError,
            "only supported when Compiled Autograd is enabled with fullgraph=True",
        ):
            with compiled_autograd._enable(make_compiler_fn(fullgraph=False)):
                b = MyFunc.apply(a)
                b.sum().backward()

    @requires_cuda_and_triton
    def test_cudagraphs_cpu_division(self):
        from torch._dynamo.testing import reduce_to_scalar_loss

        model = torch.nn.Linear(10, 10, dtype=torch.float16).cuda()
        inputs = torch.randn(10, 10, dtype=torch.float16).cuda()
        out = model(inputs)
        loss = reduce_to_scalar_loss(out)

        stderr_msgs = io.StringIO()
        with (
            mock.patch("sys.stderr", stderr_msgs),
            compiled_autograd._enable(compiler_fn),
        ):
            torch._inductor.config.triton.cudagraphs = True
            loss.backward()
            torch._inductor.config.triton.cudagraphs = False

        if inductor_config.cpp_wrapper:
            self.assertIn("skipping cudagraphs", stderr_msgs.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
        else:
            self.assertNotIn("skipping cudagraphs", stderr_msgs.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    def test_cudagraphs_cpu_graph(self):
        from torch._dynamo.testing import reduce_to_scalar_loss

        model = torch.nn.Linear(10, 10, dtype=torch.float16)
        inputs = torch.randn(10, 10, dtype=torch.float16)
        out = model(inputs)
        loss = reduce_to_scalar_loss(out)

        with compiled_autograd._enable(compiler_fn):
            torch._inductor.config.triton.cudagraphs = True
            loss.backward()
            torch._inductor.config.triton.cudagraphs = False

        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

    @requires_cuda_and_triton
    def test_cudagraphs_sdpa(self):
        query = torch.rand(
            32, 8, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True
        )
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        with (
            config.patch(compiled_autograd=True),
            inductor_config.patch("triton.cudagraphs", True),
        ):
            opt_bwd = torch.compile(lambda: out.sum().backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(
            counters["inductor"]["cudagraph_skips"],
            2 if inductor_config.cpp_wrapper else 0,
        )

    @requires_cuda_and_triton
    def test_cudagraphs_cpu_scalar_used_in_python_custom_op(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                cpu_tensor = torch.tensor(5)
                ctx.save_for_backward(x, cpu_tensor)  # visible to c++/autograd
                ctx.cpu_scalar = 5  # opaque to c++/autograd
                return x.sum()

            @staticmethod
            def backward(ctx, gO):
                x, cpu_tensor = ctx.saved_tensors
                expand = gO * torch.ones_like(x)
                return expand * cpu_tensor * ctx.cpu_scalar

        x = torch.randn(10, requires_grad=True, device="cuda")
        out = MyFn.apply(x)
        with (
            config.patch(compiled_autograd=True),
            inductor_config.patch("triton.cudagraphs", True),
        ):
            opt_bwd = torch.compile(lambda: out.backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        # Compiled autograd lifts custom autograd.Function bwd instead of tracing it.
        # Must skip since we do not know if the cpu scalar will be used only in ATen/prim ops.
        if inductor_config.graph_partition:
            # instead of skipping cudagraph, graph partition splits off cpu inputs/outputs and ops
            # and cudagraphify the remaining computation. So there is no cudagraph skip.
            expected_cudagraph_skips = 0
        else:
            expected_cudagraph_skips = 1

        self.assertEqual(
            counters["inductor"]["cudagraph_skips"], expected_cudagraph_skips
        )

    @scoped_load_inline
    @requires_cuda_and_triton
    def test_cudagraphs_cpu_scalar_used_in_cpp_custom_op(self, load_inline):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    const auto& cpu_tensor = torch::tensor(1);
    ctx->save_for_backward({x, cpu_tensor});
    ctx->saved_data["cpu_scalar"] = 1;
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor cpu_tensor = saved_variables[1];
    int cpu_scalar = ctx->saved_data["cpu_scalar"].toInt();
    auto expand = grad_output[0] * torch::ones_like(x);
    torch::autograd::variable_list grad_inputs(1);
    grad_inputs[0] = expand * cpu_tensor * cpu_scalar;  // autograd engine asserts that tensors are on same device
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_cudagraphs_cpu_scalar_used_in_cpp_custom_op, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = load_inline(
            name="test_cudagraphs_cpu_scalar_used_in_cpp_custom_op",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        x = torch.randn(2, 2, requires_grad=True, device="cuda")
        with (
            config.patch(compiled_autograd=True),
            inductor_config.patch("triton.cudagraphs", True),
        ):
            out = torch.ops.test_cudagraphs_cpu_scalar_used_in_cpp_custom_op.custom_op_backed_by_autograd_fn(
                x
            )
            opt_bwd = torch.compile(lambda: out.sum().backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        # Compiled autograd's initial capture lifts custom C++ autograd::Function bwd instead of tracing
        # into it. We must skip since we do not know if the cpu scalar will be used only in ATen/prim ops.
        # In the future, we can consider having a cpu scalar movement pass sometime after we trace
        # into the custom C++ autograd::Function (like in AOTDispatcher)
        if inductor_config.graph_partition:
            # instead of skipping cudagraph, graph partition splits off cpu inputs/outputs and ops
            # and cudagraphify the remaining computation. So there is no cudagraph skip.
            expected_cudagraph_skips = 0
        elif inductor_config.cpp_wrapper:
            expected_cudagraph_skips = 2
        else:
            expected_cudagraph_skips = 1

        self.assertEqual(
            counters["inductor"]["cudagraph_skips"],
            expected_cudagraph_skips,
        )

    def test_logs(self):
        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd"
        )
        with compiled_autograd._enable(compiler_fn), ctx():
            torch.randn(4, 4, requires_grad=True).sum().backward()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 1)
        assert "torch::autograd::AccumulateGrad (NodeCall" in logs.getvalue()
        assert (
            self.gen_cache_miss_log_prefix() + "torch::autograd::GraphRoot"
            not in logs.getvalue()
        )

    def test_logs_aot_bwd_reuse(self):
        @torch.compile(backend="aot_eager")
        def fn(x):
            return x.sum()

        with compiled_autograd._enable(compiler_fn):
            x = torch.randn(4, 4, requires_grad=True)
            y = torch.randn(4, 4, requires_grad=True)
            z = torch.randn(4, 4, requires_grad=True)
            # reuse the same AOT bwd graph 3 times
            out = fn(x) + fn(y) + fn(z)
            out.backward()
        # should not RuntimeError: Node redefined name aot0_expand!

    def test_verbose_logs_graph(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])
            result = model(x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            self.check_output_and_recompiles(fn)

        expected_logs = [
            "torch::autograd::GraphRoot (NodeCall 0)",
            "ReluBackward0 (NodeCall 2)",
            "AddmmBackward0 (NodeCall 3)",
            "ReluBackward0 (NodeCall 5)",
            "TBackward0 (NodeCall 6)",
            "torch::autograd::AccumulateGrad (NodeCall 7)",
            "torch::autograd::AccumulateGrad (NodeCall 9)",
            "TBackward0 (NodeCall 10)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
            "SumBackward0 (NodeCall 1)",
            "ReluBackward0 (NodeCall 2)",
            "AddmmBackward0 (NodeCall 3)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
            "TBackward0 (NodeCall 4)",
            "torch::autograd::AccumulateGrad (NodeCall 5)",
            "ReluBackward0 (NodeCall 6)",
            "AddmmBackward0 (NodeCall 7)",
            "torch::autograd::AccumulateGrad (NodeCall 10)",
            "TBackward0 (NodeCall 8)",
            "torch::autograd::AccumulateGrad (NodeCall 9)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
        ]

        found = 0
        for line in logs.getvalue().split("\n"):
            if found == len(expected_logs):
                break
            if expected_logs[found] in line:
                found += 1

        self.assertEqual(found, len(expected_logs))

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    @mock.patch("torch._dynamo.config.inline_inbuilt_nn_modules", True)
    def test_verbose_logs_aot_id(self, _):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])

            @torch.compile
            def forward(model, x):
                return model(x)

            result = forward(model, x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            self.check_output_and_recompiles(fn)

        expected_logs = [
            "code: CompiledFunctionBackward (NodeCall 2)",
        ]

        found = 0
        for line in logs.getvalue().split("\n"):
            if found == len(expected_logs):
                break
            if expected_logs[found] in line:
                found += 1

        self.assertEqual(found, len(expected_logs))

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_verbose_logs_aot_dispatcher_nodes(self, _):
        def fn():
            @torch.compile
            def f(x):
                tmp1 = x.sin()
                tmp2 = x.cos()
                torch._dynamo.graph_break()
                return tmp1.sin() + tmp2.cos()

            x = torch.randn(4, requires_grad=True)
            out = f(x)
            out.sum().backward()
            yield x.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            self.check_output_and_recompiles(fn)

        expected_logs = [
            "CompiledFunctionBackward1",
            "aot1_sin_1",
            "aot1_neg",
            "aot0_tangents_2",
            "aot1_cos_1",
            "aot0_tangents_1",
            "CompiledFunctionBackward0",
            "aot0_sin_1",
            "aot0_neg",
            "aot0_mul",
            "aot0_cos_1",
            "aot0_mul_1",
            "aot0_add",
        ]

        self.assertEqual(
            sum(1 for e in expected_logs if e in logs.getvalue()), len(expected_logs)
        )

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_verbose_logs_aot_dispatcher_nodes_hop(self, _):
        @dataclasses.dataclass
        class CustomObj:
            val: torch.Tensor

        def fn(x, obj):
            y = x.sin()
            closure_var = y + 1
            y.register_hook(lambda grad: grad + obj.val + closure_var)
            z = y.sin()
            return z

        opt_fn = torch.compile(fn)

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=True)
        obj = CustomObj(torch.tensor(88))
        fn(x, obj).sum().backward()

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx(), compiled_autograd._enable(compiler_fn):
            opt_fn(y, obj).sum().backward()
        self.assertEqual(x.grad, y.grad)

        expected_logs = [
            "CompiledFunctionBackward0",
            "aot0_primals_2",
            "aot0_tangents_2",
            "aot0_tangents_1",
            "aot0_sin",
            "aot0_cos",
            "aot0_mul",
            "aot0_add_1",
            "aot0_trace_wrapped",
            "aot0_cos_1",
            "aot0_mul_1",
        ]

        self.assertEqual(
            sum(1 for e in expected_logs if e in logs.getvalue()), len(expected_logs)
        )

    def test_verbose_logs_cpp(self):
        torch._logging.set_logs(compiled_autograd_verbose=True)

        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            for i in [10, 11, 12]:
                model.zero_grad()
                x = torch.randn([i, 4])
                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            self.check_output_and_recompiles(fn)

        patterns1 = [
            r".*"
            + self.gen_cache_miss_log_prefix()
            + r"torch::autograd::GraphRoot \(NodeCall 0\) with key size (\d+), previous key sizes=\[\]\n",
        ]

        all_logs = logs.getvalue()

        pattern1 = r"".join(patterns1)
        matches1 = re.findall(pattern1, all_logs)
        self.assertEqual(len(matches1), 1)
        assert isinstance(
            matches1[0], str
        )  # for a single match: matches1=['match'], for multiple matches: matches1=[('match1', 'match2')]...
        self.assertEqual(len(matches1), len(patterns1))

    @skipIfWindows(msg="node name demangling inconsistent on windows")
    def test_verbose_logs_dynamic_shapes(self):
        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )

        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
        )

        for i, j in zip([10, 11, 12], [10, 10, 11]):
            model.zero_grad()
            x = torch.randn([i, 4])
            y = torch.randn([j, 4])
            result = model(x).sum() + model(y).sum()
            with ctx(), compiled_autograd._enable(torch.compile(backend="eager")):
                result.backward()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

        actual_logs = logs.getvalue()
        expected_logs = [
            self.gen_cache_miss_log_prefix()
            + "torch::autograd::GraphRoot (NodeCall 0) with key size 39, previous key sizes=[]",
        ]
        for expected in expected_logs:
            self.assertTrue(expected in actual_logs)

    def test_verbose_logs_snapshot(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])
            result = model(x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            with compiled_autograd._enable(compiler_fn):
                # unused, verbose level already snapshot with contextmanager
                torch._logging.set_logs(compiled_autograd_verbose=True)
                fn()

        unexpected_logs = [
            self.gen_cache_miss_log_prefix() + "torch::autograd::GraphRoot (NodeCall 0)"
        ]

        self.assertEqual(sum(1 for e in unexpected_logs if e in logs.getvalue()), 0)

    def test_tensor_subclass_basic(self):
        from torch.testing._internal.two_tensor import TwoTensor, TwoTensorMode

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("to_twotensor(Tensor a, Tensor b) -> Tensor")
            lib.define("from_twotensor(Tensor c) -> (Tensor, Tensor)")

            def to_twotensor_backward(ctx, grad):
                return torch.ops.mylib.from_twotensor(grad)

            def from_twotensor_backward(ctx, grad_a, grad_b):
                raise AssertionError("shouldn't get hit")

            torch.library.register_autograd(
                "mylib::to_twotensor", to_twotensor_backward, lib=lib
            )
            torch.library.register_autograd(
                "mylib::from_twotensor", from_twotensor_backward, lib=lib
            )

            @torch.library.register_torch_dispatch(
                "mylib::to_twotensor", TwoTensorMode, lib=lib
            )
            def _(_0, _1, _2, args, kwargs):
                assert not kwargs
                a, b = args
                return TwoTensor(a.clone(), b.clone())

            @torch.library.register_torch_dispatch(
                "mylib::from_twotensor", TwoTensor, lib=lib
            )
            def _(_0, _1, _2, args, kwargs):
                assert not kwargs
                (c,) = args
                return c.a.clone(), c.b.clone()

            @torch.compile(backend="aot_eager", fullgraph=True)
            def fn(x):
                return x * x + 2

            param1 = torch.randn(4, 4, requires_grad=True)
            param2 = torch.randn(4, 4, requires_grad=True)
            with TwoTensorMode():
                x = torch.ops.mylib.to_twotensor(param1, param2)

            inner_compiler_fn = make_compiler_fn(fullgraph=True, backend="aot_eager")
            graphs = []

            def compiler_fn(gm):
                graphs.append(gm)
                return inner_compiler_fn(gm)

            with (
                compiled_autograd._enable(compiler_fn),
                mock.patch(
                    "torch._functorch.aot_autograd.AOT_COUNTER",
                    new_callable=itertools.count,
                ),
            ):
                res = fn(x)
                res.sum().backward()

            self.assertEqual(param1.grad, 2 * param1)
            self.assertEqual(param2.grad, 2 * param2)
            self.assertEqual(len(graphs), 1)

            graph_code = normalize_gm(graphs[0].print_readable(print_output=False))
            # The graph should have make_subclass calls in it.
            self.assertExpectedInline(
                graph_code,
                """\
class CompiledAutograd0(torch.nn.Module):
    def forward(self, inputs, sizes, scalars, hooks, packed_data):
        getitem = inputs[0]
        getitem_1 = inputs[1]
        getitem_2 = inputs[2]
        getitem_3 = inputs[3]
        getitem_4 = inputs[4];  inputs = None
        getitem_5 = sizes[0]
        getitem_6 = sizes[1]
        getitem_7 = sizes[2]
        getitem_8 = sizes[3]
        getitem_21 = sizes[4]
        getitem_22 = sizes[5]
        getitem_23 = sizes[6]
        getitem_24 = sizes[7];  sizes = None
        unwrap_maybe_dynamic_int = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_5);  getitem_5 = None
        unwrap_maybe_dynamic_int_1 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_6);  getitem_6 = None
        unwrap_maybe_dynamic_int_2 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_7);  getitem_7 = None
        unwrap_maybe_dynamic_int_3 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_8);  getitem_8 = None
        unwrap_maybe_dynamic_int_16 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_21);  getitem_21 = None
        unwrap_maybe_dynamic_int_17 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_22);  getitem_22 = None
        unwrap_maybe_dynamic_int_18 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_23);  getitem_23 = None
        unwrap_maybe_dynamic_int_19 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_24);  getitem_24 = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [], True, 6)]);  getitem = None
        getitem_25 = validate_outputs[0];  validate_outputs = None

        sum_backward0 = torch__dynamo_compiled_autograd_ops_SumBackward0([getitem_25], [True], [unwrap_maybe_dynamic_int, unwrap_maybe_dynamic_int_1]);  getitem_25 = unwrap_maybe_dynamic_int = unwrap_maybe_dynamic_int_1 = None
        getitem_26 = sum_backward0[0];  sum_backward0 = None
        validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_26], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_2, unwrap_maybe_dynamic_int_3], True, 6)]);  getitem_26 = unwrap_maybe_dynamic_int_2 = unwrap_maybe_dynamic_int_3 = None
        getitem_27 = validate_outputs_1[0];  validate_outputs_1 = None

        getitem_28 = hooks[0];  getitem_28 = None
        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((getitem_1, getitem_2), [], getitem_27);  getitem_1 = getitem_2 = getitem_27 = None
        aot0_primals_1 = call_aot_bwd_prologue[0]
        aot0_primals_2 = call_aot_bwd_prologue[1]
        aot0_tangents_1 = call_aot_bwd_prologue[2]
        aot0_tangents_2 = call_aot_bwd_prologue[3];  call_aot_bwd_prologue = None

        aot0_mul_2 = torch.ops.aten.mul.Tensor(aot0_tangents_1, aot0_primals_1);  aot0_tangents_1 = aot0_primals_1 = None
        aot0_mul_3 = torch.ops.aten.mul.Tensor(aot0_tangents_2, aot0_primals_2);  aot0_tangents_2 = aot0_primals_2 = None
        aot0_add_2 = torch.ops.aten.add.Tensor(aot0_mul_2, aot0_mul_2);  aot0_mul_2 = None
        aot0_add_3 = torch.ops.aten.add.Tensor(aot0_mul_3, aot0_mul_3);  aot0_mul_3 = None

        make_subclass = torch__dynamo_compiled_autograd_make_subclass(aot0_add_2, aot0_add_3);  aot0_add_2 = aot0_add_3 = None

        getitem_33 = hooks[1];  hooks = None
        call_backward = torch__dynamo_external_utils_call_backward(getitem_33, (), make_subclass);  getitem_33 = make_subclass = None
        getitem_36 = call_backward[0]
        getitem_37 = call_backward[1];  call_backward = None
        validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_36, getitem_37], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_16, unwrap_maybe_dynamic_int_17], False, 6), ((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_18, unwrap_maybe_dynamic_int_19], False, 6)]);  getitem_36 = getitem_37 = unwrap_maybe_dynamic_int_16 = unwrap_maybe_dynamic_int_17 = unwrap_maybe_dynamic_int_18 = unwrap_maybe_dynamic_int_19 = None
        getitem_39 = validate_outputs_2[0]

        call_accumulate_grad_1 = torch__dynamo_external_utils_call_accumulate_grad(getitem_4, getitem_39, False);  getitem_4 = getitem_39 = call_accumulate_grad_1 = None

        getitem_40 = validate_outputs_2[1];  validate_outputs_2 = None

        call_accumulate_grad = torch__dynamo_external_utils_call_accumulate_grad(getitem_3, getitem_40, False);  getitem_3 = getitem_40 = call_accumulate_grad = None

        _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
        return []
""",  # noqa: B950
            )

    # https://github.com/pytorch/pytorch/issues/138920
    # Inductor has a joint graph pattern to remove pointless view pairs.
    # That will remove the no-op view pairs this test is checking. Disable
    # pattern matcher for this test.
    @inductor_config.patch(pattern_matcher=False)
    def test_compiled_autograd_does_not_specialize_on_bw_symints(self):
        class Mod(torch.nn.Module):
            def __init__(self, a, b, c):
                super().__init__()
                self.a = a
                self.c = c
                self.b = b
                self.lin1 = torch.nn.Linear(b * a, b * c, device="cpu")

            def forward(self, x):
                x = x.view(-1, self.a * self.b)
                y = self.lin1(x)
                y = y.view(-1, self.c, self.b).contiguous()
                y = torch.flatten(y, start_dim=1)
                return y

        class Mod2(torch.nn.Module):
            def __init__(self, a, b, c):
                super().__init__()
                self.mod = Mod(a, b, c)

            def forward(self, s, tensor_dict):
                args = tensor_dict[s]
                x = torch.cat(list(args))
                out = self.mod(x)
                return out

        class Mod3(torch.nn.Module):
            def __init__(self, mods):
                super().__init__()
                self.mods = mods

            def forward(self, strs, tensor_dict, x):
                outs = [x]
                for i, m in enumerate(self.mods):
                    s = strs[i]
                    print("graph break")
                    out = m(s, tensor_dict)
                    outs.append(out)
                return torch.cat(outs).sum(0)

        def gen_tensor_dict(sizes):
            tensor_dict = {
                "a": [torch.randn(sizes[0], 48, device="cpu") for _ in range(4)],
                "b": [torch.randn(sizes[1], 48, device="cpu") for _ in range(7)],
            }
            return tensor_dict

        mods = [
            Mod2(192, 1, 48),
            Mod2(336, 1, 48),
        ]
        m = Mod3(mods)

        strs = ["a", "b"]

        m = torch.compile(m)

        graphs = []

        def compiler_fn(gm):
            def inner_compiler(gm_, example_inputs_):
                graphs.append(gm_)
                return gm_

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        x = torch.zeros(100, 48, device="cpu")
        tensor_dict = gen_tensor_dict([101, 102])
        out = m(strs, tensor_dict, x)

        with torch._dynamo.compiled_autograd._enable(compiler_fn) as ctx:
            out.sum().backward()

        x = torch.zeros(103, 48, device="cpu")
        tensor_dict = gen_tensor_dict([104, 105])
        out = m(strs, tensor_dict, x)

        with torch._dynamo.compiled_autograd._enable(compiler_fn) as ctx:
            out.sum().backward()

        # This test is a bit fragile (I failed to create a better repro).
        # The important bit is that the second CA graph has not specialized the value
        # of aot4_sym_size_int_ to a constant.
        # This happens via suppressing any dynamic shape guards that CA generates
        # when it runs make_fx.
        # Suppressing these guards is strictly better than the current state,
        # because we ignore all of these guards anyway in CA.
        # Once we stop using make_fx in CA, we won't have to worry about this specialization.
        view_nodes = graphs[1].graph.find_nodes(
            op="call_function", target=torch.ops.aten.reshape.default
        )
        # First 2 view nodes have a first argument that is a SymInt, not an int burned into the graph
        self.assertTrue(isinstance(view_nodes[0].args[1][0], torch.fx.Node))
        self.assertTrue(isinstance(view_nodes[1].args[1][0], torch.fx.Node))

    @requires_cuda_and_triton
    def test_flex_attention(self):
        def _squared(score, b, h, m, n):
            """Joint graph needed for correctness"""
            return score * score

        def fn():
            @torch.compile(backend="aot_eager")
            def fwd_bwd(x: torch.Tensor):
                flex_attention(x, x, x, score_mod=_squared).sum().backward()

            for a, b in zip([12, 24, 12], [64, 128, 64]):
                v = torch.zeros(
                    1,
                    1,
                    a * b,
                    b,
                    dtype=torch.bfloat16,
                    device="cuda",
                    requires_grad=True,
                )
                fwd_bwd(v)
                yield v.grad

        self.check_output_and_recompiles(
            fn, count=2, compiler_fn=make_compiler_fn(backend="aot_eager")
        )

    def test_saved_tensor_unpack_hook_ordering(self):
        def f(x, y):
            return x * y

        pack_count = 0
        unpack_count = 0

        def pack_hook(x):
            nonlocal pack_count
            pack_count += 1
            return x

        def unpack_hook(x):
            nonlocal unpack_count
            unpack_count += 1
            return x

        def tensor_hook(_):
            self.assertEqual(unpack_count, 0)

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with (
            torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook),
            compiled_autograd._enable(make_compiler_fn(fullgraph=False)),
        ):
            out_test = f(x, y)
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 0)
            loss = out_test.sum()
            loss.register_hook(
                tensor_hook
            )  # scheduled to fire before any saved activations
            loss.backward()
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 1)

    @parametrize("reentrant", (True, False))
    def test_checkpointing_simple(self, reentrant):
        def fn():
            def _fn(x):
                y = x.sin()
                z = y.cos()
                return (y * z).sum()

            inp = torch.rand(10, 10, requires_grad=True)
            out = torch.utils.checkpoint.checkpoint(_fn, inp, use_reentrant=reentrant)
            out.backward()
            yield inp.grad

        if reentrant:
            self.check_output_and_recompiles(
                fn, count=[1, 3], compiler_fn=make_compiler_fn(fullgraph=False)
            )
        else:
            # dynamo issues, just run the CA graph directly for now
            def check(gm):
                graph_code = normalize_gm(gm.print_readable(print_output=False))
                self.assertExpectedInline(
                    graph_code,
                    """\
class CompiledAutograd0(torch.nn.Module):
    def forward(self, inputs, sizes, scalars, hooks, packed_data):
        getitem = inputs[0]
        getitem_1 = inputs[1];  inputs = None
        getitem_2 = sizes[0]
        getitem_3 = sizes[1]
        getitem_4 = sizes[2]
        getitem_5 = sizes[3]
        getitem_6 = sizes[4]
        getitem_7 = sizes[5]
        getitem_8 = sizes[6]
        getitem_9 = sizes[7]
        getitem_10 = sizes[8]
        getitem_11 = sizes[9]
        getitem_12 = sizes[10]
        getitem_13 = sizes[11];  sizes = None
        unwrap_maybe_dynamic_int = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_2);  getitem_2 = None
        unwrap_maybe_dynamic_int_1 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_3);  getitem_3 = None
        unwrap_maybe_dynamic_int_2 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_4);  getitem_4 = None
        unwrap_maybe_dynamic_int_3 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_5);  getitem_5 = None
        unwrap_maybe_dynamic_int_4 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_6);  getitem_6 = None
        unwrap_maybe_dynamic_int_5 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_7);  getitem_7 = None
        unwrap_maybe_dynamic_int_6 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_8);  getitem_8 = None
        unwrap_maybe_dynamic_int_7 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_9);  getitem_9 = None
        unwrap_maybe_dynamic_int_8 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_10);  getitem_10 = None
        unwrap_maybe_dynamic_int_9 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_11);  getitem_11 = None
        unwrap_maybe_dynamic_int_10 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_12);  getitem_12 = None
        unwrap_maybe_dynamic_int_11 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_13);  getitem_13 = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [], False, 6)]);  getitem = None
        getitem_14 = validate_outputs[0];  validate_outputs = None

        sum_backward0 = torch__dynamo_compiled_autograd_ops_SumBackward0([getitem_14], [True], [unwrap_maybe_dynamic_int, unwrap_maybe_dynamic_int_1]);  getitem_14 = unwrap_maybe_dynamic_int = unwrap_maybe_dynamic_int_1 = None
        getitem_15 = sum_backward0[0];  sum_backward0 = None
        validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_15], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_2, unwrap_maybe_dynamic_int_3], False, 6)]);  getitem_15 = unwrap_maybe_dynamic_int_2 = unwrap_maybe_dynamic_int_3 = None
        getitem_16 = validate_outputs_1[0];  validate_outputs_1 = None

        getitem_17 = hooks[0]
        getitem_18 = packed_data[0]
        getitem_19 = hooks[1]
        getitem_20 = packed_data[1]
        call_hook = torch__dynamo_external_utils_call_hook(getitem_17, getitem_18, hook_type = 'unpack_hook');  getitem_17 = getitem_18 = None
        call_hook_1 = torch__dynamo_external_utils_call_hook(getitem_19, getitem_20, hook_type = 'unpack_hook');  getitem_19 = getitem_20 = None
        mul_backward0 = torch__dynamo_compiled_autograd_ops_MulBackward0([getitem_16], [True, True], call_hook, 6, call_hook_1, 6);  getitem_16 = call_hook = call_hook_1 = None
        getitem_21 = mul_backward0[0]
        getitem_22 = mul_backward0[1];  mul_backward0 = None
        validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_21, getitem_22], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_4, unwrap_maybe_dynamic_int_5], False, 6), ((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_6, unwrap_maybe_dynamic_int_7], False, 6)]);  getitem_21 = getitem_22 = unwrap_maybe_dynamic_int_4 = unwrap_maybe_dynamic_int_5 = unwrap_maybe_dynamic_int_6 = unwrap_maybe_dynamic_int_7 = None
        getitem_23 = validate_outputs_2[0]
        getitem_24 = validate_outputs_2[1];  validate_outputs_2 = None

        getitem_25 = hooks[2]
        getitem_26 = packed_data[2]
        call_hook_2 = torch__dynamo_external_utils_call_hook(getitem_25, getitem_26, hook_type = 'unpack_hook');  getitem_25 = getitem_26 = None
        cos_backward0 = torch__dynamo_compiled_autograd_ops_CosBackward0([getitem_24], [True], call_hook_2);  getitem_24 = call_hook_2 = None
        getitem_27 = cos_backward0[0];  cos_backward0 = None
        validate_outputs_3 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_27], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_8, unwrap_maybe_dynamic_int_9], False, 6)]);  getitem_27 = unwrap_maybe_dynamic_int_8 = unwrap_maybe_dynamic_int_9 = None
        getitem_28 = validate_outputs_3[0];  validate_outputs_3 = None
        add = torch.add(getitem_23, getitem_28);  getitem_23 = getitem_28 = None

        getitem_29 = hooks[3];  hooks = None
        getitem_30 = packed_data[3];  packed_data = None
        call_hook_3 = torch__dynamo_external_utils_call_hook(getitem_29, getitem_30, hook_type = 'unpack_hook');  getitem_29 = getitem_30 = None
        sin_backward0 = torch__dynamo_compiled_autograd_ops_SinBackward0([add], [True], call_hook_3);  add = call_hook_3 = None
        getitem_31 = sin_backward0[0];  sin_backward0 = None
        validate_outputs_4 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_31], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_10, unwrap_maybe_dynamic_int_11], False, 6)]);  getitem_31 = unwrap_maybe_dynamic_int_10 = unwrap_maybe_dynamic_int_11 = None
        getitem_32 = validate_outputs_4[0];  validate_outputs_4 = None

        call_accumulate_grad = torch__dynamo_external_utils_call_accumulate_grad(getitem_1, getitem_32, False);  getitem_1 = getitem_32 = call_accumulate_grad = None
        _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
        return []
""",  # noqa: B950
                )

            self.check_output_and_recompiles(
                fn,
                count=[1, 0],
                compiler_fn=make_compiler_fn(backend="ca_eager", gm_hook=check),
            )

    @requires_cuda_and_triton
    def test_cpu_offloading(self):
        def fn():
            def pack(x):
                return x.cpu()

            def unpack(x):
                return x.cuda()

            class MyMatMul(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return torch.matmul(x, x)

                @staticmethod
                def backward(ctx, grad_out):
                    (x,) = ctx.saved_tensors
                    return grad_out * x

            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                for i in [10, 100, 10, 20, 30]:
                    x = torch.randn(i, requires_grad=True).cuda()
                    MyMatMul.apply(x).sum().backward()
                    yield x.grad

        i = 0

        def check(gm):
            nonlocal i
            if i == 0:
                i += 1
                return

            graph_code = normalize_gm(gm.print_readable(print_output=False))
            self.assertExpectedInline(
                graph_code,
                """\
class CompiledAutograd1(torch.nn.Module):
    def forward(self, inputs, sizes, scalars, hooks, packed_data):
        getitem = inputs[0]
        getitem_1 = inputs[1];  inputs = None
        getitem_2 = sizes[0];  getitem_2 = None
        getitem_3 = sizes[1]
        getitem_4 = sizes[2];  sizes = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cuda', index=0), 6, 0, None), [], False)]);  getitem = None
        getitem_5 = validate_outputs[0];  validate_outputs = None

        sum_backward0 = torch__dynamo_compiled_autograd_ops_SumBackward0([getitem_5], [True], []);  getitem_5 = None
        getitem_6 = sum_backward0[0];  sum_backward0 = None
        validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_6], [((None, None, device(type='cuda', index=0), 6, 0, None), [], False)]);  getitem_6 = None
        getitem_7 = validate_outputs_1[0];  validate_outputs_1 = None

        getitem_8 = hooks[0]
        getitem_9 = packed_data[0];  packed_data = None
        getitem_10 = hooks[1];  hooks = None
        call_hook = torch__dynamo_external_utils_call_hook(getitem_8, getitem_9, hook_type = 'unpack_hook');  getitem_8 = getitem_9 = None
        call_backward = torch__dynamo_external_utils_call_backward(getitem_10, (call_hook,), getitem_7);  getitem_10 = call_hook = getitem_7 = None
        getitem_12 = call_backward[0];  call_backward = None
        validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_12], [((None, None, device(type='cuda', index=0), 6, 0, None), [getitem_3], False)]);  getitem_12 = getitem_3 = None
        getitem_13 = validate_outputs_2[0];  validate_outputs_2 = None

        to_copy_backward0 = torch__dynamo_compiled_autograd_ops_ToCopyBackward0([getitem_13], [True], (None, None, device(type='cpu'), 6, 0, None));  getitem_13 = None
        getitem_14 = to_copy_backward0[0];  to_copy_backward0 = None
        validate_outputs_3 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_14], [((None, None, device(type='cpu'), 6, 0, None), [getitem_4], False)]);  getitem_14 = getitem_4 = None
        getitem_15 = validate_outputs_3[0];  validate_outputs_3 = None

        accumulate_grad__default = torch.ops.inductor.accumulate_grad_.default(getitem_1, getitem_15);  getitem_1 = getitem_15 = accumulate_grad__default = None
        _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
        return []
""",  # noqa: B950
            )

        self.check_output_and_recompiles(
            fn, compiler_fn=make_compiler_fn(gm_hook=check)
        )

    @skipIfWindows(msg="temp dir not compatible")
    def test_disk_offloading(self):
        with tempfile.TemporaryDirectory() as d:

            def fn():
                pack_count = 0

                def pack(x):
                    nonlocal pack_count
                    path = f"{d}/{pack_count}.pt"
                    torch.save(x, path)
                    return path

                def unpack(path):
                    x = torch.load(path)
                    return x

                class MyMatMul(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x):
                        ctx.save_for_backward(x)
                        return torch.matmul(x, x)

                    @staticmethod
                    def backward(ctx, grad_out):
                        (x,) = ctx.saved_tensors
                        return grad_out * x

                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    for i in [10, 100, 10, 20, 30]:
                        x = torch.randn(i, requires_grad=True)
                        MyMatMul.apply(x).sum().backward()
                        yield x.grad

            i = 0

            def check(gm):
                nonlocal i
                if i == 0:
                    i += 1
                    return

                graph_code = normalize_gm(gm.print_readable(print_output=False))
                self.assertExpectedInline(
                    graph_code,
                    """\
class CompiledAutograd1(torch.nn.Module):
    def forward(self, inputs, sizes, scalars, hooks, packed_data):
        getitem = inputs[0]
        getitem_1 = inputs[1];  inputs = None
        getitem_2 = sizes[0];  getitem_2 = None
        getitem_3 = sizes[1];  sizes = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [], False)]);  getitem = None
        getitem_4 = validate_outputs[0];  validate_outputs = None

        sum_backward0 = torch__dynamo_compiled_autograd_ops_SumBackward0([getitem_4], [True], []);  getitem_4 = None
        getitem_5 = sum_backward0[0];  sum_backward0 = None
        validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_5], [((None, None, device(type='cpu'), 6, 0, None), [], False)]);  getitem_5 = None
        getitem_6 = validate_outputs_1[0];  validate_outputs_1 = None

        getitem_7 = hooks[0]
        getitem_8 = packed_data[0];  packed_data = None
        getitem_9 = hooks[1];  hooks = None
        call_hook = torch__dynamo_external_utils_call_hook(getitem_7, getitem_8, hook_type = 'unpack_hook');  getitem_7 = getitem_8 = None
        call_backward = torch__dynamo_external_utils_call_backward(getitem_9, (call_hook,), getitem_6);  getitem_9 = call_hook = getitem_6 = None
        getitem_11 = call_backward[0];  call_backward = None
        validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_11], [((None, None, device(type='cpu'), 6, 0, None), [getitem_3], False)]);  getitem_11 = getitem_3 = None
        getitem_12 = validate_outputs_2[0];  validate_outputs_2 = None

        accumulate_grad__default = torch.ops.inductor.accumulate_grad_.default(getitem_1, getitem_12);  getitem_1 = getitem_12 = accumulate_grad__default = None
        _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
        return []
""",  # noqa: B950
                )

            # 1 graph break on torch.load -> 2 dynamo graphs
            self.check_output_and_recompiles(
                fn,
                count=[1, 2],
                compiler_fn=make_compiler_fn(fullgraph=False, gm_hook=check),
            )

    @skipIfWindows(msg="node name demangling inconsistent on windows")
    def test_backward_hook_relative_ordering_partial(self):
        # test backward hooks for cases that CA matches eager

        def fn():
            order = []

            class MyModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 10, bias=False)

                def forward(self, x):
                    return self.linear(x)

            x = torch.randn(10, 10)
            module = MyModule()

            def make_pre_hook(id):
                return lambda _: order.append(f"pre_hook_{id}")

            def make_post_hook(id):
                return lambda _1, _2: order.append(f"post_hook_{id}")

            count = 0

            def register_hooks_on_all_nodes(nodes):
                nonlocal count
                for node, _ in nodes:
                    if node is None:
                        continue
                    count += 1
                    id = f"{node.name()}_{count}"
                    node.register_prehook(make_pre_hook(id))
                    node.register_hook(make_post_hook(id))
                    register_hooks_on_all_nodes(node.next_functions)

            loss = module(x).sum()
            register_hooks_on_all_nodes(((loss.grad_fn, None),))

            def make_tensor_pre_hook(id):
                return lambda _: order.append(f"tensor_pre_hook_{id}")

            def make_post_acc_grad_hook(id):
                return lambda _: order.append(f"post_acc_grad_hook_{id}")

            module.linear.weight.register_hook(make_tensor_pre_hook("weight"))

            module.linear.weight.register_post_accumulate_grad_hook(
                make_post_acc_grad_hook("weight")
            )

            loss.backward()
            yield tuple(order)

        self.check_output_and_recompiles(fn)

    def test_checkpointing_sac(self):
        # circular import
        from torch.utils.checkpoint import (
            checkpoint,
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def fn():
            class mlp(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer1 = nn.Linear(10, 10)
                    self.layer2 = nn.Linear(10, 10)
                    self.layer3 = nn.Linear(10, 10)
                    self.layer4 = nn.Linear(10, 10)

                def forward(self, x):
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    return x

            recompute_list = [torch.ops.aten.addmm.default]

            def recompute_policy(ctx, op, *args, **kwargs):
                if op in recompute_list:
                    return CheckpointPolicy.MUST_RECOMPUTE
                else:
                    return CheckpointPolicy.PREFER_SAVE

            def context_fn():
                return create_selective_checkpoint_contexts(recompute_policy)

            model = mlp()
            input = torch.randn(1, 10)

            out = checkpoint(model, input, use_reentrant=False, context_fn=context_fn)
            out.sum().backward()
            yield model.layer1.weight.grad
            yield model.layer1.bias.grad
            yield model.layer2.weight.grad
            yield model.layer2.bias.grad
            yield model.layer3.weight.grad
            yield model.layer3.bias.grad
            yield model.layer4.weight.grad
            yield model.layer4.bias.grad

        self.check_output_and_recompiles(
            fn, count=[1, 5], compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_dont_dce_side_effects(self):
        class SideEffectfulBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                torch.randn(10, 10)
                return gO

        x = torch.randn(10, 10, requires_grad=True)

        # https://github.com/pytorch/pytorch/issues/147171
        torch._inductor.config.fallback_random = True

        @torch.compile(backend="aot_eager")
        def fn(x):
            return SideEffectfulBackward.apply(x).sum()

        gm = None

        def extract(ca_gm):
            nonlocal gm
            gm = ca_gm
            return ca_gm

        with compiled_autograd._enable(extract):
            fn(x).backward()

        self.assertTrue("aten.randn" in str(gm))

    def test_aot_bwd_gm_runnable(self):
        # This test ensures that the bw_module saved in
        # CompiledFunction._lazy_backward_info is executable,
        # by ensuring post grad passes have not ran on it.

        post_grad_graphs = []

        def post_grad_pass(graph):
            nonlocal post_grad_graphs
            post_grad_graphs.append(graph)
            return graph

        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        # forces symints to be saved for backward
        # and forces aot compilation of the backward
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 1)

        @torch.compile
        def fn(x, y):
            return torch.matmul(x, y).sum()

        with inductor_config.patch(post_grad_custom_post_pass=post_grad_pass):
            loss = fn(x, y)
            self.assertEqual(len(post_grad_graphs), 2)  # 1 fwd and 1 bwd

        self.assertTrue(loss.grad_fn.name(), "CompiledFunctionBackward")
        self.assertIsNot(
            post_grad_graphs[1],
            loss.grad_fn._forward_cls._lazy_backward_info.bw_module.graph,
        )

        with compiled_autograd._enable(lambda gm: gm):
            loss.backward()

    def test_anomaly_mode_already_nan(self):
        def fn():
            with torch.autograd.detect_anomaly():
                a = torch.randn(5, 5, requires_grad=True)
                a.grad = torch.full((5, 5), float("nan"))
                b = torch.randn(5, 5)
                out = torch.matmul(a, b)
                loss = out.sum()
                with torch._dynamo.compiled_autograd._enable(lambda gm: gm):
                    loss.backward()

        with self.assertRaisesRegex(
            AssertionError, "already having NaN gradient. This is not supported."
        ):
            fn()

    def test_anomaly_mode_backward(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return torch.full(gO.size(), float("nan"))

            with torch.autograd.detect_anomaly():
                a = torch.randn(5, 5, requires_grad=True)
                out = MyFn.apply(a)
                loss = out.sum()
                with torch._dynamo.compiled_autograd._enable(lambda gm: gm):
                    loss.backward()

        with self.assertRaisesRegex(
            RuntimeError, "Compiled Autograd returned NaN gradients for parameters"
        ):
            fn()

    def test_anomaly_mode_grad(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return torch.full(gO.size(), float("nan"))

            with torch.autograd.detect_anomaly():
                a = torch.randn(5, 5, requires_grad=True)
                out = MyFn.apply(a)
                loss = out.sum()
                with torch._dynamo.compiled_autograd._enable(lambda gm: gm):
                    torch.autograd.grad(loss, inputs=a)

        with self.assertRaisesRegex(
            RuntimeError, "Compiled Autograd returned NaN gradients for output nodes"
        ):
            fn()

    def test_higher_order_gradients(self):
        def f(x):
            return x**3

        def fn(fwd_compiler, ca_compiler):
            torch.manual_seed(123)
            x = torch.tensor(2.0, requires_grad=True)
            first, second, third, fourth = None, None, None, None
            try:
                with compiled_autograd._enable(ca_compiler):
                    first = torch.autograd.grad(
                        fwd_compiler(f)(x), x, create_graph=True
                    )[0]
                    second = torch.autograd.grad(first, x, create_graph=True)[0]
                    third = torch.autograd.grad(second, x, create_graph=True)[0]
                    fourth = torch.autograd.grad(third, x, create_graph=True)[0]
            except RuntimeError as e:
                assert "does not currently support higher order gradients" in str(e)
                return (first, second, third, fourth)

            return (first, second, third, fourth)

        def eager():
            return torch.compile(backend="eager")

        def aot_eager():
            return torch.compile(backend="aot_eager")

        # Without AOTAutograd, no problem
        first, second, third, fourth = fn(eager(), eager())
        self.assertEqual(counters["compiled_autograd"]["captures"], 4)
        self.assertEqual(first, 12)  # 3x^2
        self.assertEqual(second, 12)  # 6x
        self.assertEqual(third, 6)  # 6
        self.assertEqual(fourth, 0)
        # and should cache hit
        counters.clear()
        _ = fn(eager(), eager())
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        torch._dynamo.reset()

        # With AOTAutograd, can't create_graph
        first, second, third, fourth = fn(aot_eager(), aot_eager())
        self.assertIsNone(second)

        first, second, third, fourth = fn(aot_eager(), eager())
        self.assertIsNone(second)

        first, second, third, fourth = fn(eager(), aot_eager())
        self.assertIsNone(third)

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "FakePG relies on distributed build",
    )
    def test_ddp_cpp_reducer_error(self):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        try:
            model = torch.nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
            model = DDP(model)
            inputs = torch.randn(10, 10)
            loss = model(inputs).sum()
            with (
                compiled_autograd._enable(compiler_fn),
                self.assertRaisesRegex(
                    RuntimeError,
                    (
                        r"Compiled autograd is not compatible with C\+\+ DDP Reducer, "
                        r'please use torch._dynamo.config.optimize_ddp="python_reducer"'
                    ),
                ),
            ):
                loss.backward()

        finally:
            dist.destroy_process_group()

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "FakePG relies on distributed build",
    )
    @config.patch(optimize_ddp="python_reducer")
    def test_ddp_python_reducer(self):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        try:
            model = torch.nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
            model = DDP(model)
            inputs = torch.randn(10, 10)
            loss = model(inputs).sum()
            with compiled_autograd._enable(compiler_fn):
                # no error expected
                loss.backward()
            self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        finally:
            dist.destroy_process_group()

    # Case 1.1: Stealable dense new_grad
    # if (!GradMode::is_enabled() && !new_grad.is_sparse() &&
    #     !new_grad.is_sparse_csr() &&
    #     !(variable.is_sparse_csr() && new_grad.layout() == at::kStrided) &&
    #     at::caching::adjusted_use_count(new_grad) <= num_expected_refs &&
    #     (new_grad.is_mkldnn() || utils::obeys_layout_contract(new_grad, variable))) {
    @unittest.expectedFailure
    def test_accumulate_grad_polyfill_case_1_1(self):
        def fn():
            class StealableDenseOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    return torch.ones_like(grad_output, requires_grad=False) * 5

            pre_hook_storage_id = None

            def check(grad):
                nonlocal pre_hook_storage_id
                assert pre_hook_storage_id is None
                pre_hook_storage_id = id(grad.untyped_storage())

            var = torch.randn(2, 2, requires_grad=True)
            var.register_hook(check)
            output = StealableDenseOp.apply(var)
            output.backward(torch.ones_like(output))

            assert var.grad is not None, "Grad should be defined"
            assert torch.equal(var.grad, torch.ones_like(var) * 5), (
                "Grad content should be as returned by backward"
            )
            assert var.grad.requires_grad is False, (
                "Detached grad should not require grad"
            )
            assert id(var.grad.untyped_storage()) == pre_hook_storage_id, (
                "Should be stolen"
            )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 2],
        )

    # Case 1.2: Stealable sparse new_grad
    # } else if (!GradMode::is_enabled() && new_grad.is_sparse() &&
    #            new_grad._indices().is_contiguous() &&
    #            new_grad._values().is_contiguous() &&
    #            new_grad._indices().use_count() <= 1 &&
    #            new_grad._values().use_count() <= 1 &&
    #            new_grad.use_count() <= num_expected_refs) {
    @unittest.expectedFailure
    def test_accumulate_grad_polyfill_case_1_2(self):
        def fn():
            class StealableSparseOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    size = grad_output.size()
                    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
                    values = torch.tensor([5.0, 5.0])
                    return torch.sparse_coo_tensor(
                        indices, values, size, requires_grad=False
                    )

            pre_hook_storages_id = None

            def check(grad):
                nonlocal pre_hook_storages_id
                assert pre_hook_storages_id is None
                pre_hook_storages_id = [
                    id(grad._indices().untyped_storage()),
                    id(grad._values().untyped_storage()),
                ]

            var = torch.randn(2, 2, requires_grad=True)
            var.register_hook(check)
            output = StealableSparseOp.apply(var)
            output.backward(torch.ones_like(output))

            assert var.grad is not None, "Grad should be defined"
            assert var.grad.is_sparse, "Grad should be sparse"
            expected_dense_grad = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
            assert torch.equal(var.grad.to_dense(), expected_dense_grad), (
                "Content should be equal after shallow copy"
            )
            assert var.grad.requires_grad is False, (
                "Detached grad should not require grad"
            )
            assert (
                id(var.grad._indices().untyped_storage()) == pre_hook_storages_id[0]
            ), "Should be stolen"
            assert (
                id(var.grad._values().untyped_storage()) == pre_hook_storages_id[1]
            ), "Should be stolen"
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 2],
        )

    # Case 1.3: Cloning sparse/nested new_grad
    # else {
    #   if (new_grad.is_sparse() || new_grad.is_sparse_csr() ||
    #       new_grad.is_nested()) {
    def test_accumulate_grad_polyfill_case_1_3(self):
        def fn():
            class CloneSparseGradOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    size = grad_output.size()
                    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
                    values = torch.tensor(
                        [5.0, 5.0], requires_grad=True
                    )  # Requires grad
                    return torch.sparse_coo_tensor(
                        indices, values, size, requires_grad=True
                    )

            pre_hook_storages_id = None

            def check(grad):
                nonlocal pre_hook_storages_id
                assert pre_hook_storages_id is None
                pre_hook_storages_id = [
                    id(grad._indices().untyped_storage()),
                    id(grad._values().untyped_storage()),
                ]

            var = torch.randn(2, 2, requires_grad=True)
            var.register_hook(check)
            output = CloneSparseGradOp.apply(var)
            output.backward(
                torch.ones_like(output), create_graph=True
            )  # grad mode == create_graph

            assert var.grad is not None, "Grad should be defined"
            assert var.grad.is_sparse, "Grad should be sparse"
            expected_dense_grad = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
            assert torch.equal(var.grad.to_dense(), expected_dense_grad), (
                "Content should be equal after clone"
            )
            assert var.grad.requires_grad, (
                "Grad should require grad for double backward"
            )
            assert (
                id(var.grad._indices().untyped_storage()) != pre_hook_storages_id[0]
            ), "Should be copied"
            assert (
                id(var.grad._values().untyped_storage()) != pre_hook_storages_id[1]
            ), "Should be copied"
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 2],
        )

    # Case 1.5.1: Dense variable gradient layout contract
    # else { // Covers various deep copy scenarios not covered by specific stealable paths
    #   ...
    #   if (new_grad.is_mkldnn()) {
    #     ...
    #   } else {
    #       // Deep copies new_grad according to the "Gradient Layout Contract."
    #       update_grad(utils::clone_obey_contract(new_grad, variable));
    #   }
    # }
    def test_accumulate_grad_polyfill_case_1_5_1(self):
        def fn():
            class NotStealableRefsOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    return torch.ones_like(grad_output, requires_grad=False) * 10.0

            var = torch.randn(2, 2, requires_grad=True)
            grad_ref_holder = [None]

            def check(grad):
                # forces a clone due to refcount
                grad_ref_holder[0] = grad
                return grad

            var.register_hook(check)
            output = NotStealableRefsOp.apply(var)
            output.backward(torch.ones_like(output))

            assert var.grad is not None, "Grad should be defined"
            assert torch.equal(var.grad, torch.ones_like(var) * 10.0), (
                "Grad content should be as returned by backward"
            )
            assert (
                grad_ref_holder[0].untyped_storage() is not var.grad.untyped_storage()
            ), "Should be copied"
            yield var.grad

        self.check_output_and_recompiles(fn)

    # Case 1.5.2: Non-dense variable gradient layout contract
    # else { // Covers various deep copy scenarios not covered by specific stealable paths
    #   ...
    #   if (new_grad.is_mkldnn()) {
    #     ...
    #   } else {
    #       // Deep copies new_grad according to the "Gradient Layout Contract."
    #       update_grad(utils::clone_obey_contract(new_grad, variable));
    #   }
    # }
    def test_accumulate_grad_polyfill_case_1_5_2(self):
        def fn():
            class SimpleDenseGradOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    return torch.ones_like(grad_output, requires_grad=False) * 7.0

            # Create a non-contiguous variable
            base_tensor = torch.randn(4, 4)
            var = base_tensor[::2, ::2]
            assert not var.is_contiguous(), (
                "Variable should be non-contiguous for this test"
            )
            var.requires_grad_(True)

            grad_ref_holder = [None]

            def check(grad):
                # forces a clone due to refcount
                grad_ref_holder[0] = grad
                return grad

            var.register_hook(check)
            output = SimpleDenseGradOp.apply(var)
            output.backward(torch.ones_like(output))

            assert var.grad is not None, "Grad should be defined"
            # The `clone_obey_contract` branch 2 (`new_grad.clone(at::MemoryFormat::Contiguous)`)
            # will make the resulting grad contiguous.
            assert var.grad.is_contiguous(), (
                "Resulting grad should be contiguous due to branch 2 of clone_obey_contract"
            )
            assert torch.equal(var.grad, torch.ones_like(var) * 7.0), (
                "Grad content should be as returned by backward"
            )
            assert (
                grad_ref_holder[0].untyped_storage() is not var.grad.untyped_storage()
            ), "Should be copied"
            yield var.grad

        self.check_output_and_recompiles(
            fn,
        )

    # Case 2.1: Sparse variable_grad + Dense new_grad
    # } else if (!GradMode::is_enabled()) {
    #   if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
    #       auto result = new_grad + variable_grad;
    def test_accumulate_grad_polyfill_case_2_1(self):
        def fn():
            class SparseVarGradDenseNewGradOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    return torch.ones_like(grad_output) * 3.0

            var = torch.randn(2, 2, requires_grad=True)
            indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
            values = torch.tensor([1.0, 1.0])
            var.grad = torch.sparse_coo_tensor(
                indices, values, var.size(), requires_grad=False
            )
            initial_grad_ref = var.grad
            output = SparseVarGradDenseNewGradOp.apply(var)

            expected_sum = (torch.ones_like(var) * 3.0) + initial_grad_ref.to_dense()
            output.backward(torch.ones_like(output))

            assert var.grad is not None, "Grad should be defined"
            assert not var.grad.is_sparse, "Resulting grad should be dense"
            assert torch.equal(var.grad, expected_sum), "Grad content should be the sum"
            assert var.grad is not initial_grad_ref, (
                "Grad object should be replaced (out-of-place)"
            )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=lambda gm: gm,  # https://github.com/pytorch/pytorch/issues/154161
            count=[1, 0],
        )

    # Case 2.3.1: Dense/Dense in-place addition
    # } else if (!GradMode::is_enabled()) {
    #   ...
    # } else {
    #   variable_grad += new_grad;
    def test_accumulate_grad_polyfill_case_2_3_1(self):
        def fn():
            class DenseVarGradDenseNewGradOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    return torch.ones_like(grad_output) * 3.0

            var = torch.randn(2, 2, requires_grad=True)
            var.grad = torch.ones_like(var) * 1.0
            initial_grad_ref = var.grad
            output = DenseVarGradDenseNewGradOp.apply(var)
            expected_sum = initial_grad_ref + (torch.ones_like(var) * 3.0)
            output.backward(torch.ones_like(output))

            assert var.grad is not None, "Grad should be defined"
            assert not var.grad.is_sparse, "Resulting grad should be dense"
            assert torch.equal(var.grad, expected_sum), "Grad content should be the sum"
            assert var.grad is initial_grad_ref, (
                "Grad object should be modified in-place (same object)"
            )
            yield var.grad

        self.check_output_and_recompiles(fn)

    # Case 2.3.2: Sparse/Sparse in-place addition
    # } else if (!GradMode::is_enabled()) {
    #   ...
    # } else {
    #   variable_grad += new_grad;
    def test_accumulate_grad_polyfill_case_2_3_2(self):
        def fn():
            class SparseVarGradSparseNewGradOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    size = grad_output.size()
                    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
                    values = torch.tensor([3.0, 3.0])
                    return torch.sparse_coo_tensor(
                        indices, values, size, requires_grad=False
                    )

            var = torch.randn(2, 2, requires_grad=True)
            indices_v = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
            values_v = torch.tensor([1.0, 2.0])
            var.grad = torch.sparse_coo_tensor(
                indices_v, values_v, var.size(), requires_grad=False
            )
            initial_grad_ref = var.grad

            output = SparseVarGradSparseNewGradOp.apply(var)

            new_grad_for_sum = torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
                torch.tensor([3.0, 3.0]),
                var.size(),
            )
            expected_sum_dense = (
                initial_grad_ref.to_dense() + new_grad_for_sum.to_dense()
            )

            output.backward(torch.ones_like(output))

            assert var.grad is not None, "Grad should be defined"
            assert var.grad.is_sparse, "Resulting grad should remain sparse"
            assert torch.equal(var.grad.to_dense(), expected_sum_dense), (
                "Grad content should be the sum of sparse grads"
            )
            assert var.grad is initial_grad_ref, (
                "Grad object should be modified in-place (same object)"
            )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=lambda gm: gm,  # https://github.com/pytorch/pytorch/issues/154161
            count=[1, 0],
        )

    # Case 2.3.3: Dense/Sparse in-place addition
    # } else if (!GradMode::is_enabled()) {
    #   ...
    # } else {
    #   variable_grad += new_grad;
    def test_accumulate_grad_polyfill_case_2_3_3(self):
        def fn():
            class DenseVarGradSparseNewGradOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    size = grad_output.size()
                    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
                    values = torch.tensor([3.0, 3.0])  # New sparse values
                    return torch.sparse_coo_tensor(
                        indices, values, size, requires_grad=False
                    )

            var = torch.randn(2, 2, requires_grad=True)
            var.grad = torch.ones_like(var) * 1.0  # Initial value
            initial_grad_ref = var.grad
            output = DenseVarGradSparseNewGradOp.apply(var)

            new_grad_for_sum = torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
                torch.tensor([3.0, 3.0]),
                var.size(),
            ).to_dense()
            expected_sum = initial_grad_ref + new_grad_for_sum

            output.backward(torch.ones_like(output))

            assert var.grad is not None, "Grad should be defined"
            assert not var.grad.is_sparse, "Resulting grad should be dense"
            assert torch.equal(var.grad, expected_sum), "Grad content should be the sum"
            assert var.grad is initial_grad_ref, (
                "Grad object should be modified in-place (same object)"
            )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 2],
        )

    # Case 3.1: Sparse variable_grad + Dense new_grad (reorder into Dense + Sparse)
    # } else { // if GradMode::is_enabled()
    #   at::Tensor result;
    #   if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
    #     result = new_grad + variable_grad;
    #   }
    # }
    def test_accumulate_grad_polyfill_case_3_1(self):
        def fn():
            class SparseVarGradDenseNewGradDoubleBackwardOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    return torch.ones_like(grad_output, requires_grad=True) * 3.0

            var = torch.randn(2, 2, requires_grad=True)
            indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
            values = torch.tensor([1.0, 1.0], requires_grad=True)
            var.grad = torch.sparse_coo_tensor(
                indices, values, var.size(), requires_grad=True
            )
            initial_grad_ref = var.grad

            output = SparseVarGradDenseNewGradDoubleBackwardOp.apply(var)

            expected_sum = (
                torch.ones_like(var, requires_grad=True) * 3.0
            ) + initial_grad_ref.to_dense()

            output.backward(torch.ones_like(output), create_graph=True)

            assert var.grad is not None, "Grad should be defined"
            assert not var.grad.is_sparse, "Resulting grad should be dense"
            assert torch.equal(var.grad, expected_sum), "Grad content should be the sum"
            assert var.grad is not initial_grad_ref, (
                "Grad object should be replaced (out-of-place)"
            )
            assert var.grad.requires_grad, (
                "Resulting grad should track history for double backward"
            )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=lambda gm: gm,  # https://github.com/pytorch/pytorch/issues/154161
            count=[1, 0],
        )

    # Case 3.2: variable_grad.defined() & GradMode::is_enabled() - Double backward (dense variable_grad + dense new_grad)
    # } else { // if GradMode::is_enabled()
    #   at::Tensor result;
    #   ...
    #   } else {
    #     result = variable_grad + new_grad;
    #   }
    # }
    def test_accumulate_grad_polyfill_case_3_2(self):
        def fn():
            class DenseVarGradDenseNewGradDoubleBackwardOp(BaseCustomOp):
                @staticmethod
                def backward(ctx, grad_output):
                    return torch.ones_like(grad_output, requires_grad=True) * 3.0

            var = torch.randn(2, 2, requires_grad=True)
            var.grad = torch.ones_like(var) * 1.0
            initial_grad_ref = var.grad

            output = DenseVarGradDenseNewGradDoubleBackwardOp.apply(var)

            expected_sum = initial_grad_ref + (
                torch.ones_like(var, requires_grad=True) * 3.0
            )

            output.backward(torch.ones_like(output), create_graph=True)

            assert var.grad is not None, "Grad should be defined"
            assert not var.grad.is_sparse, "Resulting grad should be dense"
            assert torch.equal(var.grad, expected_sum), "Grad content should be the sum"
            assert var.grad is not initial_grad_ref, (
                "Grad object should be replaced (out-of-place)"
            )
            assert var.grad.requires_grad, (
                "Resulting grad should track history for double backward"
            )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 3],
        )

    def test_torch_function_mode(self):
        called_funcs = []

        class LoggingTorchFunctionMode(BaseTorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                called_funcs.append(str(func.__name__))
                return super().__torch_function__(func, types, args, kwargs)

        class MyLoss(torch.autograd.Function):
            @staticmethod
            def forward(ctx, out):
                ctx.save_for_backward(out)
                return out.sum()

            @staticmethod
            def backward(ctx, grad_output):
                (saved,) = ctx.saved_tensors
                return torch.ones_like(saved) * grad_output

        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2)
        z = torch.randn(2, 2)

        def fwd(x, y, z):
            out = x * y * z
            loss = MyLoss.apply(out)
            return loss

        with LoggingTorchFunctionMode():
            called_funcs.append("Forward")
            loss = fwd(x, y, z)
            called_funcs.append("Backward")
            with torch._dynamo.compiled_autograd._enable(torch.compile):
                loss.backward()

        self.assertExpectedInline(
            "\n".join(called_funcs),
            """\
Forward
mul
mul
sum
Backward
_set_multithreading_enabled
backward
_set_multithreading_enabled""",
        )  # noqa: B950

    def test_torch_dispatch_mode(self):
        called_funcs = []

        class LoggingTorchDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                called_funcs.append(str(func.__name__))
                return func(*args, **kwargs)

        class MyLoss(torch.autograd.Function):
            @staticmethod
            def forward(ctx, out):
                ctx.save_for_backward(out)
                return out.sum()

            @staticmethod
            def backward(ctx, grad_output):
                (saved,) = ctx.saved_tensors
                return torch.ones_like(saved) * grad_output

        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2)
        z = torch.randn(2, 2)

        def fwd(x, y, z):
            out = x * y * z
            loss = MyLoss.apply(out)
            return loss

        with LoggingTorchDispatchMode():
            called_funcs.append("Forward")
            loss = fwd(x, y, z)
            called_funcs.append("Backward")
            with torch._dynamo.compiled_autograd._enable(lambda gm: gm):
                loss.backward()

        self.assertExpectedInline(
            "\n".join(called_funcs),
            """\
Forward
mul.Tensor
mul.Tensor
sum.default
Backward
ones_like.default
empty.memory_format
empty.memory_format
empty.memory_format
empty.memory_format
empty.memory_format
empty.memory_format
ones_like.default
mul.Tensor
mul.Tensor
mul.Tensor
new_empty_strided.default
copy_.default""",
        )  # noqa: B950


def load_test_module(name):
    testdir = Path(__file__).absolute().parent.parent
    with mock.patch("sys.path", [*sys.path, str(testdir)]):
        return SourceFileLoader(
            name, str(testdir / f"{name.replace('.', '/')}.py")
        ).load_module()


def make_wrapped(fn, ctxs):
    @functools.wraps(fn)
    def wrapped(self):
        torch._dynamo.reset()
        stack = contextlib.ExitStack()
        for ctx in ctxs:
            stack.enter_context(ctx)
        out = fn(self)
        stack.close()
        return out

    return wrapped


def lookup_backend(test_name):
    if test_name in xfail_by_backend["inductor"]:
        return "aot_eager"
    elif test_name in xfail_by_backend["aot_eager"]:
        return "eager"
    elif test_name in xfail_by_backend["eager"]:
        return "ca_eager"
    else:
        assert test_name not in xfail_by_backend["ca_eager"]
        return "inductor"


def wrap_test_class(orig_cls):
    dct = orig_cls.__dict__.copy()
    for name in list(dct.keys()):
        fn = dct[name]
        if not callable(fn) or name in skipped_tests:
            continue
        elif (
            xfail_re.match(name)
            or name in xfail_by_backend["ca_eager"]
            or name in xfail_divergence_from_eager
        ):
            dct[name] = unittest.expectedFailure
        elif name.startswith("test_"):
            backend = lookup_backend(name)
            if not HAS_CUDA_AND_TRITON and backend == "inductor":
                continue
            ctxs = [
                compiled_autograd._enable(
                    make_compiler_fn(
                        backend=backend,
                        fullgraph=name not in known_graph_breaks_tests,
                    )
                ),
                test_contexts.get(name, contextlib.nullcontext()),
            ]
            dct[name] = make_wrapped(fn, ctxs)

    cls = type(
        orig_cls.__name__ + "WithCompiledAutograd",
        orig_cls.__bases__,
        dct,
    )
    cls.__file__ = __file__
    return cls


known_graph_breaks_tests = {
    "test_hook_none",  # uses assert in hook
    "test_post_accumulate_grad_hook_e2e",  # optim.Adam manually graph breaks
    "test_tensor_hooks_inplace",  # uses assert in hook
    "test_tensor_hooks_inplace_over_view",  # uses assert in hook
    "test_grad_fn_prehooks",  # uses assert in hook
    "test_grad_fn_prehooks_multiple_outputs",  # uses assert in hook
    "test_grad_fn_prehooks_remove_hooks",  # uses handle.remove() in hook
    "test_tensor_hooks_inplace_multiple_outputs",  # uses assert in hook
    "test_hooks",  # uses assert in hook
    "test_accumulate_grad_posthooks_can_observe_tensor_prehook",  # allclose
    "test_saved_tensors_hook_version_counter_not_shared",  # assertEqual
    "test_post_accumulate_grad_hook_returns_not_None",  # throws
    "test_custom_function_cycle",  # assertEqual
    "test_mark_non_differentiable_mixed",  # assertTrue
    "test_materialize_grads",  # assertEqual
    "test_return_leaf",  # assertEqual
    "test_save_none_for_backward",  # assertIsNone
    "test_saved_variables_deprecated",  # warnings.warn
    "test_autograd_node_isinstance",  # assertIsInstance
    "test_set_materialize_non_diff_grads",  # assertIsNone
    "test_backward_dict_grad_for_nontensor",  # torch/_custom_op/autograd.py in skip files
    "test_backward_dict_invalid_keys",  # torch/_custom_op/autograd.py in skip files
    "test_backward_dict_requires_keys_for_input_optional_tensors",  # torch/_custom_op/autograd.py in skip files
    "test_backward_dict_requires_keys_for_input_tensors",  # torch/_custom_op/autograd.py in skip files
    "test_backward_grads_are_tensor_or_none",  # torch/_custom_op/autograd.py in skip files
    "test_backward_impl_on_existing_op",  # torch/_custom_op/autograd.py in skip files
    "test_backward_returns_dict",  # torch/_custom_op/autograd.py in skip files
    "test_backward_tensorlist_input_requires_list_grads",  # torch/_custom_op/autograd.py in skip files
    "test_backward_tensorlist_input_requires_list_grads_none_or_Tensor",  # torch/_custom_op/autograd.py in skip files
    "test_backward_tensorlist_input_requires_list_grads_with_same_numel",  # torch/_custom_op/autograd.py in skip files
    "test_save_for_backward_inputs_are_namedtuple",  # torch/_custom_op/autograd.py in skip files
    "test_reentrant_with_leaf_variable_hook",  # reentrant .backward
    "test_reentrant_with_non_leaf_variable_hook",  # reentrant .backward
    "test_reentrant_child_error",  # reentrant .backward
    "test_deep_reentrant",  # reentrant .backward
    "test_reentrant_priority",  # reentrant .backward
    "test_simple_reentrant",  # reentrant .backward
    "test_checkpoint_detects_non_determinism",  # unpack hook in skip files
    "test_checkpoint_valid_reset_on_error",  # unpack hook in skip files
    "test_checkpointing_non_reentrant_autocast_cpu",  # unpack hook in skip files
    "test_checkpointing_non_reentrant_autocast_gpu",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_arbitrary_input_output",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_correct_grad",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_custom_function_works",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_dataparallel",  # _get_device_index in skip files
    "test_checkpointing_without_reentrant_detached_tensor_use_reentrant_True",  # reentrant .backward
    "test_checkpointing_without_reentrant_parameter_used_in_an_out",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_with_context_fn",  # unpack hook in skip files
    "test_save_on_cpu_and_checkpoint",  # unpack hook in skip files
    "test_saved_tensor_hooks_custom_error_propagation",  # CustomError
    "test_access_saved_tensor_twice_without_recomputation_works",  # unpack hook in skip files
    "test_saved_tensor_hooks_extra_enter_during_bw_no_leak",  # ctx in skip files
    "test_saved_tensor_hooks_extra_exit_during_bw_no_crash",  # ctx in skip files
    "test_checkpointing",  # reentrant .backward
    "test_checkpointing_without_reentrant_input_requires_grad_False",  # reentrant .backward
    "test_checkpointing_without_reentrant_input_requires_grad_True",  # reentrant .backward
    "test_checkpointing_without_reentrant_memory_savings",  # reentrant .backward
    "test_dtensor_basic",  # torch._dynamo.exc.Unsupported: Failed to convert args/kwargs to proxy
    "test_dtensor_contiguous_dtensor_noncontiguous_local_as_tangent",  # subclass constructor
    "test_retain_grad",  # retains_grad_hooks
    "test_retain_grad_cycle",  # retains_grad_hooks
    "test_retain_grad_inplace",  # retains_grad_hooks
    "test_retain_grad_inplace_over_view",  # retains_grad_hooks
    "test_retains_grad_can_always_observe_tensor_prehook",  # retains_grad_hooks
    "test_retains_grad_inplace_multiple_outputs",  # retains_grad_hooks
    "test_hook_edge_case_when_called_with_grad",  # retains_grad_hooks
    "test_multi_grad_all_hooks",  # retains_grad_hooks
    "test_prehook_ordering",  # retains_grad_hooks
    "test_will_engine_execute_node",  # retains_grad_hooks
    "test_backward_to_node",  # retains_grad_hooks
    "test_backward_with_nonleaf_inputs",  # retains_grad_hook on non-leaf input
    "test_create_graph_and_full_backward_hook_cycle",  # _pack_with_none
    "test_full_backward_hook_double_backward",  # _pack_with_none
    "test_grad_mode_restored_reentrant",  # assertTrue
    "test_multi_grad_any_hooks",  # register_multi_grad_hook
    "test_saved_variable_packing_unpacking_did_not_save_original_with_hooks",  # register_hooks
    "test_graph_save_on_cpu",  # dynamo disabled
    "test_nested_checkpoint_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_kwargs_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_kwargs_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_non_tensor_inputs_and_outputs_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_non_tensor_inputs_and_outputs_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_reentrant_backwards_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_reentrant_backwards_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_same_graph_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_same_graph_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_set_early_stop",  # dynamo disable
    "test_nested_checkpoint_two_children_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_two_children_early_stop_True",  # dynamo disable
    "test_custom_autograd_ac_early_stop",  # marked as skipped
    "test_dropout",  # dynamo disable
    "test_dropout_inductor",  # dynamo disable
    "test_function_with_kwargs",  # dynamo disable
    "test_module",  # dynamo disable
}

test_contexts = {
    "test_setitem_mask": config.patch(capture_dynamic_output_shape_ops=True),
    "test_index_backward_does_not_save_tensor": config.patch(
        capture_dynamic_output_shape_ops=True
    ),
}

# These groups of tests aren't supported yet
xfail_re = re.compile(r"^test_(sparse|profiler|gradcheck|named_tensor)")

# Tests fail at different stages, we categorize them wrt to their backends
# We run only the last passing backend in this order:
# ca_eager -> eager -> aot_eager -> inductor
xfail_by_backend = {
    "ca_eager": {  # xfail
        "test_callback_propagates_errors_from_device_thread",  # fullgraph for queue_callback, but graph break for RuntimeError
        "test_reentrant_with_callbacks_both_depths",  # queue_callback
        "test_reentrant_with_callbacks_depth_0",  # queue_callback
        "test_reentrant_with_callbacks_depth_1",  # queue_callback
        "test_checkpoint_graph_execution_group",  # Attempted to call function marked as skipped
        "test_current_graph_task_execution_order",  # nodes are already freed by the time dynamo traces the lifted hook
        "test_autograd_inplace_views_cross_dtype",  # view_fn not supported by compiled autograd
        "test_post_accumulate_grad_hook_ordering",  # accuracy error
        "test_current_graph_task_id",  # autograd state already cleared once dynamo is called
        "test_custom_function_forward_mode_forward_is_no_op",  # forward AD
        "test_custom_function_forward_mode_inplace_checks",  # forward AD
        "test_custom_function_forward_mode_view_checks",  # forward AD
        "test_custom_function_forward_mode_wrong_formula",  # forward AD
        "test_node_post_hook_registered_during_unpack_hook",  # 'NoneType' object has no attribute 'register_hook'
        "test_custom_function_error",  # forward AD
        "test_custom_function_save_for_forward",  # forward AD
        "test_dont_materialize_grads",  # undefined grad
        "test_no_grad_copy",  # setting static member in lifted backward
        "test_no_grad_copy_sparse",  # setting static member in lifted backward
        "test_node_ordering_when_none_returned",  # torch._dynamo.exc.Unsupported: TypeError <built-in method clone
        "test_save_output_nr",  # output_nr grad passed as None
        # IndexError: list index out of range (NB: x.grad = y where both x and y are input tensors)
        "test_grad_nonleaf_register_hook",
        "test_backward_twice_without_saved_values",  # https://github.com/pytorch/pytorch/issues/129938
        # Category: Higher Order Gradients
        "test_default_saved_tensors_hooks_double_backward",  # wrong when pack hook returns non-leaf
        "test_saved_variable_packing_unpacking_saved_original_with_hooks",  # wrong when pack hook returns non-leaf
        "test_nested_anomaly_detect_nan",  # nested anomaly
        "test_select_sum",  # batched gradients
        "test_custom_autograd_no_early_free",  # batched gradients
        "test_grad_batched_grad",  # batched gradients
        # Uncategorized
        "test_lobpcg",  # NaNs
        "test_autograd_simple_views_python",  # gradient is None
        "test_function_returns_undefined_tensor",  # gradient is None
        "test_input_buffer_accum",  # add(sparse, dense)
        "test_return_duplicate",  # batched gradients
        "test_return_duplicate_inplace",  # batched gradients
        "test_naughty_autograd_function_stashing_ctx",  # error not raised
        "test_unrelated_inputs",  # batched gradients
        "test_nested_checkpoint_early_stop_False",  # unpack hook grad_fn semantics
        "test_nested_checkpoint_early_stop_True",  # unpack hook grad_fn semantics
        "test_nested_checkpoint_two_children_early_stop_False",  # unpack hook grad_fn semantics
        "test_nested_checkpoint_two_children_early_stop_True",  # unpack hook grad_fn semantics
        "test_dropout",  # functionalize_rng_ops not yet supported
        "test_dropout_inductor",  # functionalize_rng_ops not yet supported
        "test_function_with_kwargs",  # functionalize_rng_ops not yet supported
        "test_module",  # functionalize_rng_ops not yet supported
        "test_grad_dtype",  # AttributeError: args / Float did not match Double
    },
    "eager": {  # will be run without torch.compiling the CA graph
        "test_setup_context_when_forward_has_default_args",  # autograd.Function with class methods
        "test_accumulate_grad_tensor_reference",  # Out of bounds: frame_state_entry.stride[i] is None
        "test_custom_function_exception",  # torch.no_grad(), torch._dynamo.exc.Unsupported: missing: WITH_EXCEPT_START
        "test_to_sparse_backward",  # Out of bounds: frame_state_entry.stride[i] is None
        "test_custom_function_non_tensor_inputs_outputs",  # gradient batching rule not implemented for aten::sym_size.int
        "test_setitem",  # CopySlices accuracy error
        "test_checkpointing_without_reentrant_saved_object_identity",  # same as https://github.com/pytorch/pytorch/issues/136193
        "test_dtensor_different_gradient_placement",  # Dynamo failed to run FX node with fake tensors
        "test_dtensor_noncontiguous_output",  # Dynamo failed to run FX node with fake tensors
        "test_dtensor_partial_placement_graph_output",  # Dynamo failed to run FX node with fake tensors
        "test_unwrap_async_collective_tensor_tangent",  # AttributeError: 'PlainTensorMeta' object has no attribute 'attrs'
        "test_graph_save_on_cpu",  # torch.save should no-op and be recorded in the graph
        "test_saving_variable_to_disk",  # torch.save should no-op and be recorded in the graph
        "test_nested_checkpoint_early_stop_False",  # AOT backward higher order gradients
        # Slow tests, these tests are close to CI timeout if we try to torch.compile them
        "test_checkpointing",
        "test_checkpointing_without_reentrant_memory_savings",
        "test_checkpointing_without_reentrant_input_requires_grad_True",
        "test_checkpointing_without_reentrant_input_requires_grad_False",
    },
    "aot_eager": {  # will be run with torch.compile(backend="eager")
        # Category: FakeTensor
        "test_wrapped_number_saved_tensors_hooks",  # Proxy tensor should carryover is_wrapped_number_ of its original
        "test_scalar_grad_mixed_device",  # Fake Tensors aren't propagating device properly for 0-dim grads
        "test_grad",  # AOT backward higher order gradients
        "test_grad_materialize_grads",  # AOT backward higher order gradients
    },
    "inductor": {},  # will be run with torch.compile(backend="aot_eager")
    # tests not present in this dict will be run with torch.compile(backend="inductor")
}

# These tests fail due to difference in semantics that we won't fix
xfail_divergence_from_eager = {
    "test_invalid_gradients",  # can't give autograd error due to inaccurate output metadata of lifted backward
    "test_autograd_node_isinstance",  # backward ctx is a fake cls and not directly a Node instance
    "test_backward_hook_relative_ordering",  # compiled autograd collects breadth first, and module backward hook not supported
    "test_checkpointing_without_reentrant_custom_function_works",  # ctx.saved_tensors are cached by CA
    "test_anomaly_mode_no_check_nan",  # different error messages
    "test_anomaly_grad_warnings",  # different error messages
    "test_anomaly_detect_nan",  # fake tensor errors on NaN
    "test_once_differentiable",  # different node name: CompiledFunctionBackward
    "test_function",  # different node name: CompiledFunctionBackward
    "test_inplace_on_view_backward",  # different node name: CompiledFunctionBackward
    "test_nested_anomaly_printstack_cleanup",  # anomaly NaN error message different
    "test_not_implemented_grad",  # Dynamo changes the types of exceptions
    "test_grad_call_compiled_backward_fn",  # different functorch error
    "test_vjp_call_compiled_backward_fn",  # different functorch error
    "test_vmap_call_compiled_backward_fn",  # different functorch error
    "test_accumulate_grad",  # always out of place add for compiled autograd
    "test_current_node",  # slightly different dispatched ops
}

skipped_tests = set()

if not HAS_CUDA_AND_TRITON:
    # Found Tesla M60 which is too old to be supported by the triton GPU compiler
    skipped_tests.add("test_type_conversions")

if IS_S390X:
    skipped_tests.add("test_deep_reentrant")

test_autograd = load_test_module("test_autograd")
test_custom_ops = load_test_module("test_custom_ops")
test_higher_order_ops = load_test_module("dynamo/test_higher_order_ops")

TestAutogradWithCompiledAutograd = wrap_test_class(test_autograd.TestAutograd)
TestNestedCheckpointWithCompiledAutograd = wrap_test_class(
    test_autograd.TestNestedCheckpoint
)
TestCustomOpWithCompiledAutograd = wrap_test_class(test_custom_ops.TestCustomOp)
HigherOrderOpTestsWithCompiledAutograd = wrap_test_class(
    test_higher_order_ops.HigherOrderOpTests
)
FuncTorchHigherOrderOpTestsWithCompiledAutograd = wrap_test_class(
    test_higher_order_ops.FuncTorchHigherOrderOpTests
)
ActivationCheckpointingTestsWithCompiledAutograd = wrap_test_class(
    test_higher_order_ops.ActivationCheckpointingTests
)

if torch.distributed.is_available() and HAS_CUDA_AND_TRITON:
    test_dtensor = load_test_module("distributed/tensor/test_dtensor_compile")
    TestDTensorCompileWithCompiledAutograd = wrap_test_class(
        test_dtensor.TestDTensorCompile
    )

xfail_hops = {"local_map_hop"}


class TestCompiledAutogradOpInfo(TestCase):
    def setUp(self) -> None:
        super(TestCase, self).setUp()
        reset()

    def tearDown(self) -> None:
        super(TestCase, self).tearDown()
        reset()

    @ops(
        list(filter(lambda op: op.name not in xfail_hops, hop_db)),
        allowed_dtypes=(torch.float,),
    )
    def test_hops_in_bwd(self, device, dtype, op):
        def create_bwd_fn_closure(op_args, op_kwargs):
            op_out_ref = []

            class Foo(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, grad):
                    out = op.op(*op_args, **op_kwargs)
                    op_out_ref.append(out)
                    return grad

            def fn(x):
                return Foo.apply(x).sum()

            return fn, op_out_ref

        # Note: requires_grad=False because aot dispatch is already covered elsewhere
        for inp in op.sample_inputs(device, dtype, requires_grad=False):
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            eager_args = (*input, *inp.args)
            eager_kwargs = inp.kwargs
            compiled_args = deepcopy(eager_args)
            compiled_kwargs = deepcopy(eager_kwargs)

            # 1. Run eager
            torch.manual_seed(123)
            dummy = torch.randn(2, 2, dtype=dtype, device=device, requires_grad=True)
            fn, op_out_ref = create_bwd_fn_closure(eager_args, eager_kwargs)
            fn(dummy).backward()
            self.assertEqual(len(op_out_ref), 1)
            expected = op_out_ref[0]

            # 2. Run under CA
            torch.manual_seed(123)
            dummy = torch.randn(2, 2, dtype=dtype, device=device, requires_grad=True)
            fn, op_out_ref = create_bwd_fn_closure(compiled_args, compiled_kwargs)
            with compiled_autograd._enable(make_compiler_fn(backend="aot_eager")):
                fn(dummy).backward()
            self.assertEqual(len(op_out_ref), 1)
            actual = op_out_ref[0]

            self.assertEqual(expected, actual)


instantiate_device_type_tests(TestCompiledAutogradOpInfo, globals())
instantiate_parametrized_tests(TestCompiledAutograd)

if __name__ == "__main__":
    if HAS_CPU:
        run_tests(needs="filelock")
