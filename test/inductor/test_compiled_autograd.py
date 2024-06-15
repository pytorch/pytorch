# Owner(s): ["module: inductor"]
import functools
import io
import logging
import re
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

import logging
torch_log = logging.getLogger("torch")

import torch
import torch.nn as nn
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd, config
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
from torch.testing._internal.logging_utils import logs_to_string

# note: these tests are not run on windows due to inductor_utils.HAS_CPU


def make_compiler_fn(fullgraph=True, dynamic=True):
    def _compiler_fn(gm):
        """Same as torch.compile() but counts number of compiles"""

        torch_log.warning("Compiling autograd?")

        def _inner_compiler(gm_, example_inputs_):
            counters["compiled_autograd"]["compiles"] += 1
            return inductor.compile(gm_, example_inputs_)

        return torch.compile(
            gm, backend=_inner_compiler, fullgraph=fullgraph, dynamic=dynamic
        )

    return _compiler_fn


compiler_fn = make_compiler_fn()


# TODO(jansel): hooks as lambdas creates recompiles in dynamo, we should fix that
def hook1(grad):
    return grad * 2


def hook2(grads):
    return (grads[0] + 1,)


def hook3(gI, gO):
    return (torch.sin(gI[0]) + gO[0],)


class TestCompiledAutograd(TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch._logging.set_logs(compiled_autograd_verbose=False)
        config.compiled_autograd = False
        compiled_autograd.reset()

    def tearDown(self) -> None:
        super().tearDown()
        torch._logging.set_logs(compiled_autograd_verbose=False)
        config.compiled_autograd = False
        compiled_autograd.reset()

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
            with compiled_autograd.enable(compiler_fn):
                opt_fn = torch.compile(fn) if compile_fn else fn
                actual = list(opt_fn())
            self.assertEqual(expected, actual)
            self.assertEqual(counters["compiled_autograd"]["captures"], captures)
            self.assertEqual(counters["compiled_autograd"]["compiles"], compiles)

    def test_dynamo_flaky_segfault(self):
        import os
        import subprocess

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
        with torch._dynamo.compiled_autograd.enable(compiler_fn):
            loss.backward()
        assert(w.grad is not None)

    inner()
    torch._dynamo.reset()
    inner()

main()
        """
        # Run it three times to catch bad dynamo state resets
        for _ in range(3):
            try:
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    stderr=subprocess.STDOUT,
                    # On Windows, opening the subprocess with the default CWD makes `import torch`
                    # fail, so just set CWD to this script's directory
                    cwd=os.path.dirname(os.path.realpath(__file__)),
                )
            except subprocess.CalledProcessError as e:
                if e.returncode < 0:
                    self.fail("Subprocess exited with a fatal signal")

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

    def test_torch_compile_api_inductor(self):
        def fn():
            torch.manual_seed(123)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Sigmoid(),
            )

            res = []
            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.backward()
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                model.zero_grad()
            return res

        expected = fn()
        with config.patch(compiled_autograd=True):
            compiled_fn = torch.compile(fn)
        actual = compiled_fn()
        self.assertEqual(expected, actual)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_api_aot_eager(self):
        def fn():
            torch.manual_seed(123)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Sigmoid(),
            )

            res = []
            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.backward()
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                model.zero_grad()
            return res

        expected = fn()
        with config.patch(compiled_autograd=True):
            compiled_fn = torch.compile(fn, backend="aot_eager")
        actual = compiled_fn()
        self.assertEqual(expected, actual)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_api_eager(self):
        def fn():
            torch.manual_seed(123)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Sigmoid(),
            )

            res = []
            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.backward()
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                model.zero_grad()
            return res

        expected = fn()
        with config.patch(compiled_autograd=True):
            compiled_fn = torch.compile(fn, backend="eager")
        actual = compiled_fn()
        self.assertEqual(expected, actual)
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
                    # should be boxed inputs
                    assert len(placeholders) == 1
                    pass
                else:
                    assert len(placeholders) > 1

                return gm_

            return torch.compile(gm, backend=inner_compiler)

        fwd_compiler_fn = functools.partial(eager_with_check, is_bwd=False)
        bwd_compiler_fn = functools.partial(eager_with_check, is_bwd=True)

        def fn(inputs):
            args_0, args_1, args_2 = inputs
            out = torch.mm(args_0, args_1)
            out = torch.mm(out, args_2)
            loss = out.sum()
            with compiled_autograd.enable(bwd_compiler_fn):
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
        activ = torch.ones(100) * 2
        inputs = [param, activ]
        proxies, _ = compiler.begin_capture(inputs=inputs, sizes=[])
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
                compiled_fn=compiled_fn, inputs=[param, activ], sizes=(), hooks=()
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
            out = compiled_fn(inputs)
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

        # TODO(jansel): we should be able to get this count to 1
        self.check_output_and_recompiles(fn, count=2)

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

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_issue106555(self):
        DEVICE = torch.device("cuda:0")
        NUM_FEATURES = 256

        def bias_sigmoid_mul(x1, x2, bias):
            x2 = torch.sigmoid(x2 + bias)
            y = x1 * x2
            return y

        bias_sigmoid_mul_jit = torch.compile(bias_sigmoid_mul)

        class ModuleWithJit(nn.Module):
            def __init__(self):
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
            def __init__(self):
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

        torch.cuda.set_device(device=DEVICE)
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

        for iteration in range(10):
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

        for i in range(0, 5):
            with compiled_autograd.enable(compiler_fn):
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

        self.check_output_and_recompiles(fn, count=2)

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

        self.check_output_and_recompiles(fn, count=2)

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

        self.check_output_and_recompiles(fn, count=2)

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

        self.check_output_and_recompiles(fn, count=2)

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
            fn, count=2, compiler_fn=make_compiler_fn(fullgraph=False)
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

        self.check_output_and_recompiles(fn, count=2)

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

        self.check_output_and_recompiles(fn, count=2)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
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
                1, 10, requires_grad=True, dtype=torch.float16, device="cuda"
            )
            out = _fn(x)
            loss = out.sum()
            loss.backward()

        with compiled_autograd.enable(compiler_fn):
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

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_custom_fn_output_metadata(self):
        def my_compiler_fn(gm):
            for node in gm.graph.nodes:
                if isinstance(node.target, torch._ops.OpOverload):
                    assert (
                        node.target._name != "aten::_to_copy"
                    ), "there should be no implicit copies (e.g. dtype casting)"

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
                1, 10, requires_grad=True, dtype=torch.float16, device="cuda"
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
            fn, count=[2, 6], compiler_fn=make_compiler_fn(fullgraph=False)
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
            fn, count=2, compiler_fn=make_compiler_fn(fullgraph=False)
        )
        self.assertEqual(counters["stats"]["unique_graphs"], 5)  # 3 fw, 2 bw

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
            fn, count=[2, 6], compiler_fn=make_compiler_fn(fullgraph=False)
        )
        self.assertEqual(counters["stats"]["unique_graphs"], 9)  # 3 fw, 6 bw

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

    def test_autograd_cpp_node(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

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

TORCH_LIBRARY(test_autograd_cpp_node, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = torch.utils.cpp_extension.load_inline(
            name="test_autograd_cpp_node",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                out = torch.ops.test_autograd_cpp_node.custom_op_backed_by_autograd_fn(
                    x
                )
                loss = out.sum()
                loss.backward()
                yield x.grad

        # compiles for 10 (static) and 100 (dynamic)
        self.check_output_and_recompiles(fn, 2)

    def test_autograd_cpp_node_id(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

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
  static constexpr bool is_traceable = true;

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

TORCH_LIBRARY(test_autograd_cpp_node_id, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("custom_op_backed_by_autograd_fn2", custom_op_backed_by_autograd_fn2);
}
        """

        module = torch.utils.cpp_extension.load_inline(
            name="test_autograd_cpp_node_id",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def same_autograd_fn():
            def fn():
                x = torch.ones(10, 10, requires_grad=True)
                out = (
                    torch.ops.test_autograd_cpp_node_id.custom_op_backed_by_autograd_fn(
                        x
                    )
                )
                loss = out.sum()
                loss.backward()
                yield x.grad

            yield from fn()  # compile
            yield from fn()  # reuse
            yield from fn()  # reuse
            yield from fn()  # reuse

        self.check_output_and_recompiles(same_autograd_fn, 1)

        def different_autograd_fn():
            def fn(op):
                x = torch.ones(10, 10, requires_grad=True)
                out = op(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

            op1 = torch.ops.test_autograd_cpp_node_id.custom_op_backed_by_autograd_fn
            op2 = torch.ops.test_autograd_cpp_node_id.custom_op_backed_by_autograd_fn2
            yield from fn(op1)  # compile
            yield from fn(op2)  # compile
            yield from fn(op1)  # reuse
            yield from fn(op2)  # reuse

        self.check_output_and_recompiles(different_autograd_fn, 2)

    def test_autograd_cpp_node_saved(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

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
    int i = ctx->saved_data["int"].toInt();
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

TORCH_LIBRARY(test_autograd_cpp_node_saved, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = torch.utils.cpp_extension.load_inline(
            name="test_autograd_cpp_node_saved",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            fixed = torch.ones(2, 2)
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                y = torch.randn(i, i)
                out = torch.ops.test_autograd_cpp_node_saved.custom_op_backed_by_autograd_fn(
                    x, y, fixed
                )
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn, 2)

    def test_autograd_cpp_node_saved_dynamic(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

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

TORCH_LIBRARY(test_autograd_cpp_node_saved_dynamic, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = torch.utils.cpp_extension.load_inline(
            name="test_autograd_cpp_node_saved_dynamic",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                out = torch.ops.test_autograd_cpp_node_saved_dynamic.custom_op_backed_by_autograd_fn(
                    x
                )
                loss = out.sum()
                loss.backward()
                yield x.grad

        # can bring this down to 2 if we support dynamic shapes
        # instead of collecting the saved_data's tensor hash
        self.check_output_and_recompiles(fn, 5)

    def test_autograd_cpp_node_data_dependent(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;
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
    assert(ctx->saved_data["bool"].isBool());
    assert(ctx->saved_data["int"].isInt());
    int i = ctx->saved_data["int"].toInt();

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

TORCH_LIBRARY(test_autograd_cpp_node_data_dependent, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("reset", reset);
}
        """

        module = torch.utils.cpp_extension.load_inline(
            name="test_autograd_cpp_node_data_dependent",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            torch.ops.test_autograd_cpp_node_data_dependent.reset()
            for i in [10, 10, 10, 10]:
                x = torch.ones(i, i, requires_grad=True)
                y = torch.randn(i, i)
                (
                    out1,
                    out2,
                ) = torch.ops.test_autograd_cpp_node_data_dependent.custom_op_backed_by_autograd_fn(
                    x, y
                )
                loss = (out1 + out2).sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn, 3)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_free_activation_memory(self):
        self.assertTrue(torch.cuda.memory_allocated() == 0)

        # Use an op to check that the memory is freed by the time the op is executed
        def assertion_impl(to_clone):
            mem_allocated = torch.cuda.memory_allocated()
            self.assertTrue(
                mem_allocated < 4000000, "activations should have been freed"
            )
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
            activations = [torch.ones(1000000, dtype=torch.float32, device="cuda")]
            self.assertTrue(torch.cuda.memory_allocated() > 4000000)

            out = compiled_fn(activations)
            self.assertTrue(len(activations) == 0)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_free_activation_memory_subclass(self):
        # cover the case when aot inputs have subclasses, resulting in a different runtime wrapper
        self.assertTrue(torch.cuda.memory_allocated() == 0)

        # Use an op to check that the memory is freed by the time the op is executed
        def assertion_impl(to_clone):
            mem_allocated = torch.cuda.memory_allocated()
            self.assertTrue(
                mem_allocated < 1200000, "some activations should have been freed"
            )
            self.assertTrue(
                mem_allocated > 800000,
                "currently subclasses don't seem to be freed in inductor",
            )
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
                        torch.ones((1, 100000), device="cuda"),  # 400,000 bytes
                        torch.ones((1, 100000), device="cuda"),  # 400,000 bytes
                    ],
                    None,
                )[
                    0
                ],  # NestedTensor
                torch.ones((1, 100000), device="cuda"),  # 400,000 bytes
            ]
            # 1,200,000 bytes (3 * 4 * 100,000 bytes)
            self.assertTrue(torch.cuda.memory_allocated() > 1200000)

            out = compiled_fn(activations)
            self.assertTrue(len(activations) == 0)

    def test_callback_graph_break(self):
        # TODO(yf225): Test graph break between queue_callback and exec_final_callbacks
        # TODO(yf225): Memory check for tensors involved the callback
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
        with compiled_autograd.enable(make_compiler_fn(fullgraph=False)):
            b = MyFunc.apply(a)
            b.sum().backward()

        self.assertEqual(called[0], 1)

    def test_callback_adds_callback_graph_break(self):
        called = [0]

        def callback_final():
            called[0] += 1

        def callback_adds_callback():
            called[0] += 1
            torch.autograd.Variable._execution_engine.queue_callback(callback_final)

        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad):
                torch.autograd.Variable._execution_engine.queue_callback(callback_adds_callback)
                torch._dynamo.graph_break()
                return grad

        a = torch.rand((3, 3), requires_grad=True)
        torch._dynamo.reset()
        with compiled_autograd.enable(make_compiler_fn(fullgraph=False)):
            b = MyFunc.apply(a)
            b.sum().backward()
        self.assertEqual(called[0], 2)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_cudagraphs_cpu_division(self):
        from torch._dynamo.testing import reduce_to_scalar_loss

        model = torch.nn.Linear(10, 10, dtype=torch.float16).cuda()
        inputs = torch.randn(10, 10, dtype=torch.float16).cuda()
        out = model(inputs)
        loss = reduce_to_scalar_loss(out)

        stderr_msgs = io.StringIO()
        with mock.patch("sys.stderr", stderr_msgs), compiled_autograd.enable(
            compiler_fn
        ):
            torch._inductor.config.triton.cudagraphs = True
            loss.backward()
            torch._inductor.config.triton.cudagraphs = False

        self.assertFalse("skipping cudagraphs" in stderr_msgs.getvalue())

    def test_cudagraphs_cpu_graph(self):
        from torch._dynamo.testing import reduce_to_scalar_loss

        model = torch.nn.Linear(10, 10, dtype=torch.float16)
        inputs = torch.randn(10, 10, dtype=torch.float16)
        out = model(inputs)
        loss = reduce_to_scalar_loss(out)

        with compiled_autograd.enable(compiler_fn):
            torch._inductor.config.triton.cudagraphs = True
            loss.backward()
            torch._inductor.config.triton.cudagraphs = False

        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_cudagraphs_sdpa(self):
        query = torch.rand(
            32, 8, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True
        )
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        with config.patch(compiled_autograd=True), inductor_config.patch(
            "triton.cudagraphs", True
        ):
            opt_bwd = torch.compile(lambda: out.sum().backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
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
        with config.patch(compiled_autograd=True), inductor_config.patch(
            "triton.cudagraphs", True
        ):
            opt_bwd = torch.compile(lambda: out.backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        # Compiled autograd lifts custom autograd.Function bwd instead of tracing it.
        # Must skip since we do not know if the cpu scalar will be used only in ATen/prim ops.
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_cudagraphs_cpu_scalar_used_in_cpp_custom_op(self):
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

        module = torch.utils.cpp_extension.load_inline(
            name="test_cudagraphs_cpu_scalar_used_in_cpp_custom_op",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        x = torch.randn(2, 2, requires_grad=True, device="cuda")
        with config.patch(compiled_autograd=True), inductor_config.patch(
            "triton.cudagraphs", True
        ):
            out = torch.ops.test_cudagraphs_cpu_scalar_used_in_cpp_custom_op.custom_op_backed_by_autograd_fn(
                x
            )
            opt_bwd = torch.compile(lambda: out.sum().backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        # always safe to move, since we trace into the autograd::function bwd and can see if it's only used by aten ops
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    def test_verbose_logs_graph(self):
        torch._logging.set_logs(compiled_autograd_verbose=True)

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
            "SumBackward0 (NodeCall 1)",
            "ReluBackward0 (NodeCall 2)",
            "AddmmBackward0 (NodeCall 3)",
            "TBackward0 (NodeCall 4)",
            "torch::autograd::AccumulateGrad (NodeCall 5)",
            "ReluBackward0 (NodeCall 6)",
            "AddmmBackward0 (NodeCall 7)",
            "TBackward0 (NodeCall 8)",
            "torch::autograd::AccumulateGrad (NodeCall 9)",
            "torch::autograd::AccumulateGrad (NodeCall 10)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
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
            self.check_output_and_recompiles(fn, count=2)

        patterns1 = [
            r".*Cache miss due to new autograd node: torch::autograd::GraphRoot \(NodeCall 0\) with key size (\d+), "
            r"previous key sizes=\[\]\n",
        ]

        # recompile
        patterns2 = [
            r".*Cache miss due to changed shapes: marking size idx (\d+) of torch::autograd::GraphRoot \(NodeCall 0\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of SumBackward0 \(NodeCall 1\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of SumBackward0 \(NodeCall 1\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of ReluBackward0 \(NodeCall 2\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of AddmmBackward0 \(NodeCall 3\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of torch::autograd::AccumulateGrad "
            r"\(NodeCall 5\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of ReluBackward0 \(NodeCall 6\) as dynamic\n",
        ]

        all_logs = logs.getvalue()

        pattern1 = r"".join(patterns1)
        matches1 = re.findall(pattern1, all_logs)
        self.assertEqual(len(matches1), 1)
        assert isinstance(
            matches1[0], str
        )  # for a single match: matches1=['match'], for multiple matches: matches1=[('match1', 'match2')]...
        self.assertEqual(len(matches1), len(patterns1))

        pattern2 = r"".join(patterns2)
        matches2 = re.findall(pattern2, all_logs)
        self.assertEqual(len(matches2), 1)
        self.assertEqual(len(matches2[0]), len(patterns2))

    def test_snapshot_verbose_logs_flag(self):
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
            with compiled_autograd.enable(compiler_fn):
                # unused, verbose level already snapshot with contextmanager
                torch._logging.set_logs(compiled_autograd_verbose=True)
                fn()

        unexpected_logs = [
            "SumBackward0 (NodeCall 1)",
            "ReluBackward0 (NodeCall 2)",
            "AddmmBackward0 (NodeCall 3)",
            "TBackward0 (NodeCall 4)",
            "torch::autograd::AccumulateGrad (NodeCall 5)",
            "ReluBackward0 (NodeCall 6)",
            "AddmmBackward0 (NodeCall 7)",
            "TBackward0 (NodeCall 8)",
            "torch::autograd::AccumulateGrad (NodeCall 9)",
            "torch::autograd::AccumulateGrad (NodeCall 10)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
        ]

        self.assertEqual(sum(1 for e in unexpected_logs if e in logs.getvalue()), 0)


def load_test_module(name):
    testdir = Path(__file__).absolute().parent.parent
    with mock.patch("sys.path", [*sys.path, str(testdir)]):
        return SourceFileLoader(
            name, str(testdir / f"{name.replace('.', '/')}.py")
        ).load_module()


def make_wrapped(fn, fullgraph):
    @functools.wraps(fn)
    def wrapped(self):
        torch._dynamo.reset()
        with compiled_autograd.enable(make_compiler_fn(fullgraph=fullgraph)):
            out = fn(self)

        return out

    return wrapped


def wrap_test_class(orig_cls):
    dct = orig_cls.__dict__.copy()
    for name in list(dct.keys()):
        fn = dct[name]
        if not callable(fn):
            continue
        elif known_failures_re.match(name) or name in known_failing_tests:
            dct[name] = unittest.expectedFailure
        elif name.startswith("test_"):
            fullgraph = name not in known_graph_breaks_tests
            dct[name] = make_wrapped(fn, fullgraph)

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
}

# These groups of tests aren't supported yet
known_failures_re = re.compile(
    r"^test_(sparse|profiler|gradcheck|checkpoint|named_tensor)"
)

# Bugs needing investigation:
known_failing_tests = {
    "test_current_graph_task_execution_order",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function <
    "test_input_buffer_accum",  # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
    "test_graph_save_on_cpu_cuda",  # AssertionError: 0 not greater than 0
    "test_graph_save_on_cpu",  # torch._dynamo.exc.BackendCompilerFailed: backend='inner_compiler' raised:
    "test_reentrant_with_leaf_variable_hook",  # torch._dynamo.exc.Unsupported: inline in skipfiles: RemovableHandle.
    "test_reentrant_with_non_leaf_variable_hook",  # torch._dynamo.exc.Unsupported: inline in skipfiles: RemovableHan
    "test_saved_variable_saved_original_inplace_detach",  # AssertionError: RuntimeError not raised
    "test_saving_variable_to_disk",  # Cannot call numel() on tensor with symbolic sizes/strides
    "test_setitem_mask",  # torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: It appears that you're
    "test_wrapped_number_saved_variable_hooks",  # RuntimeError: this hook should not be called
    "test_accumulate_grad_tensor_reference",  # backend='inner_compiler' raised:
    "test_anomaly_grad_warnings",  # "one of the variables needed for gradient computation has been modified by an...
    "test_autograd_inplace_views_cross_dtype",  # view_fn not supported by compiled autograd
    "test_current_node",  # TorchDispatchMode not yet implemented for compiled autograd
    "test_custom_function_exception",  # "Simulate error on backward pass" does not match "type object 'SimulateBackwa...
    "test_grad_batched_grad",  # Cannot access storage of BatchedTensorImpl
    "test_index_backward_does_not_save_tensor",  # dynamic shape operator: aten.nonzero.default
    "test_post_accumulate_grad_hook_ordering",  # tensor_post_acc_grad_hooks not implemented for compiled autograd
    "test_post_accumulate_grad_hook_returns_not_None",  # "hooks should return None." does not match
    "test_reentrant_child_error",  # "Simulate error" does not match "type object 'ReentrantFunc' has no attribute...
    "test_retain_grad_cycle",  # retains_grad_hooks not implemented for compiled autograd
    "test_retain_grad_inplace",  # retains_grad_hooks not implemented for compiled autograd
    "test_retain_grad_inplace_over_view",  # retains_grad_hooks not implemented for compiled autograd
    "test_retains_grad_can_always_observe_tensor_prehook",  # retains_grad_hooks not implemented for compiled autograd
    "test_retains_grad_inplace_multiple_outputs",  # retains_grad_hooks not implemented for compiled autograd
    "test_to_sparse_backward",  # backend='inner_compiler' raised:
    "test_accumulate_grad",  # RuntimeError: compiled_autograd does not support create_graph
    "test_anomaly_assign_parent_cleanup",  # RuntimeError: compiled_autograd does not support create_graph
    "test_anomaly_mode_no_check_nan",  # RuntimeError: compiled_autograd does not support AnomalyMode
    "test_backward_create_graph_warns",  # RuntimeError: compiled_autograd does not support create_graph
    "test_backward_with_nonleaf_inputs",  # RuntimeError: compiled_autograd does not support create_graph
    "test_create_graph_and_full_backward_hook_cycle",  # RuntimeError: compiled_autograd does not support create_graph
    "test_current_graph_task_id",  # torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor int
    "test_custom_autograd_repeated_grad_grad",  # RuntimeError: compiled_autograd does not support create_graph
    "test_custom_function_forward_mode_forward_is_no_op",  # AttributeError: type object 'MyFn'
    "test_custom_function_forward_mode_inplace_checks",  # AttributeError: type object 'InplaceFn'
    "test_custom_function_forward_mode_view_checks",  # AttributeError: type object 'ViewFn'
    "test_custom_function_forward_mode_wrong_formula",  # AttributeError: type object 'UserFn'
    "test_default_saved_variable_hooks_double_backward",  # RuntimeError: compiled_autograd does not support create_graph
    "test_full_backward_hook_double_backward",  # RuntimeError: compiled_autograd does not support create_graph
    "test_function",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad_materialize_grads",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad_nonleaf",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad_nonleaf_many_outputs",  # RuntimeError: compiled_autograd does not support create_graph
    "test_hessian_vector",  # RuntimeError: compiled_autograd does not support create_graph
    "test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_False",  # AttributeError: type object
    "test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_True",  # AttributeError: type object
    "test_hook_edge_case_when_called_with_grad",  # retains_grad_hooks NYI
    "test_inplace_on_view_backward",  # RuntimeError: compiled_autograd does not support create_graph
    "test_multi_grad_any_hooks",  # register_multi_grad_hook NYI
    "test_multi_grad_all_hooks",  # retains_grad_hooks NYI
    "test_nested_anomaly_detect_nan",  # RuntimeError: compiled_autograd does not support create_graph
    "test_nested_anomaly_printstack_cleanup",  # RuntimeError: compiled_autograd does not support create_graph
    "test_once_differentiable",  # RuntimeError: compiled_autograd does not support create_graph
    "test_prehook_ordering",  # retains_grad_hooks NYI
    "test_retain_grad",  # RuntimeError: retains_grad_hooks not implemented for compiled autograd
    "test_saved_variable_packing_unpacking_saved_original_with_hooks",  # RuntimeError: compiled_autograd
    "test_select_sum",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_unrelated_inputs",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_will_engine_execute_node",  # retains_grad_hooks NYI
    "test_backward_to_node",  # retains_grad_hooks NYI
    "test_anomaly_detect_nan",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function aten.add.Tensor(
    "test_autograd_multiple_views_python",  # torch._dynamo.exc.Unsupported: call_function args: TensorVariable(
    "test_autograd_node_isinstance",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertIsInstance
    "test_autograd_simple_views_python",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function
    "test_custom_autograd_no_early_free",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_custom_function_cycle",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_custom_function_error",  # AssertionError: "must implement either the backward" does not match "call_function
    "test_custom_function_non_tensor_inputs_outputs",  # torch._dynamo.exc.Unsupported: call_function
    "test_custom_function_save_for_forward",  # torch._dynamo.exc.Unsupported: call_function
    "test_custom_function_setup_context_multi_input",  # torch._dynamo.exc.Unsupported: call_function args
    "test_custom_function_setup_context_multi_output",  # torch._dynamo.exc.Unsupported: call_function args
    "test_deep_reentrant",  # torch._dynamo.exc.InternalTorchDynamoError: '<' not supported between instances of
    "test_dont_materialize_grads",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertIsNone
    "test_function_returns_undefined_tensor",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function
    "test_grad_mode_restored_reentrant",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertTrue
    "test_invalid_gradients",  # AssertionError: "expected shape" does not match "The size of tensor a (5) must match
    "test_mark_non_differentiable_mixed",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertTrue
    "test_materialize_grads",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_naughty_autograd_function_stashing_ctx",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function
    "test_no_grad_copy",  # torch._dynamo.exc.Unsupported: call_function args: TensorVariable() SkipFunctionVariable()
    "test_no_grad_copy_sparse",  # torch._dynamo.exc.Unsupported: Tensor.data_ptr
    "test_reentrant_priority",  # torch._dynamo.exc.InternalTorchDynamoError: '<' not supported between instances of
    "test_reentrant_with_callbacks_both_depths",  # torch._dynamo.exc.Unsupported: call_method UserDefinedObjectVariable
    "test_reentrant_with_callbacks_depth_0",  # torch._dynamo.exc.Unsupported: call_method UserDefinedObjectVariable
    "test_reentrant_with_callbacks_depth_1",  # torch._dynamo.exc.Unsupported: Tensor.requires_grad_
    "test_return_duplicate",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_return_duplicate_inplace",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_return_leaf",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_save_none_for_backward",  # AssertionError:
    "test_save_output_nr",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_saved_variables_deprecated",  # torch._dynamo.exc.Unsupported: UNPACK_SEQUENCE SkipFunctionVariable()
    "test_set_materialize_non_diff_grads",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertIsNone
    "test_setup_context_when_forward_has_default_args",  # torch._dynamo.exc.Unsupported: call_function args
    "test_simple_reentrant",  # torch._dynamo.exc.Unsupported: call_method SkipFunctionVariable() sum [] {}
    "test_lobpcg",  # torch._dynamo.exc.Unsupported: 'call_function LOBPCGAutogradFunction.backward in skip_files
    "test_backward_dict_grad_for_nontensor",  # AssertionError: "non-Tensor-like types" does not match "'skip function
    "test_backward_dict_invalid_keys",  # AssertionError: "to have keys {'x'}" does not match "'skip function
    "test_backward_dict_requires_keys_for_input_optional_tensors",  # AssertionError: "to have keys {.*'y'.*}"
    "test_backward_dict_requires_keys_for_input_tensors",  # AssertionError: "to have keys {.*'y'.*}" does not
    "test_backward_grads_are_tensor_or_none",  # AssertionError: "either None or a Tensor" does not match "'
    "test_backward_impl_on_existing_op",  # torch._dynamo.exc.Unsupported: 'skip function
    "test_backward_returns_dict",  # AssertionError: "to be a dict" does not match "'skip function
    "test_backward_tensorlist_input_requires_list_grads",  # AssertionError: "list of gradients" does not
    "test_backward_tensorlist_input_requires_list_grads_none_or_Tensor",  # AssertionError: "None or Tensor"
    "test_backward_tensorlist_input_requires_list_grads_with_same_numel",  # AssertionError: "3 gradients
    "test_save_for_backward_inputs_are_namedtuple",  # torch._dynamo.exc.Unsupported: 'skip function
    "test_setitem",  # AssertionError: Tensor-likes are not close!
    "test_grad_nonleaf_register_hook",  # IndexError: list index out of range (NB: x.grad = y where both x and y are input tensors)
    "test_scalar_grad_mixed_device",  # Fake Tensors aren't propagating device properly for 0-dim grads
}

if not HAS_CUDA:
    # Found Tesla M60 which is too old to be supported by the triton GPU compiler
    known_failing_tests.add("test_type_conversions")

test_autograd = load_test_module("test_autograd")
test_custom_ops = load_test_module("test_custom_ops")

TestAutogradWithCompiledAutograd = wrap_test_class(test_autograd.TestAutograd)
TestCustomOpWithCompiledAutograd = wrap_test_class(test_custom_ops.TestCustomOp)

if __name__ == "__main__":
    if HAS_CPU:
        run_tests(needs="filelock")
