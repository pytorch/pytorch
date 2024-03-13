# Owner(s): ["module: dynamo"]
import functools
import math
import sys
import unittest  # noqa: F811
from importlib import import_module

import torch
import torch._dynamo.config

import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint
from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch.testing._internal.common_utils import IS_WINDOWS, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen, checkpoint

requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


def count_ops(
    gm, args, freq=None, freq_ge=None, op=None, freqs=None, freqs_ge=None, ops=None
):
    def match_rng_op(node, op):
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            if node.name == "run_and_save_rng_state":
                return node.args[0] == op
            elif node.name == "run_with_rng_state":
                return node.args[1] == op
        return False

    # assert ((freq or freq_ge) and op) or ((freqs or freqs_ge) and ops)
    if op is not None:
        ops = [op]
    if freq is not None:
        freqs = [freq]
    if freq_ge is not None:
        freqs_ge = [freq_ge]
    if freqs:
        for op, freq in zip(ops, freqs):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            assert (
                actual_count == freq
            ), f"In graph {gm}, expected {op} to have occurred {freq} times in the graph, but got {actual_count}."
    else:
        assert freqs_ge is not None
        for op, freq_ge in zip(ops, freqs_ge):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            assert (
                actual_count >= freq_ge
            ), f"In graph {gm}, expected {op} to have occurred at least {freq_ge} times in the graph, but got {actual_count}."
    return gm


class _InvalidContext:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def _invalid_context_gen():
    return _InvalidContext(), _InvalidContext()


def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def op_count(gm):
    result = 0
    for node in gm.graph.nodes:
        if "call" in node.op:
            result += 1
    return result


def _get_custom_policy(no_recompute_list=None):
    def _custom_policy(mode, func, *args, **kwargs):
        return func in no_recompute_list

    return _custom_policy


class ActivationCheckpointingViaTagsTests(torch._dynamo.test_case.TestCase):
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        torch.manual_seed(0)
        result = torch.compile(fn, fullgraph=fullgraph, backend=backend)(*cloned_args)
        result.sum().backward()

        if not skip_check:
            self.assertEqual(
                result,
                expected,
                msg="Output mismatch between torch.compile and eager versions",
            )
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(
                    arg.grad,
                    cloned_arg.grad,
                    msg="Gradient mismatch between torch.compile and eager versions",
                )

    def _compare_orig_and_checkpointed_fns(
        self, orig_fn, checkpointed_fn, *args, fullgraph=True
    ):
        # The original version and the checkpointed version of the same function
        # should produce the same outputs and the same gradients under torch.compile.

        # Run original version
        cloned_args_orig_fn = []
        for arg in args:
            cloned_args_orig_fn.append(
                arg.clone().detach().requires_grad_(arg.requires_grad)
            )
        torch.manual_seed(0)
        compiled_orig_fn = torch.compile(
            orig_fn, fullgraph=fullgraph, backend="inductor"
        )
        result_orig_fn = compiled_orig_fn(*cloned_args_orig_fn)
        result_orig_fn.sum().backward()

        # Run checkpointed version
        cloned_args_checkpointed_fn = []
        for arg in args:
            cloned_args_checkpointed_fn.append(
                arg.clone().detach().requires_grad_(arg.requires_grad)
            )
        torch.manual_seed(0)
        compiled_checkpointed_fn = torch.compile(
            checkpointed_fn, fullgraph=fullgraph, backend="inductor"
        )
        result_checkpointed_fn = compiled_checkpointed_fn(*cloned_args_checkpointed_fn)
        result_checkpointed_fn.sum().backward()

        # Check that outputs and gradients are equal
        self.assertEqual(
            result_orig_fn,
            result_checkpointed_fn,
            msg="Output mismatch between the original version and the checkpointed version of the same function",
        )
        for cloned_arg_orig_fn, cloned_arg_checkpointed_fn in zip(
            cloned_args_orig_fn, cloned_args_checkpointed_fn
        ):
            self.assertEqual(
                cloned_arg_orig_fn.grad,
                cloned_arg_checkpointed_fn.grad,
                msg="Gradient mismatch between the original version and the checkpointed version of the same function",
            )

    @requires_cuda
    def test_tags_function(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda
    def test_tags_function_via_global_checkpoint(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # This goes through VariableBuilder
            return checkpoint(gn, torch.sin(x), y, use_reentrant=True)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda
    def test_tags_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda
    def test_tags_multiple_checkpoints(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            x = torch.sin(z)
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            return z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=6, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda
    def test_tags_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        x = torch.randn(10, 10, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)

    @requires_cuda
    def test_tags_decomps(self):
        # Ensures that tags are passed on through decompositions as well
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.nn.functional.gelu(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        x = torch.randn(10, 10, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
        )
        self._validate(fn, backend, x)

    @requires_cuda
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_recomputed_rand(self):
        def gn(x, y):
            return torch.sigmoid(torch.rand_like(x) * y) * x

        def fn(x, y):
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            return z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_rand(self):
        def gn(x, y):
            x = torch.mm(x, y)
            x = torch.mm(x, y)
            return x

        def fn(x, y):
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            x = torch.sin(x)
            # x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            return x

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # backend = "aot_eager"
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_dropout(self):
        # Figure out a way to test the number of inductor_random calls
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, x):
                return self.dropout(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=True)

        x = torch.randn(10, 10, device="cuda", requires_grad=True)
        backend = "inductor"
        # rand decomps do not have have numerical results as eager
        self._validate(fn, backend, x, skip_check=True)

    @requires_cuda
    def test_fallback(self):
        def gn(x, y):
            torch._dynamo.graph_break()
            a = torch.sigmoid(torch.matmul(x, y))
            torch._dynamo.graph_break()
            return torch.cos(a)

        def fn(x, y):
            return torch.cos(checkpoint(gn, torch.sin(x), y, use_reentrant=False))

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        # One graph for torch.sin on the input, and other for torch.cos.
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(cnt.graphs), 2)

    @requires_cuda
    def test_kwargs(self):
        def gn(x, y, z=None):
            a = torch.matmul(x, y)
            if z is not None:
                return torch.matmul(a, z)
            return a

        def fn(x, y, z):
            return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        z = torch.randn(4, 4, requires_grad=True)
        args = (x, y, z)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(cnt.graphs), 1)

        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        # one for checkpoint, and 3 for x, y, z
        self.assertEqual(len(wrap_node.args), 4)

        body_function = getattr(cnt.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)

    @requires_cuda
    def test_symints_location(self):
        def gn(x, y):
            return torch.matmul(x, torch.nn.functional.dropout(y, 0.5))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)
        opt_fn = torch.compile(fn, backend=cnt)

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)

        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(cnt.graphs), 2)
        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        self.assertEqual(len(wrap_node.args), 3)

    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @torch._dynamo.config.patch(
        "_experimental_support_context_fn_in_torch_utils_checkpoint", True
    )
    def test_compile_selective_checkpoint_gemm_only(self):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return _pt2_selective_checkpoint_context_fn_gen(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # We would've expected 6 here
            # (2 matmul recompute and 2 mm ops per fwd matmul, so 2 + 2 * 2 = 6)
            # if we didn't enable selective checkpointing.
            freq=4,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @torch._dynamo.config.patch(
        "_experimental_support_context_fn_in_torch_utils_checkpoint", True
    )
    def test_compile_selective_checkpoint_tensor_subclass(self):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return _pt2_selective_checkpoint_context_fn_gen(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        rand_tensor = torch.randn(4, 4, requires_grad=True, device="cuda")

        # tensor subclasses as inputs
        x = TwoTensor(rand_tensor, rand_tensor.clone())
        y = TwoTensor(rand_tensor.clone(), rand_tensor.clone())

        fw_compiler = functools.partial(
            count_ops,
            freq=4,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # We would've expected 12 here
            # (4 matmul recompute and 4 mm ops per fwd matmul, so 4 + 2 * 4 = 12)
            # if we didn't enable selective checkpointing.
            freq=8,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @torch._dynamo.config.patch(
        "_experimental_support_context_fn_in_torch_utils_checkpoint", True
    )
    def test_compile_selective_checkpoint_custom_rule(self):
        def _get_custom_policy(meta):
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]

            def _custom_policy(mode, func, *args, **kwargs):
                mm_count_key = f"{mode}_mm_count"
                if mm_count_key not in meta:
                    meta[mm_count_key] = 0
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except second mm
                # (i.e. we will hint the partitioner to recompute second mm in backward pass)
                return func in no_recompute_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] == 2
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = {}
            return _pt2_selective_checkpoint_context_fn_gen(_get_custom_policy(meta))

        def gn(x, y):
            return torch.sigmoid(
                torch.sigmoid(torch.matmul(torch.matmul(x, y) * y, y) * y)
            )

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # Q: How do we come to this number 4?
            # A: We have 2 matmuls in the forward pass, each matmul contributes 2 `mm` ops in the backward pass,
            # so we have at least 4 `mm` ops in backward pass. It's "at least" because whether second matmul in
            # the forward pass is recomputed in the backward pass is up to the partitioner to decide.
            freq_ge=4,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @torch._dynamo.config.patch(
        "_experimental_support_context_fn_in_torch_utils_checkpoint", True
    )
    def test_compile_selective_checkpoint_partial_ctx_fn(self):
        def selective_checkpointing_context_fn(no_recompute_list):
            return _pt2_selective_checkpoint_context_fn_gen(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=functools.partial(
                    selective_checkpointing_context_fn, [torch.ops.aten.mm.default]
                ),
            )

        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # We would've expected 6 here
            # (2 matmul recompute and 2 mm ops per fwd matmul, so 2 + 2 * 2 = 6)
            # if we didn't enable selective checkpointing.
            freq=4,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @torch._dynamo.config.patch(
        "_experimental_support_context_fn_in_torch_utils_checkpoint", True
    )
    def test_compile_selective_checkpoint_outplace_op(self):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            return _pt2_selective_checkpoint_context_fn_gen(
                _get_custom_policy(no_recompute_list=no_recompute_list),
            )

        def gn(x, y):
            return torch.sigmoid(torch.selu(torch.matmul(torch.matmul(x, y), y))).relu()

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        fw_compiler = functools.partial(
            count_ops,
            freqs=[2, 1],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        bw_compiler = functools.partial(
            count_ops,
            freqs=[4, 0],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skip(
        "In-place op support in selective checkpointing + torch.compile "
        "requires TorchDispatchMode + torch.compile work to complete"
    )
    @torch._dynamo.config.patch(
        "_experimental_support_context_fn_in_torch_utils_checkpoint", True
    )
    def test_compile_selective_checkpoint_inplace_op(self):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            return _pt2_selective_checkpoint_context_fn_gen(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            return torch.sigmoid(
                torch.selu_(torch.matmul(torch.matmul(x, y), y))
            ).relu_()

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        fw_compiler = functools.partial(
            count_ops,
            freqs=[2, 1],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        bw_compiler = functools.partial(
            count_ops,
            freqs=[4, 0],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @torch._dynamo.config.patch(
        "_experimental_support_context_fn_in_torch_utils_checkpoint", True
    )
    def test_compile_selective_checkpoint_random_op(self):
        for preserve_rng_state in [True, False]:

            def selective_checkpointing_context_fn():
                no_recompute_list = [
                    torch.ops.aten.sigmoid.default,
                ]
                return _pt2_selective_checkpoint_context_fn_gen(
                    _get_custom_policy(no_recompute_list=no_recompute_list)
                )

            def gn(x):
                return torch.sigmoid(torch.dropout(torch.sigmoid(x), p=0.5, train=True))

            def fn(x):
                return torch.utils.checkpoint.checkpoint(
                    gn,
                    x,
                    use_reentrant=False,
                    # Regardless of whether `preserve_rng_state` is True or False,
                    # we will always preserve RNG state when using `torch.compile`.
                    preserve_rng_state=preserve_rng_state,
                    context_fn=selective_checkpointing_context_fn,
                )

            x = torch.randn(4, 4, requires_grad=True, device="cuda")

            fw_compiler = functools.partial(
                count_ops,
                freqs=[2, 1],
                ops=[
                    torch.ops.aten.sigmoid.default,
                    torch.ops.aten.native_dropout.default,
                ],
            )
            bw_compiler = functools.partial(
                count_ops,
                # NOTE: This unit test expects `dropout` to be recomputed (notice the count for `native_dropout` is 1).
                freqs=[0, 1],
                ops=[
                    torch.ops.aten.sigmoid.default,
                    torch.ops.aten.native_dropout.default,
                ],
            )
            backend = aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=min_cut_rematerialization_partition,
            )

            # NOTE: when `preserve_rng_state` is False, gradient will mismatch between torch.compile and eager,
            # because eager version doesn't preserve RNG state while torch.compile still does.
            # Hence when `preserve_rng_state` is False, we skip the output and gradient comparison
            # between torch.compile and eager.
            self._validate(fn, backend, x, skip_check=not preserve_rng_state)
            self._compare_orig_and_checkpointed_fns(gn, fn, x)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @torch._dynamo.config.patch(
        "_experimental_support_context_fn_in_torch_utils_checkpoint", True
    )
    def test_compile_selective_checkpoint_invalid_context(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y)) * y

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=_invalid_context_gen,
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops,
            freq=1,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            freq_ge=2,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        with self.assertRaisesRegex(
            Exception, "must generate a tuple of two `TorchDispatchMode`s"
        ):
            self._validate(fn, backend, x, y)

    @requires_cuda
    @skipIfRocm
    def test_autocast_flash_attention(self):
        def fn(primals_1, primals_2, primals_3):
            return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                primals_1, primals_2, primals_3, None, True, scale=0.17677669529663687
            )[0]

        def gn(*args):
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

        with torch.cuda.amp.autocast():
            x = torch.randn(4, 2, 16, 32, device="cuda", requires_grad=True)
            y = torch.randn(4, 2, 16, 32, device="cuda", requires_grad=True)
            z = torch.randn(4, 2, 16, 32, device="cuda", requires_grad=True)
            args = (x, y, z)

            torch.manual_seed(0)
            ref = gn(*args)

            opt_gn = torch.compile(gn)
            torch.manual_seed(0)
            res = opt_gn(*args)
            self.assertEqual(ref, res)

    @requires_cuda
    def test_error_msg(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.sin(x)
                torch._dynamo.graph_break()
                x = torch.cos(x)
                return x

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=True)

        x = torch.randn(4, 4).cuda()
        opt_fn = torch.compile(fn, fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "skip function graph_break in file"
        ):
            opt_fn(x)

    @requires_cuda
    def test_list_inputs(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, ys):
                a = torch.sin(x)
                b = torch.cos(ys[0])
                c = torch.cos(ys[1])
                return (x, [b, c])

        mod = MockModule().cuda()

        def fn(x, ys):
            return torch.utils.checkpoint.checkpoint(mod, x, ys, use_reentrant=True)

        x = torch.randn(4, 4).cuda()
        y = torch.randn(4, 4).cuda()
        z = torch.randn(4, 4).cuda()
        ref = fn(x, [y, z])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, [y, z])
        self.assertEqual(ref, res)

    @requires_cuda
    def test_pattern_matcher(self):
        # Check that the sdpa op is recomputed in the backward graph
        # tests percolate_tags

        @checkpoint_wrapper
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        def fn(query, key, value):
            # Checks that sin is not recomputed in the backward graph
            return dot_prod_attention(query.sin(), key, value)

        tensor_shape = (4, 2, 16, 32)
        dtype = torch.float16
        args1 = [
            torch.randn(tensor_shape, device="cuda", dtype=dtype, requires_grad=True),
            torch.randn(tensor_shape, device="cuda", dtype=dtype, requires_grad=True),
            torch.randn(tensor_shape, device="cuda", dtype=dtype, requires_grad=True),
        ]

        # Save the AOT graphs
        aot_graphs = []
        from torch._inductor import compile_fx

        def debug_compile_fx_inner(graph, example_inputs, *args, **kwargs):
            aot_graphs.append(graph)
            return compile_fx.compile_fx_inner(graph, example_inputs, *args, **kwargs)

        backend = functools.partial(
            compile_fx.compile_fx, inner_compile=debug_compile_fx_inner
        )

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        opt_fn(*args1).sum().backward()

        fwd_graph = aot_graphs[0]
        self.assertTrue(
            count_ops(
                fwd_graph,
                [],
                freq=1,
                op=torch.ops.aten._scaled_dot_product_flash_attention.default,
            )
        )

        bwd_graph = aot_graphs[1]
        # Check that sin is not recomputed in the backward graph - checks percolate tags
        self.assertTrue(count_ops(bwd_graph, [], freq=0, op=torch.ops.aten.sin.default))
        # Check that the sdpa op is recomputed in the backward graph
        self.assertTrue(
            count_ops(
                bwd_graph,
                [],
                freq=1,
                op=torch.ops.aten._scaled_dot_product_flash_attention.default,
            )
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
