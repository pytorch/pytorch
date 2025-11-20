# Owner(s): ["module: dynamo"]
# flake8: noqa: B950
# flake8: noqa: E731
import contextlib
import copy
import functools
import math
import unittest  # noqa: F811
from importlib import import_module

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._functorch.config
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
from functorch.compile import default_partition, min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounterWithBackend,
    normalize_gm,
)
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import IS_WINDOWS, parametrize, skipIfHpu
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


if HAS_CUDA_AND_TRITON:
    import triton
    from triton import language as tl

    @triton.jit
    def add_one_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = x + 1
        tl.store(out_ptr + offsets, output, mask=mask)


requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


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
            elif node.name == "graphsafe_run_with_rng_state":
                return node.args[0] == op
        return False

    # assert ((freq or freq_ge) and op) or ((freqs or freqs_ge) and ops)
    if op is not None:
        assert not isinstance(op, list)
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
            err_msg = f"In graph {gm}, expected {op} to have occurred {freq} times in the graph, but got {actual_count}."
            assert actual_count == freq, err_msg
    else:
        assert freqs_ge is not None
        for op, freq_ge in zip(ops, freqs_ge):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            assert actual_count >= freq_ge, (
                f"In graph {gm}, expected {op} to have occurred at least {freq_ge} times in the graph, but got {actual_count}."
            )
    return gm


def collect_fwd_graph_outputs(graph: torch.fx.Graph, *, fwd_outputs: set[str]):
    if not torch._dynamo.compiled_autograd.in_compiled_autograd_region:  # fwd graph
        return_node = list(graph.nodes)[-1]
        assert return_node.target == "output"
        for x in return_node.args[0]:
            fwd_outputs.add(str(x))


class _InvalidContext:
    def __init__(self) -> None:
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


def _get_custom_policy(no_recompute_list=None, must_recompute_list=None):
    def _custom_policy(ctx, func, *args, **kwargs):
        if no_recompute_list is not None and func in no_recompute_list:
            return CheckpointPolicy.MUST_SAVE
        if must_recompute_list is not None and func in must_recompute_list:
            return CheckpointPolicy.MUST_RECOMPUTE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE

    return _custom_policy


class ActivationCheckpointingViaTagsTests(
    torch._dynamo.test_case.TestCaseWithNestedGraphBreaks
):
    def _validate(
        self,
        fn,
        backend,
        *args,
        skip_check=False,
        fullgraph=True,
        compiled_autograd=False,
    ):
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.detach().clone().requires_grad_(arg.requires_grad))

        cloned_fn = copy.deepcopy(fn)

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        torch.manual_seed(0)
        compiled_fn = torch.compile(cloned_fn, fullgraph=fullgraph, backend=backend)
        ctx = contextlib.nullcontext()
        if compiled_autograd:
            ctx = torch._dynamo.compiled_autograd._enable(
                lambda gm: torch.compile(gm, fullgraph=fullgraph, backend=backend)
            )
        with ctx:
            result = compiled_fn(*cloned_args)
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

        def clone_args(args):
            cloned_args = []
            for arg in args:
                cloned_args.append(
                    arg.detach().clone().requires_grad_(arg.requires_grad)
                )
            return cloned_args

        def run(compiler):
            # Run original version
            cloned_args_orig_fn = clone_args(args)
            torch.manual_seed(0)
            compiled_orig_fn = compiler(orig_fn)
            result_orig_fn = compiled_orig_fn(*cloned_args_orig_fn)
            result_orig_fn.sum().backward()

            # Run checkpointed version
            cloned_args_checkpointed_fn = clone_args(args)
            torch.manual_seed(0)
            compiled_checkpointed_fn = compiler(copy.deepcopy(checkpointed_fn))
            result_checkpointed_fn = compiled_checkpointed_fn(
                *cloned_args_checkpointed_fn
            )
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

        run(functools.partial(torch.compile, fullgraph=fullgraph))
        if fullgraph:

            def export_compiler(fn):
                class WrapAsModule(nn.Module):
                    def forward(self, *args, **kwargs):
                        return fn(*args, **kwargs)

                mod = WrapAsModule()

                def runtime_wrapper(*runtime_args):
                    from torch.export import _trace

                    gm = _trace._export_to_torch_ir(
                        f=mod,
                        args=tuple(clone_args(args)),
                        kwargs={},
                        dynamic_shapes=None,
                        preserve_module_call_signature=(),
                        restore_fqn=False,
                        prefer_deferred_runtime_asserts_over_guards=False,
                        _log_export_usage=False,
                    )
                    # NOTE: this is necessary for rng to be added to the exported graph
                    return torch.compile(gm, fullgraph=False)(*runtime_args)

                return runtime_wrapper

            run(export_compiler)

    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_function(self, device, partition_fn):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)

    @requires_cuda_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_function_via_global_checkpoint(self, device, partition_fn):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # This goes through VariableBuilder
            return checkpoint(gn, torch.sin(x), y, use_reentrant=True)

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)

    @requires_cuda_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_function_with_kwargs(self, device, partition_fn):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=False
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)

    @requires_cuda_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_sequential_layers(self, device, partition_fn):
        def gn(x):
            x = x.cos()
            for _ in range(3):
                x = torch.mm(x, x)
            x = x.cos()
            return x

        def fn(x):
            x = torch.utils.checkpoint.checkpoint(gn, x)
            x = torch.utils.checkpoint.checkpoint(gn, x)
            return x

        x = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=6, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops,
            freqs=[2, 18],
            ops=[torch.ops.aten.cos.default, torch.ops.aten.mm.default],
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x)

    @requires_cuda_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_multiple_checkpoints(self, device, partition_fn):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            x = torch.sin(z)
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            return z

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=6, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)

    @requires_cuda_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_module(self, device, partition_fn):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        mod = MockModule().to(device)

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        x = torch.randn(10, 10, device=device, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x)

    @requires_cuda_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_decomps(self, device, partition_fn):
        # Ensures that tags are passed on through decompositions as well
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.nn.functional.gelu(self.linear(x))

        mod = MockModule().to(device)

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        x = torch.randn(10, 10, device=device, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
        )
        self._validate(fn, backend, x)

    @requires_cuda_and_triton
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_recomputed_rand(self, device):
        def gn(x, y):
            return torch.sigmoid(torch.rand_like(x) * y) * x

        def fn(x, y):
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            return z

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda_and_triton
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_rand(self, device):
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

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # backend = "aot_eager"
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda_and_triton
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_dropout(self, device):
        # Figure out a way to test the number of inductor_random calls
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, x):
                return self.dropout(self.linear(x))

        mod = MockModule().to(device)

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=True)

        x = torch.randn(10, 10, device=device, requires_grad=True)
        backend = "inductor"
        # rand decomps do not have have numerical results as eager
        self._validate(fn, backend, x, skip_check=True)

    @skipIfHpu
    @torch._functorch.config.patch(recompute_views=True)
    @torch._inductor.config.patch(fx_graph_cache=False)
    def test_tags_must_save_tensor_that_has_backward_hook(self):
        def my_post_forward_hook(submod, args, output):
            output.register_hook(my_backward_hook)
            return output

        def my_backward_hook(grad):
            return grad

        class MySubmod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.matmul(x, x)
                z = y * y
                return z

        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = MySubmod()
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                out = torch.utils.checkpoint.checkpoint(
                    self.submod, x, use_reentrant=False
                )
                norm_out = self.norm(out)
                return norm_out

        def _factory_fn():
            mod = MyMod()
            x = torch.ones(4, 4, dtype=torch.float32, requires_grad=True)
            backend = "inductor"
            return mod, x, backend

        mod_no_hook, x, backend = _factory_fn()
        mod_no_hook_fwd_outputs = set()

        with torch._inductor.config.patch(
            post_grad_custom_pre_pass=functools.partial(
                collect_fwd_graph_outputs, fwd_outputs=mod_no_hook_fwd_outputs
            )
        ):
            self._validate(
                mod_no_hook, backend, x, fullgraph=True, compiled_autograd=True
            )

        torch._dynamo.reset()
        mod_with_hook, x, backend = _factory_fn()
        mod_with_hook.submod.register_forward_hook(my_post_forward_hook)
        mod_with_hook_fwd_outputs = set()

        with torch._inductor.config.patch(
            post_grad_custom_pre_pass=functools.partial(
                collect_fwd_graph_outputs, fwd_outputs=mod_with_hook_fwd_outputs
            )
        ):
            self._validate(
                mod_with_hook, backend, x, fullgraph=True, compiled_autograd=True
            )

        # If `z` has a backward hook, result of `z = y * y` should also be saved in addition to the usual saved tensors.
        mod_no_hook_fwd_outputs_no_primal = {
            x for x in mod_no_hook_fwd_outputs if not x.startswith("primals_")
        }
        mod_with_hook_fwd_outputs_no_primal = {
            x for x in mod_with_hook_fwd_outputs if not x.startswith("primals_")
        }
        additional_saved_tensors = (
            mod_with_hook_fwd_outputs_no_primal - mod_no_hook_fwd_outputs_no_primal
        )
        expected_additional_saved_tensors = {"mul"}
        self.assertEqual(
            additional_saved_tensors,
            expected_additional_saved_tensors,
            f"""
Expected additional saved tensors: {expected_additional_saved_tensors} but got: {additional_saved_tensors}.
Non-primal fwd outputs from model w/ backward hook: {mod_with_hook_fwd_outputs_no_primal}.
Non-primal fwd outputs from model w/o backward hook: {mod_no_hook_fwd_outputs_no_primal}.""",
        )

    @requires_cuda_and_triton
    def test_fallback(self, device):
        def gn(x, y):
            torch._dynamo.graph_break()
            a = torch.sigmoid(torch.matmul(x, y))
            torch._dynamo.graph_break()
            return torch.cos(a)

        def fn(x, y):
            return torch.cos(checkpoint(gn, torch.sin(x), y, use_reentrant=False))

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)
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

    @requires_cuda_and_triton
    def test_kwargs(self, device):
        def gn(x, y, z=None):
            a = torch.matmul(x, y)
            if z is not None:
                return torch.matmul(a, z)
            return a

        def fn(x, y, z):
            return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)
        z = torch.randn(4, 4, requires_grad=True, device=device)
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

    @requires_cuda_and_triton
    def test_symints_location(self, device):
        def gn(x, y):
            return torch.matmul(x, torch.nn.functional.dropout(y, 0.5))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)
        opt_fn = torch.compile(fn, backend=cnt)

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)

        x = torch.randn(5, 5, requires_grad=True, device=device)
        y = torch.randn(5, 5, requires_grad=True, device=device)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(cnt.graphs), 2)
        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        self.assertEqual(len(wrap_node.args), 3)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_must_recompute(self, device, partition_fn):
        def context_fn_must_recompute_mm():
            must_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(
                    must_recompute_list=must_recompute_list,
                ),
            )

        def context_fn_no_recompute_mm():
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(
                    no_recompute_list=no_recompute_list,
                ),
            )

        def _test(context_fn, bw_compiler, partition_fn):
            def gn(x):
                return torch.cos(torch.sin(torch.matmul(x, x) @ x))

            def fn(x):
                return torch.utils.checkpoint.checkpoint(
                    gn,
                    x,
                    use_reentrant=False,
                    context_fn=context_fn,
                )

            x = torch.randn(4, 4, requires_grad=True, device=device)

            fw_compiler = functools.partial(
                count_ops,
                freq=2,
                op=torch.ops.aten.mm.default,
            )

            backend = aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=partition_fn,
            )
            self._validate(fn, backend, x)

        _test(
            context_fn=context_fn_must_recompute_mm,
            bw_compiler=functools.partial(
                count_ops,
                freq=6,  # 1 matmul recompute and 2 bwd mm ops per fwd matmul, so 2 + 2 * 2 = 6)
                op=torch.ops.aten.mm.default,
            ),
            partition_fn=partition_fn,
        )
        _test(
            context_fn=context_fn_no_recompute_mm,
            bw_compiler=functools.partial(
                count_ops,
                freq=4,  # 2 bwd mm ops per fwd matmul
                op=torch.ops.aten.mm.default,
            ),
            partition_fn=partition_fn,
        )

    def test_sac_with_partial_context_fn(self):
        class CustomPolicy:
            def __init__(self):
                super().__init__()

            def __call__(self, ctx, out, func, *args, **kwargs):
                return CheckpointPolicy.MUST_SAVE

        def f(x, y):
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        context_fn1 = functools.partial(
            create_selective_checkpoint_contexts, CustomPolicy()
        )

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                f,
                x,
                y,
                use_reentrant=False,
                context_fn=context_fn1,
            )

        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        a = torch.randn(4, 4, requires_grad=True, device="cpu")
        b = torch.randn(4, 4, requires_grad=True, device="cpu")

        expected = fn(a, b)
        result = opt_fn(a, b)
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_must_not_recompute_gemm(
        self, device, partition_fn
    ):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
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

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

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
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_must_not_recompute_gemm_no_functionalization(
        self, device, partition_fn
    ):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
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

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freq=1,
            op=torch.ops.aten.sigmoid.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # Main check here is just that sigmoid is properly recomputed
            # (we will see a sigmoid() and sigmoid_backward() in the bw graph)
            freq=1,
            op=torch.ops.aten.sigmoid.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
            disable_functionalization=True,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_triton_kernel(self, device, partition_fn):
        # Copy of the above test, but make sure that having a triton kernel in the
        # region does not error.
        def add_one(x):
            out = torch.empty_like(x)
            n_elements = x.numel()
            add_one_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)
            return out

        class AddOne(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return add_one(x)

            @staticmethod
            def backward(ctx, x):
                return x

        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            return (
                torch.sigmoid(torch.matmul(torch.matmul(AddOne.apply(x.sin()), y), y))
                * y
            )

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

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
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_tensor_subclass(self, device, partition_fn):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
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

        rand_tensor = torch.randn(4, 4, requires_grad=True, device=device)

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
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_custom_rule(self, device, partition_fn):
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
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

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

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

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
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_partial_ctx_fn(self, device, partition_fn):
        def selective_checkpointing_context_fn(no_recompute_list):
            return create_selective_checkpoint_contexts(
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

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

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
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_outplace_op(self, device, partition_fn):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            return create_selective_checkpoint_contexts(
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

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

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
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_list_ops(self, device, partition_fn):
        def selective_checkpointing_context_fn():
            # recompute everything
            no_recompute_list = []
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            return torch.cat([x, y]).sin()

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freqs=[1],
            ops=[torch.ops.aten.cat.default],
        )
        bw_compiler = functools.partial(
            count_ops,
            freqs=[1],
            ops=[torch.ops.aten.cat.default],
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skip(
        "In-place op support in selective checkpointing + torch.compile "
        "requires TorchDispatchMode + torch.compile work to complete"
    )
    @requires_cuda_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_inplace_op(self, device, partition_fn):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            return create_selective_checkpoint_contexts(
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

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

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
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @torch._inductor.config.patch(fallback_random=True)
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_random_op(self, device, partition_fn):
        for preserve_rng_state in [True, False]:

            def selective_checkpointing_context_fn():
                no_recompute_list = [
                    torch.ops.aten.sigmoid.default,
                ]
                return create_selective_checkpoint_contexts(
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

            x = torch.randn(4, 4, requires_grad=True, device=device)

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
                partition_fn=partition_fn,
            )

            # NOTE: when `preserve_rng_state` is False, gradient will mismatch between torch.compile and eager,
            # because eager version doesn't preserve RNG state while torch.compile still does.
            # Hence when `preserve_rng_state` is False, we skip the output and gradient comparison
            # between torch.compile and eager.
            self._validate(fn, backend, x, skip_check=not preserve_rng_state)
            self._compare_orig_and_checkpointed_fns(gn, fn, x)

    @requires_cuda_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_invalid_context(self, partition_fn):
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
            partition_fn=partition_fn,
        )
        with self.assertRaisesRegex(
            Exception, "must generate a tuple of two `TorchDispatchMode`s"
        ):
            self._validate(fn, backend, x, y)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_parametrization(self, partition_fn):
        def sac_policy():
            def _recomp_policy():
                def _custom_policy(ctx, func, *args, **kwargs):
                    to_recompute = func in {
                        torch.ops.aten.mul.Tensor,
                        torch.ops.aten.sigmoid.default,
                    }
                    return (
                        CheckpointPolicy.MUST_RECOMPUTE
                        if to_recompute
                        else CheckpointPolicy.MUST_SAVE
                    )

                return _custom_policy

            return create_selective_checkpoint_contexts(_recomp_policy())

        class Parametrization(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def parametrization(self, x):
                return torch.sigmoid(torch.mul(x, x))

            def forward(self, x):
                return checkpoint(
                    self.parametrization, x, use_reentrant=False, context_fn=sac_policy
                )

        def apply_parametrization(model):
            modules = list(model.modules())

            for mod in modules:
                params_dict = dict(mod.named_parameters(recurse=False))
                for p_name, p in params_dict.items():
                    mod.register_parameter(p_name, nn.Parameter(p))
                    nn.utils.parametrize.register_parametrization(
                        mod, p_name, Parametrization(), unsafe=True
                    )

            return model

        class MLPModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                torch.manual_seed(5)
                self.net1 = nn.Linear(16, 16, bias=False)

            def forward(self, x):
                return self.net1(x)

            def reset_parameters(self):
                self.net1.reset_parameters()

        fw_compiler = functools.partial(
            count_ops,
            freqs=[1, 1],
            ops=[torch.ops.aten.mul.Tensor, torch.ops.aten.sigmoid.default],
        )
        bw_compiler = functools.partial(
            count_ops,
            freqs=[
                # 1 from mul recompute, 1 from mul backward
                # w/o CSE, we have one extra mul
                3 if partition_fn is default_partition else 2,
                1,
            ],
            ops=[torch.ops.aten.mul.Tensor, torch.ops.aten.sigmoid.default],
        )

        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )

        model = MLPModule()
        model = apply_parametrization(model)
        model_compiled = torch.compile(
            copy.deepcopy(model), backend=backend, fullgraph=True
        )
        input = torch.randn(8, 16, requires_grad=True)
        input_compiled = copy.deepcopy(input)

        out = model(input)
        out.sum().backward()
        out_compiled = model_compiled(input_compiled)
        out_compiled.sum().backward()

        self.assertEqual(out, out_compiled)
        self.assertEqual(input.grad, input_compiled.grad)

    @requires_cuda_and_triton
    def test_autocast_flash_attention(self, device):
        def fn(primals_1, primals_2, primals_3):
            return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                primals_1, primals_2, primals_3, None, True, scale=0.17677669529663687
            )[0]

        def gn(*args):
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

        with torch.autocast(device_type=device):
            x = torch.randn(4, 2, 16, 32, device=device, requires_grad=True)
            y = torch.randn(4, 2, 16, 32, device=device, requires_grad=True)
            z = torch.randn(4, 2, 16, 32, device=device, requires_grad=True)
            args = (x, y, z)

            torch.manual_seed(0)
            ref = gn(*args)

            opt_gn = torch.compile(gn)
            torch.manual_seed(0)
            res = opt_gn(*args)
            self.assertEqual(ref, res)

    @requires_cuda_and_triton
    def test_error_msg(self, device):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                x = torch.sin(x)
                torch._dynamo.graph_break()
                x = torch.cos(x)
                return x

        mod = MockModule().to(device)

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=True)

        x = torch.randn(4, 4).to(device)
        opt_fn = torch.compile(fn, fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "User-inserted graph break"
        ):
            opt_fn(x)

    @requires_cuda_and_triton
    def test_list_inputs(self, device):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, ys):
                a = torch.sin(x)  # noqa: F841
                b = torch.cos(ys[0])
                c = torch.cos(ys[1])
                return (x, [b, c])

        mod = MockModule().to(device)

        def fn(x, ys):
            return torch.utils.checkpoint.checkpoint(mod, x, ys, use_reentrant=True)

        x = torch.randn(4, 4).to(device)
        y = torch.randn(4, 4).to(device)
        z = torch.randn(4, 4).to(device)
        ref = fn(x, [y, z])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, [y, z])
        self.assertEqual(ref, res)

    @requires_cuda_and_triton
    def test_pattern_matcher(self, device):
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
            torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True),
            torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True),
            torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True),
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
        op1 = torch.ops.aten._scaled_dot_product_flash_attention.default
        op2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default
        self.assertTrue(
            count_ops(
                fwd_graph,
                [],
                freq=1,
                op=op1,
            )
            or count_ops(
                fwd_graph,
                [],
                freq=1,
                op=op2,
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
                op=op1,
            )
            or count_ops(
                bwd_graph,
                [],
                freq=1,
                op=op2,
            )
        )

    @requires_distributed()
    @requires_cuda_and_triton
    def test_distributed_utils_checkpoint_wrapper(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper as dist_checkpoint_wrapper,
        )

        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.c = 2

            def forward(self, x):
                x = torch.sin(x)
                x = self.linear(x)
                x = torch.cos(x)
                return x * self.c

        mod = dist_checkpoint_wrapper(MockModule())
        x = torch.randn(4, 4)
        ref = mod(x)
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        res = opt_mod(x)
        self.assertEqual(ref, res)

    @requires_distributed()
    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    def test_dynamo_does_not_trace_getattr_as_top_frame(self):
        # inline_inbuilt_nn_modules is a proxy to emulate what FSDP tests do.
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        cnt = CompileCounterWithBackend("eager")

        lin = torch.nn.Linear(1, 1)
        mod = torch.nn.Sequential(lin, lin)
        mod = CheckpointWrapper(mod)
        mod._checkpoint_wrapped_module.a = torch.ones(1, 1)

        def fn(x):
            return mod(x) * mod.a

        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.randn(1, 1)

        self.assertEqual(opt_fn(x), fn(x))

    def test_return_same_element_twice(self):
        def gn(x):
            y = torch.sin(x)
            return y, y

        def fn(x):
            return torch.utils.checkpoint.checkpoint(gn, x, use_reentrant=True)

        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = True);  wrap_body_0 = l_x_ = None
        getitem: "f32[4, 4]" = tag_activation_checkpoint[0];  tag_activation_checkpoint = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[4, 4]"):
            y: "f32[4, 4]" = torch.sin(l_x_);  l_x_ = None
            return (y,)
""",
        )

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_nonlocal_mutation(self):
        counter = 0

        def gn(x):
            nonlocal counter
            counter += 1
            return torch.sin(x)

        def fn(x):
            return torch.utils.checkpoint.checkpoint(gn, x, use_reentrant=True)

        x = torch.randn(4, 4, requires_grad=True)
        fn(x).sum().backward()
        # The mutation is reapplied in the backward as well
        self.assertEqual(counter, 2)
        counter = 0

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        opt_fn(x).sum().backward()
        # The mutation is not reapplied in the backward because the flag was on.
        self.assertEqual(counter, 1)

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_nonlocal_list_mutation(self):
        def gn(x, z):
            out = x.sin()
            z.append(out)
            return torch.cos(torch.sin(torch.matmul(x, x) @ x)), out

        def fn(x):
            z = []

            out1, out2 = torch.utils.checkpoint.checkpoint(
                gn,
                x,
                z,
                use_reentrant=False,
            )

            return out1, z[0]

        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_nonlocal_list_mutation_hidden(self):
        def gn(x, z):
            o = torch.matmul(x, x) @ x
            out = x.sin()
            z.append(out)
            return torch.cos(torch.sin(o)), torch.sin(x)

        def fn(x):
            z = []

            outs = torch.utils.checkpoint.checkpoint(
                gn,
                x,
                z,
                use_reentrant=False,
            )
            out1 = outs[0]
            # Check that the extra output pytree handling is done properly
            out2 = outs[-1]

            return out1 + out2, z[0]

        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = False);  wrap_body_0 = l_x_ = None
        out1: "f32[4, 4]" = tag_activation_checkpoint[0]
        out2: "f32[4, 4]" = tag_activation_checkpoint[1]
        getitem_4: "f32[4, 4]" = tag_activation_checkpoint[4];  tag_activation_checkpoint = None

        add: "f32[4, 4]" = out1 + out2;  out1 = out2 = None
        return (add, getitem_4)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[4, 4]"):
            matmul: "f32[4, 4]" = torch.matmul(l_x_, l_x_)
            o: "f32[4, 4]" = matmul @ l_x_

            out: "f32[4, 4]" = l_x_.sin()

            sin_1: "f32[4, 4]" = torch.sin(o)
            cos: "f32[4, 4]" = torch.cos(sin_1)
            sin_2: "f32[4, 4]" = torch.sin(l_x_);  l_x_ = None
            return (cos, sin_2, matmul, o, out, sin_1)
""",
        )

        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 4]"):
        mm: "f32[4, 4]" = torch.ops.aten.mm.default(primals_1, primals_1)
        mm_1: "f32[4, 4]" = torch.ops.aten.mm.default(mm, primals_1);  mm = None

        sin: "f32[4, 4]" = torch.ops.aten.sin.default(primals_1)

        sin_1: "f32[4, 4]" = torch.ops.aten.sin.default(mm_1);  mm_1 = None
        cos: "f32[4, 4]" = torch.ops.aten.cos.default(sin_1);  sin_1 = None
        sin_2: "f32[4, 4]" = torch.ops.aten.sin.default(primals_1)

        add: "f32[4, 4]" = torch.ops.aten.add.Tensor(cos, sin_2);  cos = sin_2 = None
        return (add, sin, primals_1)
""",
        )


class ACReorderingTests(torch._dynamo.test_case.TestCase):
    """Tests for AC reordering optimization in full graph (forward+backward in one graph)."""

    def _get_ac_nodes(self, gm):
        """Get nodes tagged for AC recomputation."""
        from torch.utils.checkpoint import CheckpointPolicy

        ac_nodes = []
        for node in gm.graph.nodes:
            if node.meta.get("recompute") in [
                CheckpointPolicy.MUST_RECOMPUTE,
                CheckpointPolicy.PREFER_RECOMPUTE,
            ]:
                ac_nodes.append(node)
        return ac_nodes

    def _get_backward_nodes(self, gm):
        """Get nodes tagged as backward."""
        backward_nodes = []
        for node in gm.graph.nodes:
            if node.meta.get("custom", {}).get("backward") is not None:
                backward_nodes.append(node)
        return backward_nodes

    def _get_node_order(self, gm):
        """Get mapping from node to its position in graph."""
        return {node: idx for idx, node in enumerate(gm.graph.nodes)}

    def _compile_and_capture(self, fn, enable_reordering):
        """Helper to compile a function and capture the graph."""
        captured_gm = None

        def compiler(gm, example_inputs):
            nonlocal captured_gm
            captured_gm = gm
            return gm.forward

        backend = aot_autograd(
            fw_compiler=compiler,
            bw_compiler=None,
            partition_fn=None,
        )

        with torch._functorch.config.patch(
            enable_inference_mode_ac_reordering=enable_reordering
        ):
            compiled_fn = torch.compile(fn, backend=backend, fullgraph=False)
            result = compiled_fn()

        return result, captured_gm

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ac_reordering_simple_forward_backward(self):
        """AC reordering with checkpoint used in both forward and backward."""
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 4)
        y_data = torch.randn(4, 4)

        def simple_fwd_bwd():
            x = x_data.detach().requires_grad_(True)
            y = y_data.detach().requires_grad_(True)
            z = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sigmoid(torch.matmul(a, b)),
                x,
                y,
                use_reentrant=False,
            )
            loss = z.sum()

            with torch.fx.traceback.annotate({"backward": 0}):
                dx, dy = torch.autograd.grad(loss, (x, y))

            return dx.detach(), dy.detach()

        (dx1, dy1), gm_without = self._compile_and_capture(simple_fwd_bwd, False)
        (dx2, dy2), gm_with = self._compile_and_capture(simple_fwd_bwd, True)

        # Verify correctness
        self.assertTrue(torch.allclose(dx1, dx2))
        self.assertTrue(torch.allclose(dy1, dy2))

        # Verify recomputation: mm and sigmoid recomputed in backward
        self.assertExpectedInline(
            gm_with.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg0_1, arg1_1)
    sigmoid = torch.ops.aten.sigmoid.default(mm);  mm = None
    sum_1 = torch.ops.aten.sum.default(sigmoid);  sigmoid = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    mm_recomputed = torch.ops.aten.mm.default(arg0_1, arg1_1)
    sigmoid_recomputed = torch.ops.aten.sigmoid.default(mm_recomputed);  mm_recomputed = None
    detach_recomputed = torch.ops.aten.detach.default(sigmoid_recomputed);  sigmoid_recomputed = None
    detach_2 = torch.ops.aten.detach.default(detach_recomputed);  detach_recomputed = None
    sigmoid_backward = torch.ops.aten.sigmoid_backward.default(expand, detach_2);  expand = detach_2 = None
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    mm_2 = torch.ops.aten.mm.default(t, sigmoid_backward);  t = None
    t_1 = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_3 = torch.ops.aten.mm.default(sigmoid_backward, t_1);  sigmoid_backward = t_1 = None
    detach_3 = torch.ops.aten.detach.default(mm_3);  mm_3 = None
    detach_4 = torch.ops.aten.detach.default(mm_2);  mm_2 = None
    return (detach_3, detach_4)""",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ac_reordering_defers_backward_only_nodes(self):
        """AC nodes only used in backward are deferred (DCE removes forward version)."""
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 4)

        def forward_backward_with_ac():
            x = x_data.detach().requires_grad_(True)
            z = torch.utils.checkpoint.checkpoint(
                lambda a: torch.sin(a), x, use_reentrant=False
            )
            loss = z.sum()
            with torch.fx.traceback.annotate({"backward": 0}):
                dx = torch.autograd.grad(loss, x)[0]
            return dx.detach()

        dx1, gm_without = self._compile_and_capture(forward_backward_with_ac, False)
        dx2, gm_with = self._compile_and_capture(forward_backward_with_ac, True)

        self.assertExpectedInline(
            str(gm_without.code).strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1)
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None
    detach = torch.ops.aten.detach.default(mul);  mul = None
    return (detach,)""",
        )

        self.assertExpectedInline(
            str(gm_with.code).strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1)
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None
    detach = torch.ops.aten.detach.default(mul);  mul = None
    return (detach,)""",
        )

        # Verify correctness
        self.assertTrue(torch.allclose(dx1, dx2))

        # sin is used in forward (for sum), so it stays in forward
        # But DCE-based approach still works correctly
        order_with = self._get_node_order(gm_with)
        first_bwd_idx = min(order_with[n] for n in self._get_backward_nodes(gm_with))
        ac_in_fwd = sum(
            1 for ac in self._get_ac_nodes(gm_with) if order_with[ac] < first_bwd_idx
        )
        # sin is needed for forward, so it's kept (DCE doesn't remove it)
        self.assertEqual(ac_in_fwd, 1)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ac_reordering_graph_structure(self):
        """Verify graph structure with AC reordering enabled."""
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 4)
        y_data = torch.randn(4, 4)

        def simple_fwd_bwd():
            x = x_data.detach().requires_grad_(True)
            y = y_data.detach().requires_grad_(True)
            z = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.matmul(a, b), x, y, use_reentrant=False
            )
            loss = z.sum()
            with torch.fx.traceback.annotate({"backward": 0}):
                dx, dy = torch.autograd.grad(loss, (x, y))
            return dx.detach(), dy.detach()

        _, captured_gm = self._compile_and_capture(simple_fwd_bwd, True)

        # mm used in forward only (sum consumes it), DCE removes it
        self.assertExpectedInline(
            captured_gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg0_1, arg1_1)
    sum_1 = torch.ops.aten.sum.default(mm);  mm = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    mm_1 = torch.ops.aten.mm.default(t, expand);  t = None
    t_1 = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_2 = torch.ops.aten.mm.default(expand, t_1);  expand = t_1 = None
    detach = torch.ops.aten.detach.default(mm_2);  mm_2 = None
    detach_1 = torch.ops.aten.detach.default(mm_1);  mm_1 = None
    return (detach, detach_1)""",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ac_reordering_duplicates_nodes_used_in_both_regions(self):
        """AC nodes used in both forward and backward are duplicated."""
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 4)
        w_data = torch.randn(4, 4)

        def fwd_bwd_with_ac_in_both_regions():
            x = x_data.detach().requires_grad_(True)
            w = w_data.detach().requires_grad_(True)

            h = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.relu(torch.matmul(a, b)),
                x,
                w,
                use_reentrant=False,
            )
            out = h * 2.0  # h used in forward
            loss = out.sum()

            with torch.fx.traceback.annotate({"backward": 0}):
                dx, dw = torch.autograd.grad(loss, (x, w))  # relu needs h

            return out.detach(), dx.detach(), dw.detach()

        _, captured_gm = self._compile_and_capture(
            fwd_bwd_with_ac_in_both_regions, True
        )

        # mm and relu used in forward, duplicated for backward (relu_backward needs relu output)
        self.assertExpectedInline(
            captured_gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg0_1, arg1_1)
    relu = torch.ops.aten.relu.default(mm);  mm = None
    mul = torch.ops.aten.mul.Tensor(relu, 2.0);  relu = None
    sum_1 = torch.ops.aten.sum.default(mul)
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, 2.0);  expand = None
    mm_recomputed = torch.ops.aten.mm.default(arg0_1, arg1_1)
    relu_recomputed = torch.ops.aten.relu.default(mm_recomputed);  mm_recomputed = None
    detach_recomputed = torch.ops.aten.detach.default(relu_recomputed);  relu_recomputed = None
    detach_2 = torch.ops.aten.detach.default(detach_recomputed);  detach_recomputed = None
    threshold_backward = torch.ops.aten.threshold_backward.default(mul_1, detach_2, 0);  mul_1 = detach_2 = None
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    mm_2 = torch.ops.aten.mm.default(t, threshold_backward);  t = None
    t_1 = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_3 = torch.ops.aten.mm.default(threshold_backward, t_1);  threshold_backward = t_1 = None
    detach_3 = torch.ops.aten.detach.default(mul);  mul = None
    detach_4 = torch.ops.aten.detach.default(mm_3);  mm_3 = None
    detach_5 = torch.ops.aten.detach.default(mm_2);  mm_2 = None
    return (detach_3, detach_4, detach_5)""",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ac_reordering_recomputes_checkpointed_ops(self):
        """Verify AC nodes are recomputed in backward (not just deferred)."""
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 4)
        y_data = torch.randn(4, 4)

        def fwd_bwd_with_checkpoint():
            x = x_data.detach().requires_grad_(True)
            y = y_data.detach().requires_grad_(True)
            z = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sigmoid(torch.matmul(a, b)),
                x,
                y,
                use_reentrant=False,
            )
            loss = z.sum()
            with torch.fx.traceback.annotate({"backward": 0}):
                dx, dy = torch.autograd.grad(loss, (x, y))
            return dx.detach(), dy.detach()

        _, gm_with = self._compile_and_capture(fwd_bwd_with_checkpoint, True)
        _, gm_without = self._compile_and_capture(fwd_bwd_with_checkpoint, False)

        # Count recomputed ops: with reordering has extra mm and sigmoid for recomputation
        mm_with = sum(
            1 for n in gm_with.graph.nodes if n.target == torch.ops.aten.mm.default
        )
        mm_without = sum(
            1 for n in gm_without.graph.nodes if n.target == torch.ops.aten.mm.default
        )
        sigmoid_with = sum(
            1 for n in gm_with.graph.nodes if n.target == torch.ops.aten.sigmoid.default
        )
        sigmoid_without = sum(
            1
            for n in gm_without.graph.nodes
            if n.target == torch.ops.aten.sigmoid.default
        )

        # With reordering: 4 mm (1 fwd + 2 bwd grad + 1 recompute), 2 sigmoid (1 fwd + 1 recompute)
        # Without: 3 mm (1 fwd + 2 bwd grad), 1 sigmoid (1 fwd, saved)
        self.assertEqual(mm_with, 4, "mm should be recomputed in backward")
        self.assertEqual(mm_without, 3)
        self.assertEqual(sigmoid_with, 2, "sigmoid should be recomputed in backward")
        self.assertEqual(sigmoid_without, 1)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ac_reordering_chain_not_needed_for_forward(self):
        """AC chain not needed for forward output is fully deferred."""
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 4, device="cuda", requires_grad=False)
        y_data = torch.randn(4, 4, device="cuda", requires_grad=False)

        def fwd_bwd_with_ac_chain():
            x = x_data.detach().requires_grad_(True)
            y = y_data.detach().requires_grad_(True)

            # AC chain: both checkpointed, neither used in forward output
            a = torch.utils.checkpoint.checkpoint(
                lambda t: t * 2.0, x, use_reentrant=False
            )
            b = torch.utils.checkpoint.checkpoint(
                lambda t: t + 1.0, a, use_reentrant=False
            )
            z = (x + y).sum()  # doesn't use a or b

            with torch.fx.traceback.annotate({"backward": 0}):
                grad_x = torch.autograd.grad(z, x, create_graph=True)[0]
                loss = (grad_x * b).sum()  # b used only in backward
                dx = torch.autograd.grad(loss, x)[0]
            return dx.detach()

        result_with, gm_with = self._compile_and_capture(fwd_bwd_with_ac_chain, True)
        result_without, _ = self._compile_and_capture(fwd_bwd_with_ac_chain, False)

        # Verify correctness
        torch.testing.assert_close(result_with, result_without)

        # Both a and b should be deferred (DCE removes from forward)
        order = self._get_node_order(gm_with)
        first_bwd_idx = min(order[n] for n in self._get_backward_nodes(gm_with))
        ac_in_fwd = sum(
            1 for ac in self._get_ac_nodes(gm_with) if order[ac] < first_bwd_idx
        )
        self.assertEqual(ac_in_fwd, 0, "AC chain should be fully deferred")

    def test_ac_reordering_with_rng_ops_raises_error(self):
        """Verify error is raised when RNG ops are in checkpointed regions."""
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 4)

        def fwd_bwd_with_rng():
            x = x_data.detach().requires_grad_(True)

            # Checkpoint with RNG op (rand_like)
            z = torch.utils.checkpoint.checkpoint(
                lambda a: torch.sigmoid(a + torch.rand_like(a)), x, use_reentrant=False
            )
            loss = z.sum()

            with torch.fx.traceback.annotate({"backward": 0}):
                dx = torch.autograd.grad(loss, x)[0]

            return dx

        # Should raise error about RNG ops not being supported
        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Activation checkpoint reordering in fullgraph does not support RNG ops",
        ):
            self._compile_and_capture(fwd_bwd_with_rng, True)

    def test_ac_reordering_with_tuple_output(self):
        """Verify AC reordering handles checkpoints with tuple outputs (getitem ops)."""
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 6)
        y_data = torch.randn(6, 4)

        def fwd_bwd_with_tuple():
            x = x_data.detach().requires_grad_(True)
            y = y_data.detach().requires_grad_(True)

            # Checkpoint uses split (multi-output op) - sigmoid output saved for backward
            def checkpoint_fn(a, b):
                linear = torch.mm(a, b)
                chunks = torch.split(linear, [2, 2], dim=1)
                return torch.sigmoid(chunks[0]), chunks[1]

            result = torch.utils.checkpoint.checkpoint(
                checkpoint_fn, x, y, use_reentrant=False
            )
            sig_out = result[0]
            linear_out = result[1]
            loss = sig_out.sum() + linear_out.sum()

            with torch.fx.traceback.annotate({"backward": 0}):
                dx, dy = torch.autograd.grad(loss, (x, y))
            return dx, dy

        result_with, gm_with = self._compile_and_capture(fwd_bwd_with_tuple, True)
        result_without, gm_without = self._compile_and_capture(
            fwd_bwd_with_tuple, False
        )

        # Verify correctness
        torch.testing.assert_close(result_with[0], result_without[0])
        torch.testing.assert_close(result_with[1], result_without[1])

        # WITHOUT reordering: checkpointed ops saved via detach (no recomputation)
        self.assertExpectedInline(
            str(gm_without.graph).strip(),
            """\
graph():
    %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=2] = placeholder[target=arg1_1]
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg0_1, %arg1_1), kwargs = {})
    %split_with_sizes : [num_users=2] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%mm, [2, 2], 1), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%split_with_sizes, 0), kwargs = {})
    %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%split_with_sizes, 1), kwargs = {})
    %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem,), kwargs = {})
    %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%sigmoid,), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sigmoid,), kwargs = {})
    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%getitem_1,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %sum_2), kwargs = {})
    %ones_like : [num_users=2] = call_function[target=torch.ops.aten.ones_like.default](args = (%add,), kwargs = {pin_memory: False, memory_format: torch.preserve_format})
    %expand : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%ones_like, [4, 2]), kwargs = {})
    %expand_1 : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%ones_like, [4, 2]), kwargs = {})
    %detach_1 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%detach,), kwargs = {})
    %sigmoid_backward : [num_users=1] = call_function[target=torch.ops.aten.sigmoid_backward.default](args = (%expand_1, %detach_1), kwargs = {})
    %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sigmoid_backward, %expand], 1), kwargs = {})
    %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%arg0_1,), kwargs = {})
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%t, %cat), kwargs = {})
    %t_1 : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%arg1_1,), kwargs = {})
    %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%cat, %t_1), kwargs = {})
    return (mm_2, mm_1)""",  # noqa: B950
        )

        # WITH reordering: checkpointed ops recomputed in backward (_recomputed suffix)
        # Key differences: mm/split/sigmoid/getitem appear TWICE (forward + recomputed)
        self.assertExpectedInline(
            str(gm_with.graph).strip(),
            """\
graph():
    %arg0_1 : [num_users=3] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=3] = placeholder[target=arg1_1]
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg0_1, %arg1_1), kwargs = {})
    %split_with_sizes : [num_users=2] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%mm, [2, 2], 1), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%split_with_sizes, 0), kwargs = {})
    %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%split_with_sizes, 1), kwargs = {})
    %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem,), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sigmoid,), kwargs = {})
    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%getitem_1,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %sum_2), kwargs = {})
    %ones_like : [num_users=2] = call_function[target=torch.ops.aten.ones_like.default](args = (%add,), kwargs = {pin_memory: False, memory_format: torch.preserve_format})
    %expand : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%ones_like, [4, 2]), kwargs = {})
    %expand_1 : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%ones_like, [4, 2]), kwargs = {})
    %mm_recomputed : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg0_1, %arg1_1), kwargs = {})
    %split_with_sizes_recomputed : [num_users=1] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%mm_recomputed, [2, 2], 1), kwargs = {})
    %getitem_recomputed : [num_users=1] = call_function[target=operator.getitem](args = (%split_with_sizes_recomputed, 0), kwargs = {})
    %sigmoid_recomputed : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem_recomputed,), kwargs = {})
    %detach_recomputed : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%sigmoid_recomputed,), kwargs = {})
    %detach_2 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%detach_recomputed,), kwargs = {})
    %sigmoid_backward : [num_users=1] = call_function[target=torch.ops.aten.sigmoid_backward.default](args = (%expand_1, %detach_2), kwargs = {})
    %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sigmoid_backward, %expand], 1), kwargs = {})
    %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%arg0_1,), kwargs = {})
    %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%t, %cat), kwargs = {})
    %t_1 : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%arg1_1,), kwargs = {})
    %mm_3 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%cat, %t_1), kwargs = {})
    return (mm_3, mm_2)""",  # noqa: B950
        )

    def test_ac_reordering_with_selective_checkpoint_policy(self):
        """Verify AC reordering respects MUST_SAVE policy (addmm saved, relu recomputed)."""
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 128)
        weight1 = torch.randn(128, 128)
        bias1 = torch.randn(128)

        # Policy: MUST_SAVE addmm, PREFER_RECOMPUTE everything else
        def policy_fn(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.addmm.default:
                return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

        context_fn = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
        )

        def fwd_bwd_with_policy():
            x = x_data.detach().requires_grad_(True)
            w1 = weight1.detach().requires_grad_(True)
            b1 = bias1.detach().requires_grad_(True)

            def checkpoint_fn(inp, w, b):
                # addmm will be MUST_SAVE (not recomputed)
                # relu will be PREFER_RECOMPUTE (recomputed)
                linear = torch.nn.functional.linear(inp, w, b)  # addmm
                return torch.relu(linear)

            result = torch.utils.checkpoint.checkpoint(
                checkpoint_fn, x, w1, b1, use_reentrant=False, context_fn=context_fn
            )
            loss = result.sum()

            with torch.fx.traceback.annotate({"backward": 0}):
                dx, dw, db = torch.autograd.grad(loss, (x, w1, b1))
            return dx, dw, db

        result_with, gm_with = self._compile_and_capture(fwd_bwd_with_policy, True)
        result_without, gm_without = self._compile_and_capture(
            fwd_bwd_with_policy, False
        )

        # Verify correctness
        torch.testing.assert_close(result_with[0], result_without[0])
        torch.testing.assert_close(result_with[1], result_without[1])
        torch.testing.assert_close(result_with[2], result_without[2])

        # Key insight: WITH reordering, addmm is MUST_SAVE (not recomputed)
        # but relu is PREFER_RECOMPUTE (recomputed in backward)
        def count_op(gm, target):
            return sum(1 for n in gm.graph.nodes if n.target == target)

        # WITHOUT: addmm saved, relu saved
        addmm_without = count_op(gm_without, torch.ops.aten.addmm.default)
        relu_without = count_op(gm_without, torch.ops.aten.relu.default)

        # WITH: addmm still saved (MUST_SAVE), relu recomputed (PREFER_RECOMPUTE)
        addmm_with = count_op(gm_with, torch.ops.aten.addmm.default)
        relu_with = count_op(gm_with, torch.ops.aten.relu.default)

        # addmm should NOT be recomputed (same count in both)
        self.assertEqual(addmm_without, addmm_with)

        # relu SHOULD be recomputed (more in WITH due to 1 fwd + 1 recomputed)
        self.assertGreater(relu_with, relu_without)

        # Verify no addmm_recomputed node exists
        recomputed_nodes = [
            n.name for n in gm_with.graph.nodes if "_recomputed" in n.name
        ]
        self.assertNotIn("addmm_recomputed", recomputed_nodes)

        # Verify relu_recomputed exists
        self.assertTrue(
            any("relu" in name for name in recomputed_nodes),
            f"Expected relu_recomputed but got: {recomputed_nodes}",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ac_reordering_transitive_dependency_sorting(self):
        """Verify transitive dependencies are sorted correctly in forward order.

        This tests the fix for the bug where processing inputs one at a time
        could violate forward topological order. With inputs a, b, c (where c=a+b),
        and d, if backward node processes [d, c], we must get order a, b, c, d
        (not d, a, b, c).
        """
        torch._dynamo.allow_in_graph(torch.autograd.grad)

        x_data = torch.randn(4, 4)
        y_data = torch.randn(4, 4)
        z_data = torch.randn(4, 4)

        def fwd_bwd_with_transitive_deps():
            x = x_data.detach().requires_grad_(True)
            y = y_data.detach().requires_grad_(True)
            z = z_data.detach().requires_grad_(True)

            # Forward region - all checkpointed
            a = torch.utils.checkpoint.checkpoint(
                lambda t: t.clone(), x, use_reentrant=False
            )
            b = torch.utils.checkpoint.checkpoint(
                lambda t: t.clone(), y, use_reentrant=False
            )
            c = torch.utils.checkpoint.checkpoint(
                lambda t1, t2: t1 + t2, a, b, use_reentrant=False
            )
            d = torch.utils.checkpoint.checkpoint(
                lambda t: t.clone(), z, use_reentrant=False
            )

            # Backward region starts here
            # Using d + c (instead of c + d) to trigger the bug where
            # all_input_nodes returns [d, c] in the wrong order
            with torch.fx.traceback.annotate({"backward": 0}):
                e = d + c
                loss = e.sum()
                dx = torch.autograd.grad(loss, x)[0]

            return dx.detach()

        _, captured_gm = self._compile_and_capture(fwd_bwd_with_transitive_deps, True)

        # Verify correct forward topological order: a, b, c, d (not d, a, b, c)
        self.assertExpectedInline(
            captured_gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    clone_1 = torch.ops.aten.clone.default(arg1_1);  arg1_1 = None
    add_recomputed = torch.ops.aten.add.Tensor(clone, clone_1);  clone = clone_1 = None
    clone_2_recomputed = torch.ops.aten.clone.default(arg2_1);  arg2_1 = None
    add_2 = torch.ops.aten.add.Tensor(clone_2_recomputed, add_recomputed);  clone_2_recomputed = add_recomputed = None
    sum_1 = torch.ops.aten.sum.default(add_2);  add_2 = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    detach = torch.ops.aten.detach.default(expand);  expand = None
    return (detach,)""",
        )


devices = ["cuda", "hpu"]
instantiate_device_type_tests(
    ActivationCheckpointingViaTagsTests, globals(), only_for=devices
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
