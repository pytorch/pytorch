# Owner(s): ["module: dynamo"]
import contextlib
import copy
import functools
import math
import operator
import re
import unittest
from importlib import import_module
from types import SimpleNamespace

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._functorch.config
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from functorch.compile import (
    default_partition,
    min_cut_rematerialization_partition,
    nop,
)
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.source import ConstantSource
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounterWithBackend,
    normalize_gm,
)
from torch._functorch.partitioners import has_recomputable_rng_ops, is_rng_op
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    skipIfHpu,
    skipIfXpu,
    TEST_CUDA,
)
from torch.testing._internal.inductor_utils import HAS_GPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


if HAS_GPU_AND_TRITON:
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

    @triton.jit
    def times_two_and_three_kernel(
        in_ptr0,
        out_ptr0,
        out_ptr1,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        tl.store(out_ptr0 + offsets, x * 2, mask=mask)
        tl.store(out_ptr1 + offsets, x * 3, mask=mask)


requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


def _grad(*args, **kwargs):
    return torch.autograd.grad(*args, **kwargs)


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
        if isinstance(op, list):
            raise AssertionError("Expected op to not be a list")
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
            if actual_count != freq:
                raise AssertionError(err_msg)
    else:
        if freqs_ge is None:
            raise AssertionError("Expected freqs_ge to not be None")
        for op, freq_ge in zip(ops, freqs_ge):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            if actual_count < freq_ge:
                raise AssertionError(
                    f"In graph {gm}, expected {op} to have occurred at least {freq_ge} times in the graph, but got {actual_count}."
                )
    return gm


def collect_fwd_graph_outputs(graph: torch.fx.Graph, *, fwd_outputs: set[str]):
    if not torch._dynamo.compiled_autograd.in_compiled_autograd_region:  # fwd graph
        return_node = list(graph.nodes)[-1]
        if return_node.target != "output":
            raise AssertionError(
                f"Expected return_node.target to be 'output', got {return_node.target}"
            )
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


def count_shared_recompute_calls(gm, args, expected):
    """Assert gm calls ``expected`` shared checkpoint-recompute subgraphs, each
    reading all of its operands."""
    from torch._higher_order_ops import invoke_subgraph

    calls = [
        n
        for n in gm.graph.find_nodes(op="call_function", target=invoke_subgraph)
        if n.args[1].startswith("recompute_")
    ]
    if len(calls) != expected:
        raise AssertionError(f"expected {expected} shared recompute calls in {gm}")
    for name, mod in gm.named_children():
        if not name.startswith("recompute_"):
            continue
        # (A SymInt input binds its symbol.)
        unread = [
            p.name
            for p in mod.graph.find_nodes(op="placeholder")
            if not p.users and not isinstance(p.meta.get("val"), torch.SymInt)
        ]
        if unread:
            raise AssertionError(f"{name} doesn't read {unread} in {gm}")
    return gm


save_mm_context_fn = functools.partial(
    create_selective_checkpoint_contexts, [torch.ops.aten.mm.default]
)


def shared_recompute_code(code, side):
    """Map piece name (e.g. recompute_0, prefixed inside a nested_compile_region's
    body) -> (function body, sources of the kernels it launches) for each shared
    checkpoint-recompute subgraph in Inductor output ``code``, where ``side`` is
    "fw" or "bw". The subgraph name, the kernel names (which number kernels
    across the whole graph), provenance comments, the ``optimize_mem``
    autotuning hint (which the forward and backward compiles set differently)
    and the kernel's cache path (a hash of its source, hint included) are
    normalized away so both sides compare equal."""
    shared = {}
    for name in re.findall(rf"def (\w*?recompute_\d+_{side}_mod)\(args\):", code):
        # The function's body: its indented (or blank) lines.
        match = re.search(rf"def {name}\(args\):\n((?:(?: +[^\n]*)?\n)+)", code)
        if match is None:
            raise AssertionError(f"no body for {name} in:\n{code}")
        body = match.group(1).rstrip().replace(name, "shared")
        # (Provenance comments name the call node, which differs between the
        # graphs inside a nested_compile_region's body.)
        body = re.sub(
            r"\n *# (Topologically Sorted|Unsorted) Source Nodes: [^\n]*", "", body
        )
        kernels = list(dict.fromkeys(re.findall(r"(triton_\w+)\.run\(", body)))
        sources = []
        for j, k in enumerate(kernels):
            d = code.index(f"\ndef {k}(")
            end = re.compile(r"\n(?=\S)").search(code, d + 1).start()
            source = code[code.rindex("\n@triton_heuristics.", 0, d) : end]
            source = re.sub(r"\n *filename=[^\n]*", "", source)
            source = re.sub(r"'optimize_mem': (True|False),? ?", "", source)
            sources.append(re.sub(rf"\b{k}\b", f"kernel{j}", source))
            body = re.sub(rf"\b{k}\b", f"kernel{j}", body)
        # (Inductor prefixes a region body's subgraphs with the side, too.)
        shared[re.sub(rf"partitioned_{side}_|_{side}_mod$", "", name)] = (body, sources)
    return shared


class ActivationCheckpointingViaTagsTests(torch._dynamo.test_case.TestCase):
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
                    return torch.compile(
                        gm, fullgraph=False, backend="aot_eager_decomp_partition"
                    )(*runtime_args)

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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
    def test_checkpoint_shows_tags_in_tlparse(self, device):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        def partition_fn(joint_gm, *args, **kwargs):
            gm_str = joint_gm.print_readable(print_output=False)
            # Check for the pattern with any graph ID (the ID depends on test order)
            self.assertTrue(
                re.search(r"# ac_graph_id: \d+ - PREFER_RECOMPUTE", gm_str),
                lambda msg: f"{msg}\nExpected ac_graph_id pattern not found in:\n{gm_str}",
            )
            return min_cut_rematerialization_partition(joint_gm, *args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop, bw_compiler=nop, partition_fn=partition_fn
        )
        _ = torch.compile(fn, backend=backend)(x, y)

    @requires_gpu_and_triton
    def test_ac_tags_through_custom_autograd_function(self, device):
        class MyMM(torch.autograd.Function):
            @staticmethod
            def forward(x, w):
                return x @ w

            @staticmethod
            def setup_context(ctx, inputs, output):
                x, w = inputs
                ctx.save_for_backward(x, w)

            @staticmethod
            def backward(ctx, grad):
                x, w = ctx.saved_tensors
                return grad @ w.t(), x.t() @ grad

        def gn(x, w):
            return MyMM.apply(x, w)

        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)

        x = torch.randn(4, 4, device=device, requires_grad=True)
        w = torch.randn(4, 4, device=device, requires_grad=True)

        def partition_fn(joint_gm, *args, **kwargs):
            fwd_mm_nodes = [
                node
                for node in joint_gm.graph.nodes
                if node.op == "call_function"
                and node.target == torch.ops.aten.mm.default
                and node.meta.get("partitioner_tag") == "is_forward"
            ]
            self.assertTrue(
                fwd_mm_nodes, "Expected forward mm nodes in the joint graph"
            )
            for node in fwd_mm_nodes:
                self.assertIn("recompute", node.meta)
                self.assertIn("ac_graph_id", node.meta)
            return min_cut_rematerialization_partition(joint_gm, *args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop, bw_compiler=nop, partition_fn=partition_fn
        )
        out = torch.compile(fn, backend=backend)(x, w)
        out.sum().backward()

    @requires_gpu_and_triton
    def test_sac_tags_through_custom_autograd_function(self, device):
        class MyMM(torch.autograd.Function):
            @staticmethod
            def forward(x, w):
                return x @ w

            @staticmethod
            def setup_context(ctx, inputs, output):
                x, w = inputs
                ctx.save_for_backward(x, w)

            @staticmethod
            def backward(ctx, grad):
                x, w = ctx.saved_tensors
                return grad @ w.t(), x.t() @ grad

        def gn(x, w):
            return MyMM.apply(x, w)

        context_fn = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts,
            lambda ctx,
            op,
            *args,
            **kwargs: torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE,
        )

        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(
                gn, x, w, use_reentrant=False, context_fn=context_fn
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        w = torch.randn(4, 4, device=device, requires_grad=True)

        def partition_fn(joint_gm, *args, **kwargs):
            fwd_mm_nodes = [
                node
                for node in joint_gm.graph.nodes
                if node.op == "call_function"
                and node.target == torch.ops.aten.mm.default
                and node.meta.get("partitioner_tag") == "is_forward"
            ]
            self.assertTrue(
                fwd_mm_nodes, "Expected forward mm nodes in the joint graph"
            )
            for node in fwd_mm_nodes:
                self.assertIn("recompute", node.meta)
                self.assertIn("ac_graph_id", node.meta)
            return min_cut_rematerialization_partition(joint_gm, *args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop, bw_compiler=nop, partition_fn=partition_fn
        )
        out = torch.compile(fn, backend=backend)(x, w)
        out.sum().backward()

    @requires_gpu_and_triton
    def test_tangent_placeholders_have_is_backward_tag(self, device):
        """Test that tangent placeholders in the joint graph are tagged with is_backward."""

        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=False
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        def partition_fn(joint_gm, *args, **kwargs):
            # Check partitioner_tag on placeholder nodes
            for node in joint_gm.graph.nodes:
                if node.op == "placeholder":
                    if "tangents" in str(node.target):
                        self.assertTrue(
                            "is_backward" in node.meta.get("partitioner_tag", "")
                        )
                    else:
                        self.assertTrue(
                            "is_forward" in node.meta.get("partitioner_tag", "")
                        )
            return min_cut_rematerialization_partition(joint_gm, *args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop, bw_compiler=nop, partition_fn=partition_fn
        )
        _ = torch.compile(fn, backend=backend)(x, y)

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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
            lambda msg: f"""{msg}\n
Expected additional saved tensors: {expected_additional_saved_tensors} but got: {additional_saved_tensors}.
Non-primal fwd outputs from model w/ backward hook: {mod_with_hook_fwd_outputs_no_primal}.
Non-primal fwd outputs from model w/o backward hook: {mod_no_hook_fwd_outputs_no_primal}.""",
        )

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @parametrize(
        "policy,expected_calls",
        [
            (CheckpointPolicy.PREFER_RECOMPUTE, 1),
            (CheckpointPolicy.MUST_CPU_OFFLOAD, 1),
            (CheckpointPolicy.MUST_RECOMPUTE, None),
        ],
    )
    def test_selective_checkpoint_preserves_registered_effect(
        self, device, policy, expected_calls
    ):
        call_count = 0
        with torch.library._scoped_library("test_compile_sac_effect", "FRAGMENT"):

            @torch.library.custom_op(
                "test_compile_sac_effect::identity", mutates_args=()
            )
            def effectful_identity(x: torch.Tensor) -> torch.Tensor:
                nonlocal call_count
                call_count += 1
                return x.clone()

            @effectful_identity.register_fake
            def _(x):
                return torch.empty_like(x)

            def backward(_ctx, grad_output):
                return grad_output

            effectful_identity.register_autograd(backward)
            effectful_identity.register_effect(torch.library.EffectType.ORDERED)

            def context_fn():
                return create_selective_checkpoint_contexts(
                    lambda _ctx, _op, *args, **kwargs: policy
                )

            def fn(x):
                return checkpoint(
                    lambda value: effectful_identity(value).sin(),
                    x,
                    use_reentrant=False,
                    context_fn=context_fn,
                )

            x = torch.randn(3, device=device, requires_grad=True)
            run_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
            if expected_calls is None:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "does not support MUST_RECOMPUTE for effectful operations",
                ):
                    run_fn(x).sum().backward()
                return
            run_fn(x).sum().backward()

            self.assertEqual(call_count, expected_calls)
            self.assertEqual(x.grad, x.cos())

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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
    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @requires_gpu_and_triton
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

    @skipIfXpu(msg="XPU has some known limitations on flash attention.")
    @requires_gpu_and_triton
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

            opt_gn = torch.compile(gn, backend="aot_eager_decomp_partition")
            torch.manual_seed(0)
            res = opt_gn(*args)
            self.assertEqual(ref, res)

    @requires_gpu_and_triton
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "This platform doesn't support efficient attention",
    )
    def test_compile_checkpoint_mem_eff_attention_no_dropout(self, device):
        def eager_attn(x):
            return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                x, x, x, None, True, dropout_p=0.0
            )[0]

        @torch.compile(mode="reduce-overhead")  # noqa: UNSPECIFIED_BACKEND
        def attn(x):
            return eager_attn(x)

        def block(x):
            return x + attn(x)

        x_ref = torch.randn(3, 1, 77, 64, device=device, requires_grad=True)
        _, _, philox_seed, philox_offset = (
            torch.ops.aten._scaled_dot_product_efficient_attention.default(
                x_ref, x_ref, x_ref, None, True, dropout_p=0.0
            )
        )
        self.assertEqual(philox_seed, torch.zeros((), device=device, dtype=torch.int64))
        self.assertEqual(
            philox_offset, torch.zeros((), device=device, dtype=torch.int64)
        )

        ref = x_ref + eager_attn(x_ref)
        ref.sum().backward()

        x = x_ref.detach().clone().requires_grad_(True)
        y = checkpoint(block, x, use_reentrant=False)
        y.sum().backward()

        self.assertEqual(ref, y)
        self.assertEqual(x_ref.grad, x.grad)

    @requires_gpu_and_triton
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
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager_decomp_partition")
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "User-inserted graph break"
        ):
            opt_fn(x)

    @requires_gpu_and_triton
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
        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        res = opt_fn(x, [y, z])
        self.assertEqual(ref, res)

    @requires_gpu_and_triton
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
        from torch._inductor.utils import fresh_cache

        def debug_compile_fx_inner(graph, example_inputs, *args, **kwargs):
            aot_graphs.append(graph)
            return compile_fx.compile_fx_inner(graph, example_inputs, *args, **kwargs)

        backend = functools.partial(
            compile_fx.compile_fx, inner_compile=debug_compile_fx_inner
        )

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        # A warm FX-graph cache short-circuits inner_compile, leaving aot_graphs
        # empty; use a fresh cache so the hook always runs.
        with fresh_cache():
            opt_fn(*args1).sum().backward()

        fwd_graph = aot_graphs[0]
        # Determine which fused attention backend is expected based on the
        # prioritization logic in sdp_utils.cpp:check_prefer_cudnn_attention.
        dprops = torch.get_device_module(device_type).get_device_properties(device)
        cudnn_version = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 0
        )
        prefer_cudnn = cudnn_version > 91500 and dprops.major in (9, 10)
        if prefer_cudnn and torch.version.cuda and TEST_CUDA:
            sdpa_op = torch.ops.aten._scaled_dot_product_cudnn_attention.default
        else:
            sdpa_op = torch.ops.aten._scaled_dot_product_flash_attention.default
        self.assertTrue(count_ops(fwd_graph, [], freq=1, op=sdpa_op))
        bwd_graph = aot_graphs[1]
        # Check that sin is not recomputed in the backward graph - checks percolate tags
        self.assertTrue(count_ops(bwd_graph, [], freq=0, op=torch.ops.aten.sin.default))
        # Check that the sdpa op is recomputed in the backward graph
        self.assertTrue(count_ops(bwd_graph, [], freq=1, op=sdpa_op))

    @requires_distributed()
    @requires_gpu_and_triton
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
        opt_mod = torch.compile(
            mod, backend="aot_eager_decomp_partition", fullgraph=True
        )
        res = opt_mod(x)
        self.assertEqual(ref, res)

    @requires_distributed()
    @requires_gpu_and_triton
    def test_dynamo_does_not_trace_getattr_as_top_frame(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        cnt = CompileCounterWithBackend("aot_eager_decomp_partition")

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

        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
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

        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
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
        getitem_3: "f32[4, 4]" = tag_activation_checkpoint[0]
        getitem_4: "f32[4, 4]" = tag_activation_checkpoint[1]

        add: "f32[4, 4]" = getitem_3 + getitem_4;  getitem_3 = getitem_4 = None

        getitem_5: "f32[4, 4]" = tag_activation_checkpoint[2];  tag_activation_checkpoint = None
        return (add, getitem_5)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[4, 4]"):
            matmul: "f32[4, 4]" = torch.matmul(l_x_, l_x_)
            o: "f32[4, 4]" = matmul @ l_x_;  matmul = None

            out: "f32[4, 4]" = l_x_.sin()

            sin_1: "f32[4, 4]" = torch.sin(o);  o = None
            cos: "f32[4, 4]" = torch.cos(sin_1);  sin_1 = None
            sin_2: "f32[4, 4]" = torch.sin(l_x_);  l_x_ = None
            return (cos, sin_2, out)
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

    def test_frozen_dataclass_pytree_output(self):
        import dataclasses

        from torch.utils import _pytree as pytree

        @dataclasses.dataclass(frozen=True)
        class InputNode:
            x: torch.Tensor

        @dataclasses.dataclass(frozen=True)
        class OutputNode:
            y: torch.Tensor

        pytree.register_dataclass(InputNode)
        pytree.register_dataclass(OutputNode)

        def cleanup():
            # Clean up pytree registrations to avoid leaking state to other tests.
            for cls in [InputNode, OutputNode]:
                pytree._deregister_pytree_node(cls)

        try:

            class TinyMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(4, 8)
                    self.fc2 = nn.Linear(8, 4)

                def forward(self, inp: InputNode) -> OutputNode:
                    h = self.fc1(inp.x)
                    h = torch.nn.functional.silu(h)
                    y = self.fc2(h)
                    return OutputNode(y=y)

            mlp = TinyMLP()

            def checkpointed_forward(inp):
                return torch.utils.checkpoint.checkpoint(
                    mlp.forward,
                    inp,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )

            input_eager = InputNode(x=torch.randn(2, 4, requires_grad=True))
            torch.manual_seed(0)
            output_eager = checkpointed_forward(input_eager)
            output_eager.y.sum().backward()

            input_compiled = InputNode(
                x=input_eager.x.detach().clone().requires_grad_(True)
            )
            torch.manual_seed(0)
            compiled_fn = torch.compile(
                checkpointed_forward,
                fullgraph=True,
                backend="aot_eager_decomp_partition",
            )
            output_compiled = compiled_fn(input_compiled)
            output_compiled.y.sum().backward()

            self.assertEqual(output_eager.y, output_compiled.y)
            self.assertEqual(input_eager.x.grad, input_compiled.x.grad)
        finally:
            cleanup()

    def test_checkpoint_with_record_function(self):
        # Test that record_function ops are allowed inside checkpointed functions.
        # record_function is technically "impure" but safe to duplicate during
        # activation checkpointing recompute since it only sets up profiling spans.
        # This test verifies:
        # 1. No assertion error about impure ops in AC
        # 2. Forward graph contains record_function ops
        # 3. Code produces correct results
        def gn(x, y):
            with torch.profiler.record_function("matmul_region"):
                return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=False
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        # Verify record_function_enter_new appears in forward graph
        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.profiler._record_function_enter_new.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=nop,
            partition_fn=default_partition,
        )
        # Enable capture_profiler_record_function to trace record_function ops
        with torch._dynamo.config.patch(capture_profiler_record_function=True):
            self._validate(fn, backend, x, y)

    def _get_sac_annotations(self, checkpointed_fn, policy_fn, decompositions=None):
        annotations = []

        def capture_partition(joint_gm, joint_args, **kwargs):
            for node in joint_gm.graph.nodes:
                if node.op == "call_function":
                    recompute = node.meta.get("recompute", None)
                    if recompute is not None:
                        annotations.append(
                            f"{node.name}: {node.target} -> {recompute.name}"
                        )
            return min_cut_rematerialization_partition(joint_gm, joint_args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=capture_partition,
            decompositions=decompositions,
        )

        def fn(x):
            return checkpoint(
                checkpointed_fn,
                x,
                use_reentrant=False,
                context_fn=functools.partial(
                    create_selective_checkpoint_contexts, policy_fn
                ),
            )

        x = torch.randn(4, requires_grad=True)
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend=backend)
        out = compiled(x)
        out.sum().backward()
        return "\n".join(annotations)

    def test_pre_mode_decomp_has_sac_ignored_ops(self):
        SAVE_OPS = {
            torch.ops.aten.sin.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.cos.default,
        }

        def policy_fn(ctx, func, *args, **kwargs):
            if func in SAVE_OPS:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        @torch._dynamo.allow_in_graph
        def op_with_detach(x):
            a = x.sin()
            out = a.detach() + a
            out = out.cos()
            return out

        self.assertExpectedInline(
            self._get_sac_annotations(op_with_detach, policy_fn),
            """\
sin: aten.sin.default -> MUST_SAVE
detach_1: aten.detach.default -> PREFER_RECOMPUTE
add: aten.add.Tensor -> MUST_SAVE
cos: aten.cos.default -> MUST_SAVE""",
        )

    def test_post_mode_decomp(self):
        from torch._inductor.compile_fx import select_decomp_table

        def policy_fn(ctx, func, *args, **kwargs):
            if func == torch.ops.aten.silu.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            x = x.sin()
            x = torch.nn.functional.silu(x)
            x = x.cos()
            return x

        self.assertExpectedInline(
            self._get_sac_annotations(
                fn, policy_fn, decompositions=select_decomp_table
            ),
            """\
sin: aten.sin.default -> PREFER_RECOMPUTE
neg: aten.neg.default -> MUST_SAVE
exp: aten.exp.default -> MUST_SAVE
add: aten.add.Tensor -> MUST_SAVE
div: aten.div.Tensor -> MUST_SAVE
cos: aten.cos.default -> PREFER_RECOMPUTE""",
        )

    def test_multi_output_op(self):
        def policy_fn(ctx, func, *args, **kwargs):
            if func == torch.ops.aten.topk.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            x = x.sin()
            vals, idxs = torch.topk(x, k=2)
            out = vals.sum()
            out = out.cos()
            return out

        self.assertExpectedInline(
            self._get_sac_annotations(fn, policy_fn),
            """\
sin: aten.sin.default -> PREFER_RECOMPUTE
topk: aten.topk.default -> MUST_SAVE
getitem: <built-in function getitem> -> MUST_SAVE
getitem_1: <built-in function getitem> -> MUST_SAVE
sum_1: aten.sum.default -> PREFER_RECOMPUTE
cos: aten.cos.default -> PREFER_RECOMPUTE""",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_region_activation_memory_budget_reduces_act_mem(self):
        N, NUM_LAYERS = 1000, 4

        class Model(torch.nn.Module):
            def __init__(self, budget=None):
                super().__init__()
                self.budget = budget
                self.linears = torch.nn.ModuleList(
                    [torch.nn.Linear(N, N) for _ in range(NUM_LAYERS)]
                )

            def forward(self, x):
                if self.budget is not None:
                    with torch.autograd.graph.region_activation_memory_budget(
                        self.budget
                    ):
                        for linear in self.linears:
                            x = linear(x).relu()
                        return x.sum()
                else:
                    for linear in self.linears:
                        x = linear(x).relu()
                    return x.sum()

        def get_act_mem(f):
            out = f()
            out.backward()
            start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            out = f()
            act_mem = (
                torch.cuda.memory_stats()["requested_bytes.all.current"] - start_mem
            )
            out.backward()
            return act_mem

        x = torch.randn(N, N, device="cuda")

        torch._dynamo.reset()
        compiled = torch.compile(Model().cuda(), backend="aot_eager")
        self.assertGreater(get_act_mem(lambda: compiled(x)), 0)

        torch._dynamo.reset()
        compiled = torch.compile(Model(budget=0.0).cuda(), backend="aot_eager")
        self.assertEqual(get_act_mem(lambda: compiled(x)), 0)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_region_activation_memory_budget_survives_graph_break(self):
        from unittest.mock import patch

        import torch._functorch.partitioners as partitioners

        budgets = []
        choose_saved_values_set = partitioners.choose_saved_values_set

        def record_budget(joint_graph, node_info, memory_budget=1):
            budgets.append(memory_budget)
            return choose_saved_values_set(
                joint_graph, node_info, memory_budget=memory_budget
            )

        def fn(x, budget):
            with torch.autograd.graph.region_activation_memory_budget(budget):
                x = x.sin()
                torch._dynamo.graph_break()
                return x.cos()

        backend = aot_autograd(
            fw_compiler=lambda gm, _: gm.forward,
            bw_compiler=lambda gm, _: gm.forward,
            partition_fn=min_cut_rematerialization_partition,
        )

        with (
            patch.object(partitioners, "choose_saved_values_set", record_budget),
            torch._functorch.config.patch(activation_memory_budget=0.9),
        ):
            compiled = torch.compile(fn, backend=backend)
            for budget in (0.0, 0.7, 0.0):
                x = torch.randn(4, requires_grad=True)
                compiled(x, budget).sum().backward()

        self.assertEqual(budgets, [0.0, 0.0, 0.7, 0.7])

    @torch._functorch.config.patch(activation_memory_budget_require_full_coverage=False)
    def test_region_activation_memory_budget_partial_coverage_survives_graph_break(
        self,
    ):
        def fn(x):
            with torch.autograd.graph.region_activation_memory_budget(0.2):
                x = x.sin()
                torch._dynamo.graph_break()
                x = x.cos()
            return x + 1

        compiled = torch.compile(fn, backend="aot_eager")
        x = torch.randn(4, requires_grad=True)
        out = compiled(x)
        self.assertEqual(out, x.sin().cos() + 1)
        out.sum().backward()

    def test_region_activation_memory_budget_nested_graph_breaks(self):
        graphs = []

        def backend(gm, _):
            graphs.append(gm)
            return gm.forward

        def fn(x):
            with torch.autograd.graph.region_activation_memory_budget(0.2):
                x = x.sin()
                torch._dynamo.graph_break()
                with torch.autograd.graph.region_activation_memory_budget(0.7):
                    x = x.cos()
                    torch._dynamo.graph_break()
                    x = x.tan()
                torch._dynamo.graph_break()
                x = x + 1
            torch._dynamo.graph_break()
            return x * 2

        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), (x.sin().cos().tan() + 1) * 2)
        self.assertEqual(
            [
                [
                    torch.fx.traceback._get_memory_budget_annotation(node)
                    for node in gm.graph.nodes
                    if node.op in ("call_function", "call_method")
                ]
                for gm in graphs
            ],
            [[0.2], [0.7], [0.7], [0.2], [None]],
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_region_activation_memory_budget_per_region(self):
        """Different graphs (separated by a graph break) can have different
        memory budgets."""
        N, NUM_LAYERS = 1000, 2

        class Model(torch.nn.Module):
            def __init__(self, budget_a, budget_b):
                super().__init__()
                self.budget_a = budget_a
                self.budget_b = budget_b
                self.linears_a = torch.nn.ModuleList(
                    [torch.nn.Linear(N, N) for _ in range(NUM_LAYERS)]
                )
                self.linears_b = torch.nn.ModuleList(
                    [torch.nn.Linear(N, N) for _ in range(NUM_LAYERS)]
                )

            def forward(self, x):
                with torch.autograd.graph.region_activation_memory_budget(
                    self.budget_a
                ):
                    for linear in self.linears_a:
                        x = linear(x).relu()
                torch._dynamo.graph_break()
                with torch.autograd.graph.region_activation_memory_budget(
                    self.budget_b
                ):
                    for linear in self.linears_b:
                        x = linear(x).relu()
                    return x.sum()

        def get_act_mem(f):
            out = f()
            out.backward()
            start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            out = f()
            act_mem = (
                torch.cuda.memory_stats()["requested_bytes.all.current"] - start_mem
            )
            out.backward()
            return act_mem

        x = torch.randn(N, N, device="cuda")

        torch._dynamo.reset()
        both_save = torch.compile(Model(1.0, 1.0).cuda(), backend="aot_eager")
        mem_both_save = get_act_mem(lambda: both_save(x))

        torch._dynamo.reset()
        a_recomp = torch.compile(Model(0.0, 1.0).cuda(), backend="aot_eager")
        mem_a_recomp = get_act_mem(lambda: a_recomp(x))

        torch._dynamo.reset()
        b_recomp = torch.compile(Model(1.0, 0.0).cuda(), backend="aot_eager")
        mem_b_recomp = get_act_mem(lambda: b_recomp(x))

        torch._dynamo.reset()
        both_recomp = torch.compile(Model(0.0, 0.0).cuda(), backend="aot_eager")
        mem_both_recomp = get_act_mem(lambda: both_recomp(x))

        # Both save > either one recomputing > both recomputing
        self.assertGreater(mem_both_save, mem_a_recomp)
        self.assertGreater(mem_both_save, mem_b_recomp)
        self.assertGreater(mem_a_recomp, mem_both_recomp)
        self.assertGreater(mem_b_recomp, mem_both_recomp)

    def test_region_activation_memory_budget_validation(self):
        region_activation_memory_budget = (
            torch.autograd.graph.region_activation_memory_budget
        )
        with self.assertRaisesRegex(TypeError, "expects a float"):
            region_activation_memory_budget(True)
        with self.assertRaisesRegex(TypeError, "expects a float"):
            region_activation_memory_budget("0.5")
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            region_activation_memory_budget(2.0)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            region_activation_memory_budget(-0.1)

    def test_region_activation_memory_budget_eager_raises(self):
        """Using the context manager outside a torch.compile region is an error."""
        with self.assertRaisesRegex(RuntimeError, "inside a torch.compile region"):
            with torch.autograd.graph.region_activation_memory_budget(0.5):
                pass

    def test_region_activation_memory_budget_conflict_raises(self):
        """Two different budgets in one graph (no graph break) is an error: the
        partitioner applies a single budget per graph."""

        def fn(x, y):
            with torch.autograd.graph.region_activation_memory_budget(0.3):
                a = (torch.mm(x, y) + 1).relu()
            with torch.autograd.graph.region_activation_memory_budget(0.8):
                return (torch.mm(a, y) + 1).relu()

        cfn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "conflicting budgets"):
            cfn(x, y).sum().backward()

    def test_region_activation_memory_budget_partial_annotation_raises(self):
        """The default rejects a context that annotates only part of a graph."""

        def fn(x, y):
            with torch.autograd.graph.region_activation_memory_budget(0.3):
                a = (torch.mm(x, y) + 1).relu()
            # This op is outside the region -> partial annotation.
            return (a * 2).relu()

        cfn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "must cover the entire forward"):
            cfn(x, y).sum().backward()

    @torch._functorch.config.patch(activation_memory_budget_require_full_coverage=False)
    def test_region_activation_memory_budget_partial_coverage_allowed(self):
        from unittest.mock import patch

        import torch._functorch.partitioners as partitioners

        budgets = []
        choose_saved_values_set = partitioners.choose_saved_values_set

        def record_budget(joint_graph, node_info, memory_budget=1):
            budgets.append(memory_budget)
            return choose_saved_values_set(
                joint_graph, node_info, memory_budget=memory_budget
            )

        def fn(x, y):
            with torch.autograd.graph.region_activation_memory_budget(0.3):
                a = (torch.mm(x, y) + 1).relu()
            return (a * 2).relu()

        backend = aot_autograd(
            fw_compiler=lambda gm, _: gm.forward,
            bw_compiler=lambda gm, _: gm.forward,
            partition_fn=min_cut_rematerialization_partition,
        )
        with patch.object(partitioners, "choose_saved_values_set", record_budget):
            cfn = torch.compile(fn, backend=backend, fullgraph=True)
            x = torch.randn(8, 8, requires_grad=True)
            y = torch.randn(8, 8, requires_grad=True)
            cfn(x, y).sum().backward()

        self.assertEqual(budgets, [0.3])

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._functorch.config.patch(activation_memory_budget_require_full_coverage=False)
    def test_region_activation_memory_budget_partial_coverage_invoke_subgraph(self):
        from unittest.mock import patch

        import torch._functorch.partitioners as partitioners
        from torch.compiler import nested_compile_region

        budgets = []
        choose_saved_values_set = partitioners.choose_saved_values_set

        def record_budget(joint_graph, node_info, memory_budget=1):
            budgets.append(memory_budget)
            return choose_saved_values_set(
                joint_graph, node_info, memory_budget=memory_budget
            )

        weight = torch.randn(8, 8, requires_grad=True)

        @nested_compile_region
        def region(x):
            with torch.autograd.graph.region_activation_memory_budget(0.3):
                return (x @ weight).relu()

        def fn(x):
            return (region(x) + 1).sum()

        backend = aot_autograd(
            fw_compiler=lambda gm, _: gm.forward,
            bw_compiler=lambda gm, _: gm.forward,
            partition_fn=min_cut_rematerialization_partition,
        )
        with patch.object(partitioners, "choose_saved_values_set", record_budget):
            cfn = torch.compile(fn, backend=backend, fullgraph=True)
            x = torch.randn(8, 8, requires_grad=True)
            cfn(x).backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(budgets, [0.3, 0.3])

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_region_activation_memory_budget_covers_invoke_subgraph(self):
        """A budget covering a forward that contains an invoke_subgraph
        (nested_compile_region) applies inside the HOP body too: the budget
        propagates into the body so the whole forward is consistently covered."""
        from torch.compiler import nested_compile_region

        N, NUM_LAYERS = 1000, 4

        def build(budget):
            linears = torch.nn.ModuleList(
                [torch.nn.Linear(N, N) for _ in range(NUM_LAYERS)]
            ).cuda()

            @nested_compile_region
            def region(x):
                for lin in linears:
                    x = lin(x).relu()
                return x

            def fn(x):
                # Wrap the entire forward (HOP call + reduction); the budget must
                # consistently cover the HOP body too.
                if budget is None:
                    return region(x).sum()
                with torch.autograd.graph.region_activation_memory_budget(budget):
                    return region(x).sum()

            return fn

        def get_act_mem(f):
            out = f()
            out.backward()
            start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            out = f()
            act_mem = (
                torch.cuda.memory_stats()["requested_bytes.all.current"] - start_mem
            )
            out.backward()
            return act_mem

        x = torch.randn(N, N, device="cuda")

        torch._dynamo.reset()
        baseline = torch.compile(build(None), backend="aot_eager", fullgraph=True)
        self.assertGreater(get_act_mem(lambda: baseline(x)), 0)

        torch._dynamo.reset()
        recompute = torch.compile(build(0.0), backend="aot_eager", fullgraph=True)
        self.assertEqual(get_act_mem(lambda: recompute(x)), 0)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_region_activation_memory_budget_distinct_per_invoke_subgraph_raises(self):
        """Different budgets in two invoke_subgraph regions of a single graph are
        rejected: the agreement check recurses into nested subgraphs, so the
        outermost graph sees both budgets. Use a graph break for different
        budgets."""
        from torch.compiler import nested_compile_region

        wa = torch.randn(8, 8, requires_grad=True)
        wb = torch.randn(8, 8, requires_grad=True)

        @nested_compile_region
        def region_a(x):
            with torch.autograd.graph.region_activation_memory_budget(0.3):
                return (x @ wa).relu()

        @nested_compile_region
        def region_b(x):
            with torch.autograd.graph.region_activation_memory_budget(0.8):
                return (x @ wb).relu()

        def fn(x):
            return region_b(region_a(x)).sum()

        cfn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "conflicting budgets"):
            cfn(x).backward()

    def _shared_recompute_backend(
        self, shared, fw_mms=None, bw_mms=None, bw_order=None
    ):
        """aot_autograd backend asserting each graph makes ``shared`` shared
        recompute calls and, if given, runs fw_mms / bw_mms matmuls (counted
        inside the shared subgraphs too), and runs the backward's shared calls
        and matmuls in ``bw_order`` (e.g. ["recompute", "mm"]). Each backward
        call must read what the forward's call read: the same primal or saved
        value, or the same output of an earlier call."""
        from torch._higher_order_ops import invoke_subgraph

        def piece(call):
            return call.args[1].rsplit("_", 1)[0]

        def call_output(n):
            if n.target is operator.getitem and n.args[0].target is invoke_subgraph:
                return piece(n.args[0]), n.args[1]
            return None

        fw_operands = {}

        def check(gm, args, mms, order=None, bw=False):
            count_shared_recompute_calls(gm, args, shared)
            for n in gm.graph.find_nodes(op="call_function", target=invoke_subgraph):
                ops = n.args[2:]
                if not bw:
                    fw_operands[piece(n)] = [(a.name, call_output(a)) for a in ops]
                    continue
                for a, (name, out) in zip(ops, fw_operands[piece(n)], strict=True):
                    same = a.op == "placeholder" and a.name == name
                    if not same and (out is None or call_output(a) != out):
                        raise AssertionError(f"{a} isn't the forward's {name} in {gm}")
            if order is not None:
                actual = [
                    "recompute" if n.target is invoke_subgraph else "mm"
                    for n in gm.graph.nodes
                    if n.target in (invoke_subgraph, torch.ops.aten.mm.default)
                ]
                if actual != order:
                    raise AssertionError(f"expected order {order}, got {actual}:\n{gm}")
            if mms is not None:
                actual = sum(
                    n.target is torch.ops.aten.mm.default
                    for m in gm.modules()
                    if isinstance(m, torch.fx.GraphModule)
                    for n in m.graph.nodes
                )
                if actual != mms:
                    raise AssertionError(f"expected {mms} mms in {gm}, got {actual}")
            return gm

        return aot_autograd(
            fw_compiler=functools.partial(check, mms=fw_mms),
            bw_compiler=functools.partial(check, mms=bw_mms, order=bw_order, bw=True),
            partition_fn=min_cut_rematerialization_partition,
        )

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute(self, device):
        # Each region's recomputed ops, including its interior mm, become one
        # subgraph called from both the forward and the backward (backward: 2
        # recomputed + 4 grad mms). Each backward call runs right before the
        # first node that reads it, so one region's gradients come before the
        # next region is recomputed instead of every region recomputing (and
        # staying alive) up front.
        def gn(x, w):
            return torch.sigmoid(torch.matmul(x, w)).cos()

        def fn(x, w):
            y = torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)
            return torch.utils.checkpoint.checkpoint(gn, y, w, use_reentrant=False)

        x = torch.randn(4, 4, device=device, requires_grad=True)
        w = torch.randn(4, 4, device=device, requires_grad=True)
        order = ["recompute", "mm", "mm"] * 2
        backend = self._shared_recompute_backend(2, 2, 6, bw_order=order)
        self._validate(fn, backend, x, w)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_separate_subgraph_copies(self, device):
        # Inductor rewrites a graph's subgraph modules in place while compiling
        # it, so the forward and backward each get their own copy of the shared
        # subgraph; with one shared object the backward compile would see the
        # forward compile's rewrites. The forward compiler here marks its copy.
        def shared_mods(gm):
            return [
                m for name, m in gm.named_children() if name.startswith("recompute_")
            ]

        def fw_compiler(gm, args):
            for m in shared_mods(gm):
                m.meta["rewritten_by_fw_compile"] = True
            return gm

        bw_marked = []

        def bw_compiler(gm, args):
            mods = shared_mods(gm)
            bw_marked.extend("rewritten_by_fw_compile" in m.meta for m in mods)
            return gm

        def gn(x, w):
            return torch.sigmoid(torch.matmul(x, w)).cos()

        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)

        x = torch.randn(4, 4, device=device, requires_grad=True)
        w = torch.randn(4, 4, device=device, requires_grad=True)
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        torch.compile(fn, backend=backend, fullgraph=True)(x, w).sum().backward()
        self.assertEqual(bw_marked, [False])

    @requires_gpu_and_triton
    @torch._inductor.config.patch(fx_graph_cache=True, fx_graph_remote_cache=False)
    @torch._functorch.config.patch(
        deterministic_recompute=True, enable_autograd_cache=False
    )
    def test_deterministic_recompute_fx_graph_cache(self, device):
        # Shared subgraphs are named by their order in the graph rather than a
        # process-global id, and their region config is part of the cache key,
        # so recompiling the same function hits the FX graph cache for both the
        # forward and the backward.
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        def gn(x, w):
            return torch.sigmoid(torch.matmul(x, w)).cos()

        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)

        x = torch.randn(4, 4, device=device, requires_grad=True)
        w = torch.randn(4, 4, device=device, requires_grad=True)
        with fresh_cache():
            for _ in range(2):
                counters.clear()
                compiled = torch.compile(fn, backend="inductor", fullgraph=True)
                _, (fw_code, bw_code) = run_and_get_code(
                    lambda: compiled(x, w).sum().backward()
                )
                self.assertEqual(len(shared_recompute_code(fw_code, "fw")), 1)
                self.assertEqual(len(shared_recompute_code(bw_code, "bw")), 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_nested(self, device):
        # A region nested in another is recomputed as part of the outer one.
        def inner(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def outer(x, y):
            return torch.utils.checkpoint.checkpoint(inner, x, y, use_reentrant=False)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(outer, x, y, use_reentrant=False)

        x = torch.randn(8, 8, device=device, requires_grad=True)
        y = torch.randn(8, 8, device=device, requires_grad=True)
        self._validate(fn, self._shared_recompute_backend(1), x, y)

    @parametrize("backend", ["aot_eager", "inductor"])
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_forward_fallback(self, device, backend):
        # The piece (sin, mul, cos) reads m, which the forward computes after
        # the saved exp reads sin, so the call goes after m and exp moves below
        # it.
        if backend == "inductor" and not HAS_GPU_AND_TRITON:
            self.skipTest("requires a GPU and Triton")
        saved = [torch.ops.aten.mm.default, torch.ops.aten.exp.default]
        context_fn = functools.partial(create_selective_checkpoint_contexts, saved)

        def gn(x, y, w):
            a = x.sin()
            d = a.exp()
            m = y @ w
            return (a * m).cos() + d

        def fn(x, y, w):
            return torch.utils.checkpoint.checkpoint(
                gn, x, y, w, use_reentrant=False, context_fn=context_fn
            )

        args = [
            torch.randn(16, 16, device=device, requires_grad=True) for _ in range(3)
        ]
        if backend == "aot_eager":
            self._validate(fn, self._shared_recompute_backend(1), *args)
            return
        _, (fw_code, bw_code) = run_and_get_code(self._validate, fn, "inductor", *args)
        self.assertIn("def recompute_0_fw_mod", fw_code)
        self.assertIn("def recompute_0_bw_mod", bw_code)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_keeps_random_op_order(self, device):
        # As in forward_fallback, but the saved op reading sin is a dropout:
        # moving it below the call, past the dropout of y, would change what
        # each draws, so the region is left as partitioned.
        from torch._functorch._aot_autograd import deterministic_recompute

        saved = [torch.ops.aten.mm.default, torch.ops.aten.native_dropout.default]
        context_fn = functools.partial(create_selective_checkpoint_contexts, saved)

        def gn(x, y, w):
            a = x.sin()
            d = F.dropout(a, 0.5)
            m = F.dropout(y, 0.5) @ w
            return (a * m).cos() + d

        def fn(x, y, w):
            return torch.utils.checkpoint.checkpoint(
                gn, x, y, w, use_reentrant=False, context_fn=context_fn
            )

        check = functools.partial(count_shared_recompute_calls, expected=0)
        backend = aot_autograd(
            fw_compiler=check,
            bw_compiler=check,
            partition_fn=min_cut_rematerialization_partition,
        )
        args = [
            torch.randn(16, 16, device=device, requires_grad=True) for _ in range(3)
        ]
        with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
            torch.compile(fn, backend=backend, fullgraph=True)(*args).sum().backward()
        self.assertIn("it would move the random op", "\n".join(logs.output))

    @torch._functorch.config.patch(
        deterministic_recompute=True, activation_memory_budget=0.0
    )
    def test_deterministic_recompute_reads_earlier_piece(self, device):
        # With no memory to spare, the partitioner recomputes each region's
        # output instead of saving it, so each region's piece reads the
        # previous one's call in both graphs.
        def gn(x, w):
            return torch.sigmoid(torch.matmul(x, w)).cos()

        def fn(x, w):
            for _ in range(3):
                x = torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)
            return x

        x = torch.randn(16, 16, device=device, requires_grad=True)
        w = torch.randn(16, 16, device=device, requires_grad=True)
        self._validate(fn, self._shared_recompute_backend(3), x, w)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_memory_pool(self, device):
        # Nodes allocating in a user memory pool carry it in their metadata,
        # which the call replacing them wouldn't, so such a graph is left as
        # partitioned, and so is a nested_compile_region body called in one,
        # whose nodes carry only the metadata.
        from torch._functorch._aot_autograd import deterministic_recompute

        pool = torch.cuda.MemPool()

        def body(x, w):
            return (x @ w).sin().cos()

        def fn(x, w):
            with torch.cuda.use_mem_pool(pool):
                a = torch.utils.checkpoint.checkpoint(body, x, w, use_reentrant=False)
            return a * a

        @torch.compiler.nested_compile_region
        def layer(x, w):
            return torch.utils.checkpoint.checkpoint(body, x, w, use_reentrant=False)

        def in_region(x, w):
            with torch.cuda.use_mem_pool(pool):
                a = layer(layer(x, w), w)
            return a * a

        def check(gm, args):
            hop = torch.ops.higher_order.invoke_subgraph
            for m in gm.modules():
                for n in m.graph.find_nodes(op="call_function", target=hop):
                    if "recompute_" in n.args[1]:
                        raise AssertionError(f"{n.args[1]} is shared in {gm}")
            return gm

        backend = aot_autograd(
            fw_compiler=check,
            bw_compiler=check,
            partition_fn=min_cut_rematerialization_partition,
        )
        for f, prefix in ((fn, ""), (in_region, "subgraph_0_0_")):
            with self.subTest(f.__name__):
                torch._dynamo.reset()
                x = torch.randn(16, 16, device="cuda", requires_grad=True)
                w = torch.randn(16, 16, device="cuda", requires_grad=True)
                log = deterministic_recompute.log
                with self.assertLogs(log, level="INFO") as logs:
                    self._validate(f, backend, x, w)
                self.assertRegex(logs.output[0], f": {prefix}\\w+ is in a memory pool")

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_nested_compile_region(self, device):
        # A nested_compile_region called twice is partitioned on its own, so the
        # checkpoint inside it is shared within the region's forward and
        # backward, which Inductor lower identically.
        from torch._higher_order_ops import invoke_subgraph

        def body(x, w):
            return (x @ w).sin().cos()

        @torch.compiler.nested_compile_region
        def layer(x, w):
            return torch.utils.checkpoint.checkpoint(body, x, w, use_reentrant=False)

        def fn(x, w):
            return layer(layer(x, w), w)

        def check(gm, args):
            targets = [
                n.args[1]
                for m in gm.modules()
                for n in m.graph.find_nodes(op="call_function", target=invoke_subgraph)
            ]
            if sum("recompute_" in t for t in targets) != 1:
                raise AssertionError(f"expected one shared recompute call: {targets}")
            return gm

        aot = aot_autograd(
            fw_compiler=check,
            bw_compiler=check,
            partition_fn=min_cut_rematerialization_partition,
        )

        def args():
            return [
                torch.randn(8, 8, device=device, requires_grad=True) for _ in range(2)
            ]

        self._validate(fn, aot, *args())
        torch._dynamo.reset()
        _, (fw_code, bw_code) = run_and_get_code(
            self._validate, fn, "inductor", *args()
        )
        fw = shared_recompute_code(fw_code, "fw")
        self.assertEqual(len(fw), 1)
        self.assertEqual(fw, shared_recompute_code(bw_code, "bw"))
        ((_, sources),) = fw.values()
        for source in sources:
            self.assertIn("'deterministic': True", source)

    @parametrize("view", [False, True])
    @parametrize("backend", ["aot", "inductor"])
    @requires_gpu_and_triton
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_input_mutation(self, device, view, backend):
        # buf is mutated after the region by a copy_ that the graph keeps at its
        # end. Nothing orders the shared call against it, so the call must not
        # move past it, or the forward would read the mutated buf; also when the
        # region reads a view of buf, whose base the copy_ writes.
        from torch._higher_order_ops import invoke_subgraph

        def fn(x, buf):
            b = buf.t() if view else buf
            y = torch.utils.checkpoint.checkpoint(
                lambda x: (x + b).relu(), x, use_reentrant=False
            )
            buf.add_(1)
            return y

        def call_before_copy(gm, args):
            # The call goes right before the copy_ (its outputs then follow it).
            count_shared_recompute_calls(gm, args, 1)
            nodes = [n for n in gm.graph.nodes if n.target is not operator.getitem]
            copy = gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.copy_.default
            )[0]
            if nodes[nodes.index(copy) - 1].target is not invoke_subgraph:
                raise AssertionError(f"the call isn't right before the copy_ in {gm}")
            return gm

        if backend == "aot":
            backend = aot_autograd(
                fw_compiler=call_before_copy,
                bw_compiler=functools.partial(count_shared_recompute_calls, expected=1),
                partition_fn=min_cut_rematerialization_partition,
                keep_inference_input_mutations=True,
            )
        x = torch.randn(8, 8, device=device, requires_grad=True)
        buf = torch.randn(8, 8, device=device)
        _, code = run_and_get_code(self._validate, fn, backend, x, buf)
        if backend == "inductor":
            fw_code, bw_code = code
            self.assertEqual(len(shared_recompute_code(fw_code, "fw")), 1)
            self.assertEqual(len(shared_recompute_code(bw_code, "bw")), 1)

    @requires_distributed()
    @parametrize(
        "case",
        [
            "reads_after_write",
            "read_before_and_after",
            "write_between_reads",
            "reads_around_write",
        ],
    )
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_inplace_write(self, device, case):
        # fsdp.copy_ (an in-place write the graph keeps where it is) writes buf,
        # so only buf's storage orders it against the region's reads of buf. A
        # piece reading buf after the write goes after it, also when a saved mm
        # reads the piece before the write (the mm moves below the call). With
        # a policy saving the mm, a piece whose call has to wait for the mm
        # would move the write, which reads the piece's sin, past the read of
        # buf in between: that region is left as partitioned, and so is one
        # whose piece reads buf on both sides of the write.
        import torch.distributed.fsdp
        from torch._functorch._aot_autograd import deterministic_recompute

        def reads_after_write(x, w, buf):
            y = (x @ w).sin()
            torch.ops.fsdp.copy_(buf, x.detach() * 2)
            return (y * buf).cos()

        def read_before_and_after(x, w, buf):
            a = x.sin()
            b = a @ w
            torch.ops.fsdp.copy_(buf, x.detach() * 2)
            return (a * buf).cos() + b

        def write_between_reads(x, w, buf):
            s = x.sin()
            torch.ops.fsdp.copy_(buf, s.detach())
            t = buf * 3
            return (s * (x @ w)).cos() + t

        def reads_around_write(x, w, buf):
            a = (x @ w) + buf
            torch.ops.fsdp.copy_(buf, x.detach() * 2)
            return (a * buf).sin()

        gn, kwargs, shared = {
            "reads_after_write": (reads_after_write, {}, 1),
            "read_before_and_after": (
                read_before_and_after,
                {"context_fn": save_mm_context_fn},
                1,
            ),
            "write_between_reads": (
                write_between_reads,
                {"context_fn": save_mm_context_fn},
                0,
            ),
            "reads_around_write": (reads_around_write, {}, 0),
        }[case]

        def fn(x, w, buf):
            return torch.utils.checkpoint.checkpoint(
                gn, x, w, buf, use_reentrant=False, **kwargs
            )

        x = torch.randn(8, 8, device=device, requires_grad=True)
        w = torch.randn(8, 8, device=device, requires_grad=True)
        buf = torch.zeros(8, 8, device=device)
        xe, we = (t.detach().clone().requires_grad_() for t in (x, w))
        # (Eager checkpoint would write buf again in its recompute.)
        expected = gn(xe, we, buf.clone())
        expected.sum().backward()
        compiled = torch.compile(
            fn, backend=self._shared_recompute_backend(shared), fullgraph=True
        )
        logs = self.assertNoLogs if shared else self.assertLogs
        with logs(deterministic_recompute.log, level="INFO") as cm:
            out = compiled(x, w, buf.clone())
            out.sum().backward()
        self.assertEqual(out, expected)
        # (In reads_around_write the partitioner recomputes a from the written
        # buf, flag or not, so its gradients are wrong either way.)
        if case != "reads_around_write":
            self.assertEqual(x.grad, xe.grad)
            self.assertEqual(w.grad, we.grad)
        if not shared:
            self.assertIn("would reorder", cm.output[0])

    @parametrize("flag", ["enable_complex_wrapper", "compiled_autograd"])
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_not_applied(self, device, flag):
        # Both trace the graphs again after the pass, and can't trace its calls:
        # the complex wrapper failed to compile, and compiled autograd
        # graph-broke on each backward call and compiled it without deterministic
        # mode. So the pass leaves the graphs alone.
        def gn(x, w):
            return (x @ w).sin().cos()

        def step(x, w):
            out = torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)
            loss = out.abs().sum()
            loss.backward()
            return loss

        dtype = torch.complex64 if flag == "enable_complex_wrapper" else torch.float32
        x = torch.randn(8, 8, dtype=dtype, device=device, requires_grad=True)
        w = torch.randn(8, 8, dtype=dtype, device=device, requires_grad=True)
        xe, we = (t.detach().clone().requires_grad_() for t in (x, w))
        step(xe, we)
        check = functools.partial(count_shared_recompute_calls, expected=0)
        backend = aot_autograd(
            fw_compiler=check,
            bw_compiler=check,
            partition_fn=min_cut_rematerialization_partition,
        )
        functorch_cfg = {flag: True} if flag == "enable_complex_wrapper" else {}
        dynamo_cfg = {flag: True} if flag == "compiled_autograd" else {}
        with (
            torch._functorch.config.patch(**functorch_cfg),
            torch._dynamo.config.patch(**dynamo_cfg),
        ):
            torch.compile(step, backend=backend)(x, w)
        self.assertEqual(x.grad, xe.grad)
        self.assertEqual(w.grad, we.grad)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_input_free_piece(self, device):
        # The recompute is only constants (a factory op and its exp), which are
        # copied into the pieces that read them, so it has no piece of its own;
        # the backward reads the exp directly, which the log says.
        from torch._functorch._aot_autograd import deterministic_recompute

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                lambda x: x * torch.full((16,), 0.5, device=x.device).exp(),
                x,
                use_reentrant=False,
            )

        x = torch.randn(16, 16, device=device, requires_grad=True)
        with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
            _, codes = run_and_get_code(self._validate, fn, "inductor", x)
        for code in codes:
            self.assertNotIn("def recompute_", code)
        self.assertIn("the backward reads it directly", logs.output[0])

    @parametrize("case", ["view", "saved_cat", "no_size_asserts"])
    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_unbound_symbols(self, device, case):
        # Under dynamic shapes the piece's input has a size that binds no
        # symbol for Inductor on its own: h is s0*s1 by s2, a saved cat is
        # 2*(s0//2) by s1, and without size_asserts no input size binds one.
        # The piece also takes the SymInt graph inputs that are those symbols.
        def policy(ctx, op, *args, **kwargs):
            if op is torch.ops.aten.cat.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def view(x, w):
            h = x.view(-1, x.shape[-1]) @ w if case == "view" else x @ w
            return torch.utils.checkpoint.checkpoint(
                lambda h: h.sin().sin(), h, use_reentrant=False
            )

        def saved_cat(x, w):
            def gn(x, w):
                half = x[: x.shape[0] // 2]
                return ((torch.cat([half, half]) @ w).sin() @ w).cos()

            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                w,
                use_reentrant=False,
                context_fn=functools.partial(
                    create_selective_checkpoint_contexts, policy
                ),
            )

        fn = saved_cat if case == "saved_cat" else view
        shape = {"view": (4, 8, 16), "saved_cat": (6, 16)}.get(case, (8, 16))
        x = torch.randn(shape, device=device, requires_grad=True)
        w = torch.randn(16, 16, device=device, requires_grad=True)
        xe, we = (t.detach().clone().requires_grad_() for t in (x, w))
        fn(xe, we).sum().backward()
        with torch._inductor.config.patch(size_asserts=case != "no_size_asserts"):
            compiled = torch.compile(fn, backend="inductor", dynamic=True)
            _, (fw_code, bw_code) = run_and_get_code(
                lambda: compiled(x, w).sum().backward()
            )
        self.assertEqual(len(shared_recompute_code(fw_code, "fw")), 1)
        self.assertEqual(len(shared_recompute_code(bw_code, "bw")), 1)
        self.assertEqual(x.grad, xe.grad, atol=1e-4, rtol=1e-4)
        self.assertEqual(w.grad, we.grad, atol=1e-4, rtol=1e-4)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(fx_graph_cache=True, fx_graph_remote_cache=False)
    @torch._functorch.config.patch(
        deterministic_recompute=True, enable_autograd_cache=True
    )
    def test_deterministic_recompute_compiled_autograd_cache_key(self, device):
        # The pass isn't applied under compiled autograd, so the AOTAutograd
        # cache must not serve a graph compiled without it.
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(
                lambda x, w: (x @ w).sin(), x, w, use_reentrant=False
            )

        x = torch.randn(16, 16, device=device, requires_grad=True)
        w = torch.randn(16, 16, device=device, requires_grad=True)
        with fresh_cache():
            for compiled_autograd in (False, True):
                torch._dynamo.reset()
                counters.clear()
                with torch._dynamo.config.patch(compiled_autograd=compiled_autograd):
                    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
                    compiled(x, w).sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_resized_storage(self, device):
        # resize_ frees buf's storage after the region reads it (as FSDP2 does
        # after an unshard), which its schema doesn't show as a write: the call
        # has to stay before it, also in a graph with another in-place write
        # (the counter's).
        def fn(x, buf, counter):
            y = torch.utils.checkpoint.checkpoint(
                lambda x: (x @ buf).sin(), x, use_reentrant=False
            )
            buf.untyped_storage().resize_(0)
            counter.add_(1)
            return y.cos()

        x = torch.randn(8, 8, device=device, requires_grad=True)
        buf0 = torch.randn(8, 8, device=device)
        xe = x.detach().clone().requires_grad_()
        expected = (xe @ buf0).sin().cos()
        expected.sum().backward()
        buf = buf0.clone()
        nbytes = buf.untyped_storage().nbytes()
        backend = self._shared_recompute_backend(1)
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        out = compiled(x, buf, torch.zeros(1, device=device))
        # Like FSDP2 before the backward, bring buf back.
        buf.untyped_storage().resize_(nbytes)
        with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(buf):
            buf.copy_(buf0)
        out.sum().backward()
        self.assertEqual(out, expected)
        self.assertEqual(x.grad, xe.grad)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_unbacked(self, device):
        # .item() and nonzero define unbacked symbols. Inductor can't bind one
        # coming out of a subgraph, nor in general one passed in, so every
        # recomputed node that reads or produces a value carrying it stays out
        # of the pieces, and so does everything computed from it (the mask's sum
        # and cos); the matmul before it is still shared. An unbacked size of a
        # graph input (x's rows in unbacked_input) is fine.
        from torch._functorch._aot_autograd import deterministic_recompute

        def item_gn(x, w):
            h = x @ w
            return h.sin() * int((h > 0).sum().item())

        def nonzero_gn(x, w):
            h = x @ w
            return h[torch.nonzero(h[:, 0] > 0).squeeze(-1)].sin().sum(0)

        def mask_gn(x, w):
            h = x @ w
            return h.sin() * h[h[:, 0] > 0].sum(0).cos()

        def input_gn(x, w):
            h = x @ w
            return h[:, w[0] > 0].sin()

        def ckpt(gn, **kwargs):
            return lambda x, w: torch.utils.checkpoint.checkpoint(
                gn, x, w, use_reentrant=False, **kwargs
            )

        def saving(op):
            return functools.partial(create_selective_checkpoint_contexts, [op])

        item = {"capture_scalar_outputs": True}
        nonzero = {"capture_dynamic_output_shape_ops": True}
        # Each case also names the node the log reports, where the drop starts.
        cases = {
            "item": (ckpt(item_gn), item, "_local_scalar_dense"),
            "nonzero": (ckpt(nonzero_gn), nonzero, "nonzero"),
            # The forward computes nonzero, outside any region.
            "saved_nonzero": (
                ckpt(nonzero_gn, context_fn=saving(torch.ops.aten.nonzero.default)),
                nonzero,
                "squeeze",
            ),
            "mask": (ckpt(mask_gn), nonzero, "index"),
            "unbacked_input": (ckpt(input_gn), nonzero, "index"),
        }
        for name, (fn, dynamo_cfg, start) in cases.items():
            with self.subTest(name), torch._dynamo.config.patch(**dynamo_cfg):
                torch._dynamo.reset()
                x = torch.randn(8, 8, device=device, requires_grad=True)
                w = torch.randn(8, 8, device=device, requires_grad=True)
                if name == "unbacked_input":
                    torch._dynamo.decorators.mark_unbacked(x, 0)
                xe, we = (t.detach().clone().requires_grad_() for t in (x, w))
                fn(xe, we).sum().backward()
                compiled = torch.compile(fn, backend="inductor", fullgraph=True)
                with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
                    _, (fw_code, bw_code) = run_and_get_code(
                        lambda: compiled(x, w).sum().backward()
                    )
                (log,) = logs.output
                self.assertIn(
                    f"{start} left as partitioned, with what is computed", log
                )
                for code, side in ((fw_code, "fw"), (bw_code, "bw")):
                    ((body, _),) = shared_recompute_code(code, side).values()
                    self.assertIn("extern_kernels.mm(", body)
                self.assertEqual(x.grad, xe.grad, atol=1e-3, rtol=1e-3)
                self.assertEqual(w.grad, we.grad, atol=1e-3, rtol=1e-3)

    @requires_gpu_and_triton
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_get_attr(self, device):
        # A tensor constant and flex_attention's score_mod are graph attributes
        # (get_attr). They are copied into the pieces that read them, so the
        # recompute reading them, and everything computed from it, is shared.
        from torch.nn.attention.flex_attention import flex_attention

        def const_gn(x, w):
            c = torch.tensor([1.0, 2.0, 3.0, 4.0], device=x.device).repeat(4)
            return ((x @ w).sin() * c @ w).sigmoid()

        def flex_gn(x, w):
            q = (x @ w).view(1, 1, 16, 16)
            out = flex_attention(q, q, q, score_mod=lambda s, b, h, i, j: s * 0.5)
            return out.view(16, 16).sin() @ w

        recomputed = {
            "const": [torch.ops.aten.sin.default, torch.ops.aten.sigmoid.default],
            "flex": [torch.ops.higher_order.flex_attention, torch.ops.aten.sin.default],
        }

        def check(gm, args, name):
            unshared = [n for n in gm.graph.nodes if n.target in recomputed[name]]
            if unshared:
                raise AssertionError(f"recomputed outside a shared piece: {unshared}")
            return gm

        for name, gn in (("const", const_gn), ("flex", flex_gn)):
            aot = aot_autograd(
                fw_compiler=functools.partial(check, name=name),
                bw_compiler=functools.partial(check, name=name),
                partition_fn=min_cut_rematerialization_partition,
            )
            for backend in (aot, "inductor"):
                with self.subTest(name, inductor=backend == "inductor"):

                    def fn(x, w, gn=gn):
                        return torch.utils.checkpoint.checkpoint(
                            gn, x, w, use_reentrant=False
                        )

                    x = torch.randn(16, 16, device=device, requires_grad=True)
                    w = torch.randn(16, 16, device=device, requires_grad=True)
                    xe, we = (t.detach().clone().requires_grad_() for t in (x, w))
                    fn(xe, we).sum().backward()
                    compiled = torch.compile(fn, backend=backend, fullgraph=True)
                    _, code = run_and_get_code(lambda: compiled(x, w).sum().backward())
                    if backend == "inductor":
                        fw_code, bw_code = code
                        self.assertEqual(len(shared_recompute_code(fw_code, "fw")), 1)
                        self.assertEqual(len(shared_recompute_code(bw_code, "bw")), 1)
                    self.assertEqual(x.grad, xe.grad, atol=1e-3, rtol=1e-3)
                    self.assertEqual(w.grad, we.grad, atol=1e-3, rtol=1e-3)

    @parametrize("save_mm", [False, True])
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_unsafe_view(self, device, save_mm):
        # A linear on a 3D input reshapes its mm with _unsafe_view, a view whose
        # schema doesn't say so; without decompositions it stays in the graph.
        # It is copied into the piece (after the mm, or reading the mm a policy
        # saves), never returned.
        kwargs = {"context_fn": save_mm_context_fn} if save_mm else {}

        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(
                lambda x, w: F.linear(x, w).sin().cos(),
                x,
                w,
                use_reentrant=False,
                **kwargs,
            )

        target = torch.ops.aten._unsafe_view.default

        def check(gm, args):
            count_shared_recompute_calls(gm, args, 1)
            mods = [m for m in gm.modules() if isinstance(m, torch.fx.GraphModule)]
            if not any(n.target is target for m in mods for n in m.graph.nodes):
                raise AssertionError(f"no _unsafe_view in {gm}")
            for name, mod in gm.named_children():
                outs = mod.graph.output_node().args[0]
                if any(o.target is target for o in outs):
                    raise AssertionError(f"{name} returns an _unsafe_view")
            return gm

        backend = aot_autograd(
            fw_compiler=check,
            bw_compiler=check,
            partition_fn=min_cut_rematerialization_partition,
        )
        x = torch.randn(2, 4, 8, device=device, requires_grad=True)
        w = torch.randn(8, 8, device=device, requires_grad=True)
        self._validate(fn, backend, x, w)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_view_of_view(self, device):
        # The piece reads the inner view only through the outer one, so it copies
        # in both.
        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(
                lambda x, w: (x.view(64).view(8, 8) @ w).sin(),
                x,
                w,
                use_reentrant=False,
            )

        args = [torch.randn(8, 8, device=device, requires_grad=True) for _ in range(2)]
        self._validate(fn, self._shared_recompute_backend(1), *args)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_returns_only_live_values(self, device):
        # The view between mm and sigmoid is read only inside the piece, so the
        # piece returns neither it nor the mm it views: only the sigmoid, which
        # the backward reads.
        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(
                lambda x, w: (x @ w).view(-1).sigmoid(), x, w, use_reentrant=False
            )

        def check(gm, args):
            count_shared_recompute_calls(gm, args, 1)
            for name, mod in gm.named_children():
                outs = mod.graph.output_node().args[0]
                if [o.target for o in outs] != [torch.ops.aten.sigmoid.default]:
                    raise AssertionError(
                        f"{name} returns more than sigmoid:\n{mod.code}"
                    )
            return gm

        backend = aot_autograd(
            fw_compiler=check,
            bw_compiler=check,
            partition_fn=min_cut_rematerialization_partition,
        )
        args = [torch.randn(4, 4, device=device, requires_grad=True) for _ in range(2)]
        self._validate(fn, backend, *args)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_dynamic_shapes(self, device):
        # The region reshapes by a SymInt computed from x's size, which the
        # piece computes itself.
        def gn(x, y):
            h = torch.sigmoid(torch.matmul(x, y))
            return h.reshape(x.shape[0] // 2, -1).cos()

        for sac in (False, True):
            kwargs = {"context_fn": save_mm_context_fn} if sac else {}
            with self.subTest(sac=sac):
                torch._dynamo.reset()

                def fn(x, y):
                    return torch.utils.checkpoint.checkpoint(
                        gn, x, y, use_reentrant=False, **kwargs
                    ).sum()

                x = torch.randn(8, 4, device=device, requires_grad=True)
                y = torch.randn(4, 4, device=device, requires_grad=True)
                xe, ye = (t.detach().clone().requires_grad_() for t in (x, y))
                fn(xe, ye).backward()
                backend = self._shared_recompute_backend(1)
                torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)(
                    x, y
                ).backward()
                self.assertEqual(x.grad, xe.grad)
                self.assertEqual(y.grad, ye.grad)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_sac(self, device):
        # With a policy that saves the mm, only the rest of the region is
        # recomputed and shared (backward: 2 grad mms); one that saves nothing
        # recomputes the whole region, like plain AC (backward: 1 recomputed + 2
        # grad mms).
        def gn(x, w):
            return torch.sigmoid(torch.matmul(x, w)).cos()

        save_nothing = functools.partial(create_selective_checkpoint_contexts, [])
        for context_fn, bw_mms in ((save_mm_context_fn, 2), (save_nothing, 3)):
            with self.subTest(saves_mm=context_fn is save_mm_context_fn):
                torch._dynamo.reset()

                def fn(x, w, context_fn=context_fn):
                    return torch.utils.checkpoint.checkpoint(
                        gn, x, w, use_reentrant=False, context_fn=context_fn
                    )

                x = torch.randn(4, 4, device=device, requires_grad=True)
                w = torch.randn(4, 4, device=device, requires_grad=True)
                backend = self._shared_recompute_backend(1, 1, bw_mms)
                self._validate(fn, backend, x, w)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_sac_returns_no_views(self, device):
        # The shared subgraph recomputes gelu, but h.t() (a view of one of its
        # outputs) and w.t() (a view of one of its inputs) stay inline outside
        # it: invoke_subgraph outputs may not alias its inputs or each other.
        def gn(x, w):
            h = F.gelu(torch.matmul(x, w))
            return torch.matmul(h.t(), w.t())

        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(
                gn, x, w, use_reentrant=False, context_fn=save_mm_context_fn
            )

        def storage(n):
            return StorageWeakRef(n.meta["val"].untyped_storage())

        def check(gm, args):
            for name, mod in gm.named_children():
                if not name.startswith("recompute_"):
                    continue
                ins = {storage(n) for n in mod.graph.find_nodes(op="placeholder")}
                outs = [storage(n) for n in mod.graph.output_node().args[0]]
                if len(set(outs)) != len(outs) or ins & set(outs):
                    raise AssertionError(f"{name} returns an alias:\n{mod.code}")
            return count_shared_recompute_calls(gm, args, 1)

        x = torch.randn(4, 4, device=device, requires_grad=True)
        w = torch.randn(4, 4, device=device, requires_grad=True)
        backend = aot_autograd(
            fw_compiler=check,
            bw_compiler=check,
            partition_fn=min_cut_rematerialization_partition,
        )
        self._validate(fn, backend, x, w)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_multi_output_op(self, device):
        # A tuple-output op stays whole in its piece, which returns whichever of
        # its results either graph reads: var_mean's getitems; native_layer_norm's
        # mean and rstd, and SDPA's logsumexp etc., which only the backward
        # reads. A multi-output view op (split, unbind, and unsafe_chunk's
        # unsafe_split, whose schema doesn't say it's a view) stays inline
        # instead.
        def var_mean(x, y):
            h = torch.matmul(x, y)
            v, m = torch.var_mean(h, dim=-1, keepdim=True)
            return (h - m) * torch.rsqrt(v + 1e-5)

        def layer_norm(x, y):
            h = torch.matmul(x, y)
            return torch.ops.aten.native_layer_norm(
                h, (h.shape[-1],), None, None, 1e-5
            )[0]

        def sdpa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True).sin()

        def split(x, y):
            a, b = torch.matmul(x, y).sin().split(4, dim=-1)
            return a.cos() * b

        def unbind(x, y):
            a, b = torch.matmul(x, y).sin().view(8, 2, 4).unbind(1)
            return a.cos() * b

        def unsafe_chunk(x, y):
            a, b = torch.matmul(x, y).sin().unsafe_chunk(2, dim=-1)
            return a.cos() * b

        save_mm = {"context_fn": save_mm_context_fn}
        cases = (
            (var_mean, save_mm, [(8, 8)] * 2),
            (layer_norm, save_mm, [(8, 8)] * 2),
            (sdpa, {}, [(2, 4, 16, 8)] * 3),
            (split, {}, [(8, 8)] * 2),
            (unbind, {}, [(8, 8)] * 2),
            (unsafe_chunk, {}, [(8, 8)] * 2),
        )
        for gn, kwargs, shapes in cases:
            with self.subTest(gn.__name__):
                torch._dynamo.reset()

                def fn(*args, gn=gn, kwargs=kwargs):
                    return torch.utils.checkpoint.checkpoint(
                        gn, *args, use_reentrant=False, **kwargs
                    )

                args = [
                    torch.randn(s, device=device, requires_grad=True) for s in shapes
                ]
                self._validate(fn, self._shared_recompute_backend(1), *args)

    @parametrize("case", ["plain", "sac", "benchmark_fusion"])
    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_identical_kernels(self, device, case):
        # The forward and backward compiles each lower the shared subgraph. They
        # must emit the same code for it, and its kernels must be compiled with
        # deterministic=True so runtime autotuning can't pick different configs
        # for the two copies; kernels outside it keep normal autotuning. The
        # subgraph turns off benchmark_fusion, which deterministic mode doesn't
        # allow, if it's on globally.
        def gn(x, w):
            h = torch.matmul(x, w).float()
            scale = torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6)
            return (h * scale).to(x.dtype)

        kwargs = {"context_fn": save_mm_context_fn} if case == "sac" else {}

        def fn(x, w):
            y = torch.utils.checkpoint.checkpoint(
                gn, x, w, use_reentrant=False, **kwargs
            )
            return (y * 3.0).square().sum()

        # Rows of 4096 make the norm a looped reduction, whose autotuned configs
        # would change the order it accumulates in.
        x = torch.randn(8, 64, dtype=torch.bfloat16, device=device)
        w = torch.randn(64, 4096, dtype=torch.bfloat16, device=device)
        x.requires_grad_()
        with torch._inductor.config.patch(benchmark_fusion=case == "benchmark_fusion"):
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            _, (fw_code, bw_code) = run_and_get_code(lambda: compiled(x, w).backward())
        fw = shared_recompute_code(fw_code, "fw")
        self.assertEqual(len(fw), 1)
        self.assertEqual(fw, shared_recompute_code(bw_code, "bw"))
        ((_, sources),) = fw.values()
        self.assertTrue(any("@triton_heuristics.reduction(" in src for src in sources))
        for source in sources:
            self.assertIn("'deterministic': True", source)
        for code in (fw_code, bw_code):
            call = re.search(
                r"\n    def call\(self, args\):\n((?: +[^\n]*\n|\n)+)", code
            )
            kernels = set(re.findall(r"(triton_\w+)\.run\(", call.group(1)))
            self.assertTrue(kernels)
            for k in kernels:
                d = code.index(f"\ndef {k}(")
                meta = code[code.rindex("\n@triton_heuristics.", 0, d) : d]
                self.assertIn("'deterministic': False", meta)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        force_disable_caches=True, combo_kernels=True, benchmark_combo_kernel=True
    )
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_benchmark_combo_kernel(self, device):
        # Deterministic mode doesn't allow benchmarking combo kernels (here sin
        # and cos), so a piece turns that off when it's on globally.
        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(
                lambda x, w: torch.sin(x) @ torch.cos(w), x, w, use_reentrant=False
            ).sum()

        x = torch.randn(8, 64, device=device, requires_grad=True)
        w = torch.randn(64, 4096, device=device, requires_grad=True)
        _, codes = run_and_get_code(self._validate, fn, "inductor", x, w)
        for code, side in zip(codes, ("fw", "bw"), strict=True):
            self.assertEqual(len(shared_recompute_code(code, side)), 1)

    @requires_gpu_and_triton
    @parametrize(
        "case", ["looped_reduction", "layer_norm", "padded_output", "padded_input"]
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_bitwise(self, device, case):
        # gh-186572: the backward's recomputed activation must equal the forward's
        # bit for bit. With g all ones, y * g is the forward's y, and g's gradient
        # is the backward's y. Without the flag, the two compiles can autotune
        # the looped reduction differently. With rows of 1101 floats, the
        # backward's compile would pad the piece's outputs, or the saved a it
        # reads, where the forward's doesn't, and the two copies would then
        # accumulate in different orders.
        def norm(h):
            return h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6)

        ckpt = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        region, shapes = {
            "looped_reduction": (lambda x, w: norm(x @ w), [(64, 64), (64, 4096)]),
            "layer_norm": (
                lambda x, w: F.layer_norm(x @ w, (w.shape[-1],)),
                [(64, 64), (64, 4096)],
            ),
            "padded_output": (lambda x: norm(x.sin()), [(64, 1101)]),
            # x.sum(0) is saved, and padded by the forward's compile.
            "padded_input": (None, [(2, 64, 1101)]),
        }[case]

        def fn(g, *args):
            if region is None:
                return ckpt(norm, args[0].sum(0)) * g
            return ckpt(region, *args) * g

        args = [torch.randn(s, device=device, requires_grad=True) for s in shapes]
        g = torch.ones(64, shapes[-1][-1], device=device, requires_grad=True)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)

        def run():
            out = compiled(g, *args)
            out.backward(torch.ones_like(out))
            return out

        out, (fw_code, bw_code) = run_and_get_code(run)
        self.assertEqual(len(shared_recompute_code(fw_code, "fw")), 1)
        self.assertEqual(len(shared_recompute_code(bw_code, "bw")), 1)
        self.assertEqual(out, g.grad, atol=0, rtol=0)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_only_views(self, device):
        # With the mm saved, the recompute is just the transposes, views with no
        # kernel to drift: nothing to share.
        from torch._functorch._aot_autograd import deterministic_recompute

        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(
                lambda x, w: x.t() @ w.t(),
                x,
                w,
                use_reentrant=False,
                context_fn=save_mm_context_fn,
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        w = torch.randn(4, 4, device=device, requires_grad=True)
        with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
            self._validate(fn, self._shared_recompute_backend(0), x, w)
        self.assertIn("it recomputes only views", logs.output[0])

    @requires_gpu_and_triton
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_primal_not_overwritten(self, device):
        # The shared subgraph reads views of the weight (w.t()) and of saved
        # values. Inductor treats invoke_subgraph outputs as fresh buffers, so if
        # it returned them Inductor could reuse w's buffer in place for a
        # gradient.
        def plain(x, w):
            return x + F.silu(x @ w) @ w.t()

        def selective(x, w):
            h = torch.mm(x, w)
            return torch.relu(h) * w.t() + h.t() * w

        for sac in (False, True):
            gn = selective if sac else plain
            kwargs = {"context_fn": save_mm_context_fn} if sac else {}
            with self.subTest(sac=sac):
                torch._dynamo.reset()

                def fn(x, w):
                    for _ in range(2):
                        x = torch.utils.checkpoint.checkpoint(
                            gn, x, w, use_reentrant=False, **kwargs
                        )
                    return (x.cos() * x).sum()

                x = torch.randn(64, 64, device=device)
                w = torch.randn(64, 64, device=device) * 0.05
                xe, we = x.clone().requires_grad_(), w.clone().requires_grad_()
                xc, wc = x.clone().requires_grad_(), w.clone().requires_grad_()
                fn(xe, we).backward()
                compiled = torch.compile(fn, backend="inductor", fullgraph=True)
                _, (fw_code, bw_code) = run_and_get_code(
                    lambda: compiled(xc, wc).backward()
                )
                fw = shared_recompute_code(fw_code, "fw")
                self.assertEqual(len(fw), 2)
                self.assertEqual(fw, shared_recompute_code(bw_code, "bw"))
                self.assertEqual(wc, w)
                self.assertEqual(xc.grad, xe.grad, atol=1e-3, rtol=1e-3)
                self.assertEqual(wc.grad, we.grad, atol=1e-3, rtol=1e-3)

    @requires_distributed()
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_collective(self, device):
        # A collective inside a checkpoint region is recomputed with it, so it
        # runs inside the shared subgraph, out of reach of Inductor's
        # communication overlap passes. Under a policy that saves it, its wait
        # is a view of a piece input (Inductor lowers it to return its input),
        # so it stays inline in both graphs and the piece, whose backward reads
        # it here, must not return it; the same goes for wait_tensors and its
        # getitems. A single-rank gloo group makes the collective well defined.
        import torch.distributed._functional_collectives as ft_c

        dist.init_process_group(
            backend="gloo", store=dist.HashStore(), rank=0, world_size=1
        )
        try:

            def gn(x, y):
                h = ft_c.all_reduce(torch.matmul(x, y), "sum", dist.group.WORLD)
                return torch.sigmoid(h).cos()

            def saved_gn(x, y):
                h = ft_c.all_reduce(torch.matmul(x, y), "sum", dist.group.WORLD)
                return (h * h.sigmoid()).cos()

            c10d = torch.ops._c10d_functional
            group = dist.group.WORLD.group_name

            def saved_wait_tensors_gn(x, y):
                h = c10d.all_reduce(torch.matmul(x, y), "sum", group)
                g = c10d.all_reduce(torch.matmul(y, x), "sum", group)
                h, g = c10d.wait_tensors([h, g])
                return (h * h.sigmoid() + g).cos()

            save = functools.partial(
                create_selective_checkpoint_contexts, [c10d.all_reduce.default]
            )
            x = torch.randn(8, 8, device=device, requires_grad=True)
            y = torch.randn(8, 8, device=device, requires_grad=True)

            def fn(x, y):
                return torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=False)

            def saving(x, y, gn=saved_gn):
                return torch.utils.checkpoint.checkpoint(
                    gn, x, y, use_reentrant=False, context_fn=save
                )

            for backend in (self._shared_recompute_backend(1), "inductor"):
                torch._dynamo.reset()
                xc, yc = (t.detach().clone().requires_grad_() for t in (x, y))
                self._validate(fn, backend, xc, yc)

            waits = (c10d.wait_tensor.default, c10d.wait_tensors.default)

            def returns_no_wait(gm, args):
                for name, mod in gm.named_children():
                    for o in mod.graph.output_node().args[0]:
                        if o.target is operator.getitem:
                            o = o.args[0]
                        if o.target in waits:
                            raise AssertionError(f"{name} returns a wait:\n{mod.code}")
                return count_shared_recompute_calls(gm, args, 1)

            aot = aot_autograd(
                fw_compiler=returns_no_wait,
                bw_compiler=returns_no_wait,
                partition_fn=min_cut_rematerialization_partition,
            )
            for gn in (saved_gn, saved_wait_tensors_gn):
                for backend in (aot, "inductor"):
                    torch._dynamo.reset()
                    xc, yc = (t.detach().clone().requires_grad_() for t in (x, y))
                    self._validate(functools.partial(saving, gn=gn), backend, xc, yc)
        finally:
            dist.destroy_process_group()

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_split_read_outside(self, device):
        # A pass that strips the split's getitems of their region leaves them
        # outside the piece, reading the split. The split stays inline with
        # them: returned from the piece, they would be views of its buffer,
        # which Inductor would offset twice.
        def strip(graph):
            for n in graph.nodes:
                if n.target is operator.getitem and n.args[0].target in (
                    torch.ops.aten.split.Tensor,
                ):
                    n.meta.pop("ac_graph_id", None)

        def fn(x, w):
            a, b = torch.utils.checkpoint.checkpoint(
                lambda x, w: (x @ w).sin().split(64, dim=-1),
                x,
                w,
                use_reentrant=False,
            )
            c = a.cos()
            return c * b.exp() + c.sum()

        x = torch.randn(64, 128, device=device, requires_grad=True)
        w = torch.randn(128, 128, device=device, requires_grad=True)
        with torch._inductor.config.patch(joint_custom_pre_pass=strip):
            _, codes = run_and_get_code(self._validate, fn, "inductor", x, w)
        for code, side in zip(codes, ("fw", "bw"), strict=True):
            self.assertEqual(len(shared_recompute_code(code, side)), 1)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_multi_output_op_inductor(self, device):
        # SDPA's piece also returns its logsumexp etc., which only the backward
        # reads, and Inductor lowers both calls the same way.
        def fn(q, k, v):
            return torch.utils.checkpoint.checkpoint(
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v).sin(),
                q,
                k,
                v,
                use_reentrant=False,
            )

        args = [torch.randn(2, 4, 64, 32, device=device) for _ in range(3)]
        args = [a.half().requires_grad_() for a in args]
        _, (fw_code, bw_code) = run_and_get_code(self._validate, fn, "inductor", *args)
        fw = shared_recompute_code(fw_code, "fw")
        self.assertEqual(len(fw), 1)
        self.assertEqual(fw, shared_recompute_code(bw_code, "bw"))

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_multi_output_skips(self, device):
        # The default partitioner saves var_mean's variance v, which the next
        # region reads, while recomputing var_mean for its mean. The first
        # region also reads v in a node the backward recomputes, so its piece
        # would both read v and return it: left as partitioned.
        from torch._functorch._aot_autograd import deterministic_recompute

        def first(x, w):
            v, m = torch.var_mean(x @ w, dim=-1, keepdim=True)
            return (x @ w - m).sin() * (v + 1).cos(), v

        def second(a, v, w):
            return (a * torch.rsqrt(v + 1e-5)) @ w

        def fn(x, w, w2):
            ckpt = functools.partial(
                torch.utils.checkpoint.checkpoint, use_reentrant=False
            )
            a, v = ckpt(first, x, w)
            return ckpt(second, a, v, w2).sin()

        args = [
            torch.randn(16, 16, device=device, requires_grad=True) for _ in range(3)
        ]
        backend = aot_autograd(
            fw_compiler=lambda gm, _: gm, bw_compiler=lambda gm, _: gm
        )
        with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
            self._validate(fn, backend, *args)
        self.assertIn("its input getitem depends on its own recompute", logs.output[0])

    @requires_gpu_and_triton
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_user_triton_kernel(self, device):
        # A user Triton kernel's wrapper returns a dict of its results, which a
        # piece can't return. Computed in each graph like a scalar, it would run
        # twice in the forward, which reads one result outside the piece, so the
        # piece is left as partitioned.
        from torch._functorch._aot_autograd import deterministic_recompute

        class TwoOutputs(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y, z = torch.empty_like(x), torch.empty_like(x)
                n = x.numel()
                times_two_and_three_kernel[(triton.cdiv(n, 16),)](
                    x, y, z, n, BLOCK_SIZE=16
                )
                return y, z

            @staticmethod
            def backward(ctx, gy, gz):
                return gy * 2 + gz * 3

        def gn(x, w):
            y, z = TwoOutputs.apply(x @ w)
            return (y.sin() @ w).cos(), z

        def fn(x, w):
            out, z = torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)
            return out + z.tanh()

        args = [
            torch.randn(16, 16, device=device, requires_grad=True) for _ in range(2)
        ]
        with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
            self._validate(fn, self._shared_recompute_backend(0), *args)
        self.assertIn("isn't a tensor or a scalar", logs.output[0])

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_scalar_result(self, device):
        # A custom op returns a tensor and an int, which only its backward reads,
        # a SymInt under dynamic shapes. A piece can't return a scalar (Inductor
        # couldn't lower it), so it is left as partitioned.
        from torch._functorch._aot_autograd import deterministic_recompute

        with torch.library._scoped_library("test_dr_scalar_result", "FRAGMENT"):

            @torch.library.custom_op(
                "test_dr_scalar_result::scale_and_len", mutates_args=()
            )
            def scale_and_len(x: torch.Tensor) -> tuple[torch.Tensor, int]:
                return x * 2, x.shape[0] * 3

            @scale_and_len.register_fake
            def _(x):
                return x * 2, x.shape[0] * 3

            def setup_context(ctx, inputs, output):
                ctx.n = output[1]

            def backward(ctx, gy, gn):
                return gy * 2 + gy.new_zeros(ctx.n).sum()

            scale_and_len.register_autograd(backward, setup_context=setup_context)

            def gn(x, w):
                return scale_and_len((x @ w).sin())[0].cos()

            def piece_reads_it(x, w):
                # The piece reads the int too, sizing a constant it copies in.
                y, n = scale_and_len((x @ w).sin())
                return (y * torch.arange(n, device=x.device).float().sum()).cos()

            for region in (gn, piece_reads_it):
                with self.subTest(region.__name__):
                    torch._dynamo.reset()

                    def fn(x, w, region=region):
                        return torch.utils.checkpoint.checkpoint(
                            region, x, w, use_reentrant=False
                        )

                    args = [
                        torch.randn(8, 8, device=device, requires_grad=True)
                        for _ in range(2)
                    ]
                    ref = [a.detach().clone().requires_grad_() for a in args]
                    fn(*ref).sum().backward()
                    backend = self._shared_recompute_backend(0)
                    log = deterministic_recompute.log
                    with self.assertLogs(log, level="INFO") as logs:
                        compiled = torch.compile(fn, backend=backend, dynamic=True)
                        compiled(*args).sum().backward()
                    self.assertIn("getitem_1 isn't a tensor", logs.output[-1])
                    for a, r in zip(args, ref):
                        self.assertEqual(a.grad, r.grad)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_symint_constant(self, device):
        # The cos table's length h.shape[-1] // 2 is SymInt arithmetic the
        # partitioner recomputes outside the region rather than saves, so the
        # backward has no value of it to pass: the piece copies it in.
        def gn(x, w):
            h = x @ w
            c = (torch.arange(h.shape[-1] // 2, device=x.device).float() / 32).cos()
            a, b = h.chunk(2, -1)
            return torch.cat((a * c - b * c, a * c + b * c), -1).sin()

        def fn(x, w):
            return torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)

        x = torch.randn(16, 16, device=device, requires_grad=True)
        w = torch.randn(16, 16, device=device, requires_grad=True)
        xe, we = (t.detach().clone().requires_grad_() for t in (x, w))
        fn(xe, we).sum().backward()
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        _, (fw_code, bw_code) = run_and_get_code(
            lambda: compiled(x, w).sum().backward()
        )
        self.assertEqual(len(shared_recompute_code(fw_code, "fw")), 1)
        self.assertEqual(len(shared_recompute_code(bw_code, "bw")), 1)
        self.assertEqual(x.grad, xe.grad, atol=1e-3, rtol=1e-3)
        self.assertEqual(w.grad, we.grad, atol=1e-3, rtol=1e-3)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_saved_symint(self, device):
        # The arange's length T // 2 is SymInt arithmetic the partitioner saves,
        # since the backward reshapes a gradient with it. The piece takes it as
        # an operand, and also the SymInt input T, which binds the symbol its
        # sizes use.
        def gn(h0, w, T):
            h = h0 @ w
            c = torch.arange(T // 2, device=h.device).float()
            y = (h.unsqueeze(1) * c.view(1, -1, 1)).sin()
            return y.reshape(h.shape[0], (T // 2) * h.shape[1]).cos()

        def fn(h0, w, t):
            return torch.utils.checkpoint.checkpoint(
                gn, h0, w, t.shape[0], use_reentrant=False
            )

        h0 = torch.randn(4, 8, device=device, requires_grad=True)
        w = torch.randn(8, 8, device=device, requires_grad=True)
        t = torch.randn(6, device=device)
        he, we = (a.detach().clone().requires_grad_() for a in (h0, w))
        fn(he, we, t).sum().backward()
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        _, (fw_code, bw_code) = run_and_get_code(
            lambda: compiled(h0, w, t).sum().backward()
        )
        self.assertEqual(len(shared_recompute_code(fw_code, "fw")), 1)
        self.assertEqual(len(shared_recompute_code(bw_code, "bw")), 1)
        self.assertEqual(h0.grad, he.grad, atol=1e-3, rtol=1e-3)
        self.assertEqual(w.grad, we.grad, atol=1e-3, rtol=1e-3)

    @parametrize("constant", [False, True])
    @torch._functorch.config.patch(
        deterministic_recompute=True, activation_memory_budget=0.1
    )
    def test_deterministic_recompute_reads_separate_recompute(self, device, constant):
        # Under the budget, the partitioner recomputes y = x.cos() outside the
        # region instead of saving it, so each graph computes the region's input
        # on its own. The piece must read it the same way in both (the backend
        # checks): the piece recomputes y itself, with the tensor constant (a
        # graph attribute) it may read.
        def fn(x, w):
            if constant:
                x = x * torch.tensor([1.0, 2.0, 3.0, 4.0] * 2, device=x.device)
            y = x.cos()
            return torch.utils.checkpoint.checkpoint(
                lambda y: (y @ w).sin(), y, use_reentrant=False
            )

        x = torch.randn(8, 8, device=device, requires_grad=True)
        w = torch.randn(8, 8, device=device, requires_grad=True)
        self._validate(fn, self._shared_recompute_backend(1), x, w)

    @torch._functorch.config.patch(
        deterministic_recompute=True, activation_memory_budget=0.1
    )
    def test_deterministic_recompute_replays_only_cheap_inputs(self, device):
        # Under the budget, the partitioner recomputes the MLP outside each
        # region, so each graph computes the region's input on its own. A piece
        # would recompute it itself, moving the MLP's matmuls into the piece,
        # but only cheap inputs are recomputed so: it's left as partitioned.
        from torch._functorch._aot_autograd import deterministic_recompute

        def fn(x, w1, w2, w):
            h = x
            for _ in range(4):
                h = h + (h @ w1).relu() @ w2
                h = h + torch.utils.checkpoint.checkpoint(
                    lambda h: (h @ w).sin(), h, use_reentrant=False
                )
            return h

        args = [
            torch.randn(64, 64, device=device, requires_grad=True) for _ in range(4)
        ]
        backend = self._shared_recompute_backend(0, fw_mms=12, bw_mms=35)
        with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
            self._validate(fn, backend, *args)
        self.assertIn("is computed in each graph", logs.output[0])

    @torch._functorch.config.patch(activation_memory_budget=0.1)
    def test_deterministic_recompute_returns_replayed_inputs(self, device):
        # Under the budget, the partitioner recomputes the residual adds outside
        # the regions, so each region's piece recomputes the add it reads. It
        # returns it, since the next add reads it too, so the next piece reads
        # it rather than replaying the residual stream again: each graph runs as
        # many adds as with the flag off.
        def fn(x, w):
            h = x
            for _ in range(8):
                h = h + torch.utils.checkpoint.checkpoint(
                    lambda h: (h @ w).sin(), h, use_reentrant=False
                )
            return h

        def count(gm, args, counts, flag):
            count_shared_recompute_calls(gm, args, 8 if flag else 0)
            add = torch.ops.aten.add.Tensor
            mods = [m for m in gm.modules() if isinstance(m, torch.fx.GraphModule)]
            counts.append(sum(n.target is add for m in mods for n in m.graph.nodes))
            return gm

        adds = {}
        for flag in (False, True):
            torch._dynamo.reset()
            check = functools.partial(
                count, counts=adds.setdefault(flag, []), flag=flag
            )
            backend = aot_autograd(
                fw_compiler=check,
                bw_compiler=check,
                partition_fn=min_cut_rematerialization_partition,
            )
            args = [
                torch.randn(64, 64, device=device, requires_grad=True) for _ in range(2)
            ]
            with torch._functorch.config.patch(deterministic_recompute=flag):
                self._validate(fn, backend, *args)
        self.assertEqual(adds[True], adds[False])

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_nested_list_result(self, device):
        # A custom op returns a tensor and a list of tensors. The piece reads one
        # element of the list and the next region reads the other, so the
        # list's getitem, a value a piece can't return, goes in the piece and its
        # element is returned.
        with torch.library._scoped_library("test_dr_nested_list", "FRAGMENT"):

            @torch.library.custom_op("test_dr_nested_list::f", mutates_args=())
            def f(x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
                return x * 2, [x.sin(), x.cos()]

            @f.register_fake
            def _(x):
                return x * 2, [x.sin(), x.cos()]

            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(inputs[0])

            def backward(ctx, ga, gl):
                (x,) = ctx.saved_tensors
                g = ga * 2
                if gl[0] is not None:
                    g = g + gl[0] * x.cos()
                if gl[1] is not None:
                    g = g - gl[1] * x.sin()
                return g

            f.register_autograd(backward, setup_context=setup_context)

            def first(x, w):
                a, (b, c) = f(x @ w)
                return a.sin() + b.cos(), c

            def fn(x, w, w2):
                ckpt = functools.partial(
                    torch.utils.checkpoint.checkpoint, use_reentrant=False
                )
                o, c = ckpt(first, x, w)
                return ckpt(lambda o, c: (o * c.exp()) @ w2, o, c).sin()

            check = functools.partial(count_shared_recompute_calls, expected=2)
            backend = aot_autograd(fw_compiler=check, bw_compiler=check)
            args = [
                torch.randn(8, 8, device=device, requires_grad=True) for _ in range(3)
            ]
            self._validate(fn, backend, *args)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_saved_tensor_hooks(self, device):
        # The default partitioner saves the first region's var_mean results,
        # which the second region reads. The shared call takes over var_mean but
        # keeps its getitems, since graph saved-tensor hooks pair each saved
        # forward output with its backward placeholder by name.
        def first(x, w):
            v, m = torch.var_mean(x @ w, dim=-1, keepdim=True)
            return (x @ w - m).sin(), v

        def second(a, v, w):
            return (a * torch.rsqrt(v + 1e-5)) @ w

        def fn(x, w, w2):
            ckpt = functools.partial(
                torch.utils.checkpoint.checkpoint, use_reentrant=False
            )
            a, v = ckpt(first, x, w)
            return ckpt(second, a, v, w2).sin()

        check = functools.partial(count_shared_recompute_calls, expected=2)
        backend = aot_autograd(fw_compiler=check, bw_compiler=check)
        pack = torch.fx.symbolic_trace(lambda t: t * 2)
        unpack = torch.fx.symbolic_trace(lambda t: t / 2)
        args = [
            torch.randn(16, 16, device=device, requires_grad=True) for _ in range(3)
        ]
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            self._validate(fn, backend, *args)

    @parametrize(
        "op",
        [
            "index_add",
            "index_reduce",
            "index_put",
            "scatter_add",
            "histc",
            "cumsum",
            "int_scatter_add",
            "index_put_overwrite",
            "scatter_amax",
        ],
    )
    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_skips_atomics(self, device, op):
        # Floating-point accumulation in an order that can change run to run
        # (atomics over repeated indices, histc's atomic counts, a scan) can
        # differ between the forward and the recompute however each lowers, so
        # the region is left as partitioned and the log says why. Integer
        # accumulation is exact in any order, and overwriting or taking the max
        # doesn't accumulate: those regions are shared.
        from torch._functorch._aot_autograd import deterministic_recompute

        rep = torch.tensor([0, 1, 0, 1], device=device)
        uniq = torch.tensor([0, 1, 2, 3], device=device)
        rep2d = torch.tensor([[0] * 8, [1] * 8] * 2, device=device)
        ones = torch.ones(1, 4, dtype=torch.long, device=device)

        def full(x, value, rows=2):
            return torch.full((rows, 8), value, device=x.device)

        def int_scatter_add(x):
            counts = full(x, 0, rows=1).long().scatter_add(1, rep[None], ones)
            return x * counts.sum()

        gn, shared = {
            "index_add": (lambda x: full(x, 0.0).index_add(0, rep, x), 0),
            "index_reduce": (
                lambda x: full(x, 1.0).index_reduce(0, rep, x, "prod"),
                0,
            ),
            "index_put": (
                lambda x: full(x, 0.0).index_put((rep,), x, accumulate=True),
                0,
            ),
            "scatter_add": (lambda x: full(x, 0.0).scatter_add(0, rep2d, x), 0),
            "histc": (lambda x: x * torch.histc(x * 2, bins=4).max(), 0),
            "cumsum": (lambda x: x.cumsum(0), 0),
            "int_scatter_add": (int_scatter_add, 1),
            "index_put_overwrite": (
                lambda x: full(x, 0.0, rows=4).index_put((uniq,), x),
                1,
            ),
            "scatter_amax": (
                lambda x: full(x, 0.0).scatter_reduce(0, rep2d, x, "amax"),
                1,
            ),
        }[op]

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                lambda x: gn(x).sin(), x, use_reentrant=False
            )

        x = torch.randn(4, 8, device=device, requires_grad=True)
        backend = self._shared_recompute_backend(shared)
        if shared:
            self._validate(fn, backend, x)
            return
        with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
            self._validate(fn, backend, x)
        expected = "accumulates in an order that can change"
        self.assertTrue(any(expected in line for line in logs.output), logs.output)

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_keeps_memory_budget(self, device):
        # region_activation_memory_budget requires its annotation on every op
        # of a nested_compile_region's body, so the calls replacing the
        # checkpoint's recompute there carry it too, and so do the getitems of
        # layer_norm's results that the piece returns; in norm_only, those are
        # all the piece returns.
        def body(x, w):
            return F.layer_norm((x @ w).sin(), (w.shape[-1],)).cos()

        def norm_only(x, w):
            return F.layer_norm(x, (w.shape[-1],))

        for gn in (body, norm_only):
            with self.subTest(gn.__name__):
                torch._dynamo.reset()

                @torch.compiler.nested_compile_region
                def layer(x, w, gn=gn):
                    with torch.autograd.graph.region_activation_memory_budget(0.5):
                        return torch.utils.checkpoint.checkpoint(
                            gn, x, w, use_reentrant=False
                        )

                def fn(x, w1, w2, layer=layer):
                    with torch.autograd.graph.region_activation_memory_budget(0.5):
                        return layer(layer(x, w1), w2)

                args = [
                    torch.randn(16, 16, device=device, requires_grad=True)
                    for _ in range(3)
                ]
                ref = [a.detach().clone().requires_grad_() for a in args]
                gn(gn(ref[0], ref[1]), ref[2]).sum().backward()
                pieces = []

                def count(gm, args, pieces=pieces):
                    hop = torch.ops.higher_order.invoke_subgraph
                    for m in gm.modules():
                        calls = m.graph.find_nodes(op="call_function", target=hop)
                        pieces.extend(
                            n.args[1] for n in calls if "recompute_" in n.args[1]
                        )
                    return gm

                backend = aot_autograd(
                    fw_compiler=count,
                    bw_compiler=count,
                    partition_fn=min_cut_rematerialization_partition,
                )
                compiled = torch.compile(fn, backend=backend, fullgraph=True)
                compiled(*args).sum().backward()
                expected = {
                    "subgraph_0_0_recompute_0_fw",
                    "subgraph_0_0_recompute_0_bw",
                }
                self.assertEqual(set(pieces), expected)
                for a, r in zip(args, ref):
                    self.assertEqual(a.grad, r.grad, atol=1e-5, rtol=1e-5)


class ActivationCheckpointingSharedModuleTests(torch._dynamo.test_case.TestCase):
    """Checkpointing the same module at two sibling call sites. See
    https://github.com/pytorch/pytorch/issues/193194."""

    def test_sac_with_bound_method_context_fn(self):
        # A bound method is a valid context_fn; extracting the underlying
        # function instead would call it with the receiver missing.
        class Ctxs:
            def make(self):
                return create_selective_checkpoint_contexts([torch.ops.aten.mm.default])

        ctxs = Ctxs()

        def f(x, y):
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                f, x, y, use_reentrant=False, context_fn=ctxs.make
            )

        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        a = torch.randn(4, 4, requires_grad=True, device="cpu")
        b = torch.randn(4, 4, requires_grad=True, device="cpu")
        self.assertEqual(opt_fn(a, b), fn(a, b))

    def test_sac_partial_context_fn_with_tensor_arg_graph_breaks(self):
        # A partial bound to a tensor cannot be resolved to a Python callable at
        # trace time. That is a graph break, not an internal error.
        def make(t):
            return create_selective_checkpoint_contexts([torch.ops.aten.mm.default])

        def f(x):
            return torch.sin(x) @ torch.eye(3)

        def fn(x):
            return checkpoint(
                f, x, use_reentrant=False, context_fn=functools.partial(make, x)
            )

        x = torch.ones(3, 3, requires_grad=True)
        cnt = CompileCounterWithBackend("aot_eager")
        self.assertEqual(torch.compile(fn, backend=cnt)(x), fn(x))
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Not a Python constant"
        ):
            torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

    def test_sac_bound_method_context_fn_nonconstant_receiver_graph_breaks(self):
        # The receiver is built inside the region, so the method cannot be bound
        # to a real object; that is a graph break naming context_fn.
        class Ctxs:
            def make(self):
                return create_selective_checkpoint_contexts([torch.ops.aten.mm.default])

        def f(x):
            return torch.sin(x) @ torch.eye(3)

        def fn(x):
            return checkpoint(f, x, use_reentrant=False, context_fn=Ctxs().make)

        x = torch.ones(3, 3, requires_grad=True)
        cnt = CompileCounterWithBackend("aot_eager")
        self.assertEqual(torch.compile(fn, backend=cnt)(x), fn(x))
        # The break precedes any op, so the frame is skipped; start clean.
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "bound to a non-constant receiver"
        ):
            torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

    def test_dynamic_shape_checkpoint_shared_module_two_call_sites(self):
        # An unspecialized plain-float module attribute (self.eps), read
        # inside a torch.utils.checkpoint region that's entered from two
        # sibling call sites under dynamic shapes, used to hard crash with
        # AssertionError: lift_tracked_freevar_to_input should not be called
        # on root SubgraphTracer.
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.eps = 1e-6

            def forward(self, x, extra):
                out = x * self.eps
                if extra:
                    out = out + x
                return out

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()

            def forward(self, x):
                out1 = torch.utils.checkpoint.checkpoint(
                    self.block, x, False, use_reentrant=False
                )
                out2 = torch.utils.checkpoint.checkpoint(
                    self.block, x, True, use_reentrant=False
                )
                return out1 + out2

        model = Model()
        cnt = CompileCounterWithBackend("aot_eager")
        # dynamic=True is what unspecializes self.eps (via wrap_symfloat)
        # rather than specializing it to a constant; it also happens to
        # cover the two different sequence lengths below with one compile.
        compiled_model = torch.compile(model, backend=cnt, dynamic=True, fullgraph=True)

        for seq_len in (8, 16):
            x = torch.randn(2, seq_len, requires_grad=True)
            expected = model(x)
            result = compiled_model(x)
            self.assertEqual(result, expected)
            # Exercise the AC joint-graph/recompute boundary, where this
            # issue was originally reported.
            result.sum().backward()

        # Confirms dynamic shapes were actually exercised: one compile
        # covering both sequence lengths, not a silent recompile.
        self.assertEqual(cnt.frame_count, 1)


class RematerializeACNodesPassTests(torch._dynamo.test_case.TestCase):
    """Tests for AC reordering optimization in full graph (forward+backward in one graph)."""

    def count_op(self, gm, target):
        return sum(1 for n in gm.graph.nodes if n.target == target)

    def _compile_and_capture(self, fn, remat_using_tags_for_fwd_loss_bwd_graph, inputs):
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

        with (
            torch._functorch.config.patch(
                remat_using_tags_for_fwd_loss_bwd_graph=remat_using_tags_for_fwd_loss_bwd_graph
            ),
            torch._dynamo.config.patch(trace_autograd_ops=True),
        ):
            compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
            result = compiled_fn(*inputs)

        return result, captured_gm

    @unittest.skipIf(not HAS_GPU_AND_TRITON, "GPU not available")
    def test_ac_rematerialize_simple_forward_backward(self):
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        def simple_fwd_bwd(x, y):
            z = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sigmoid(torch.matmul(a, b)),
                x,
                y,
                use_reentrant=False,
            )
            loss = z.sum()

            dx, dy = _grad(loss, (x, y))

            return dx.detach(), dy.detach()

        (dx1, dy1), gm_without = self._compile_and_capture(
            simple_fwd_bwd, False, (x, y)
        )
        (dx2, dy2), gm_with = self._compile_and_capture(simple_fwd_bwd, True, (x, y))

        self.assertTrue(torch.allclose(dx1, dx2))
        self.assertTrue(torch.allclose(dy1, dy2))

        mm_with = self.count_op(gm_with, torch.ops.aten.mm.default)
        mm_without = self.count_op(gm_without, torch.ops.aten.mm.default)
        sigmoid_with = self.count_op(gm_with, torch.ops.aten.sigmoid.default)
        sigmoid_without = self.count_op(gm_without, torch.ops.aten.sigmoid.default)
        self.assertEqual(mm_with, 4, "mm should be recomputed in backward")
        self.assertEqual(mm_without, 3)
        self.assertEqual(sigmoid_with, 2, "sigmoid should be recomputed in backward")
        self.assertEqual(sigmoid_without, 1)

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
    detach_2_recomputed = torch.ops.aten.detach.default(sigmoid_recomputed);  sigmoid_recomputed = None
    detach_4 = torch.ops.aten.detach.default(detach_2_recomputed);  detach_2_recomputed = None
    sigmoid_backward = torch.ops.aten.sigmoid_backward.default(expand, detach_4);  expand = detach_4 = None
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    mm_2 = torch.ops.aten.mm.default(t, sigmoid_backward);  t = None
    t_1 = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_3 = torch.ops.aten.mm.default(sigmoid_backward, t_1);  sigmoid_backward = t_1 = None
    detach_5 = torch.ops.aten.detach.default(mm_3);  mm_3 = None
    detach_6 = torch.ops.aten.detach.default(mm_2);  mm_2 = None
    return (detach_5, detach_6)""",
        )

    def test_ac_rematerialize_with_rng_ops_raises_error(self):
        x = torch.randn(4, 4, requires_grad=True)

        def fwd_bwd_with_rng(x):
            z = torch.utils.checkpoint.checkpoint(
                lambda a: torch.sigmoid(a + torch.rand_like(a)), x, use_reentrant=False
            )
            loss = z.sum()

            dx = _grad(loss, x)[0]

            return dx

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Activation checkpoint rematerialization in `forward-loss-backward` graph does not support RNG ops in recompute regions.",
        ):
            self._compile_and_capture(fwd_bwd_with_rng, True, (x,))

    def test_ac_rematerialize_sdpa_rng_classification(self):
        sdpa_ops = [
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten._scaled_dot_product_attention_math_for_mps.default,
            torch.ops.aten._scaled_dot_product_cudnn_attention.default,
            torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.quantized,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
            torch.ops.aten._flash_attention_forward.default,
            torch.ops.aten._flash_attention_forward.quantized,
            torch.ops.aten._flash_attention_forward_no_dropout_inplace.default,
            torch.ops.aten._efficient_attention_forward.default,
            torch.ops.aten._cudnn_attention_forward.default,
        ]
        shape_env = ShapeEnv()
        zero_symbol = shape_env.create_symbol(
            0.0,
            source=ConstantSource("dropout_zero"),
            dynamic_dim=DimDynamic.DUCK,
            constraint_dim=None,
        )
        nonzero_symbol = shape_env.create_symbol(
            0.1,
            source=ConstantSource("dropout_nonzero"),
            dynamic_dim=DimDynamic.DUCK,
            constraint_dim=None,
        )
        symfloat_zero = torch.SymFloat(SymNode(zero_symbol, shape_env, float, hint=0.0))
        symfloat_nonzero = torch.SymFloat(
            SymNode(nonzero_symbol, shape_env, float, hint=0.1)
        )
        dropout_cases = [
            (0, False),
            (0.0, False),
            (-0.0, False),
            (1e-12, True),
            (0.1, True),
            (float("nan"), True),
        ]

        for op in sdpa_ops:
            dropout_arg_idx = next(
                idx
                for idx, arg in enumerate(op._schema.arguments)
                if arg.name == "dropout_p"
            )
            for dropout_p, expected in dropout_cases:
                with self.subTest(op=op, dropout_p=dropout_p):
                    graph = torch.fx.Graph()
                    q = graph.placeholder("q")
                    args = [q] * (dropout_arg_idx + 1)
                    args[dropout_arg_idx] = dropout_p
                    node = graph.call_function(op, tuple(args), {})
                    node.meta["recompute"] = (
                        torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
                    )
                    graph.output(node)
                    gm = torch.fx.GraphModule({}, graph)

                    self.assertEqual(has_recomputable_rng_ops(gm), expected)

            with self.subTest(op=op, dropout_p="dynamic"):
                graph = torch.fx.Graph()
                q = graph.placeholder("q")
                dropout_p = graph.placeholder("dropout_p")
                args = [q] * (dropout_arg_idx + 1)
                args[dropout_arg_idx] = dropout_p
                node = graph.call_function(op, tuple(args), {})
                node.meta["recompute"] = (
                    torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
                )
                graph.output(node)
                gm = torch.fx.GraphModule({}, graph)

                self.assertTrue(has_recomputable_rng_ops(gm))

            for dropout_p, expected in dropout_cases:
                with self.subTest(op=op, kwarg_dropout_p=dropout_p):
                    graph = torch.fx.Graph()
                    q = graph.placeholder("q")
                    node = graph.call_function(
                        op, tuple([q] * dropout_arg_idx), {"dropout_p": dropout_p}
                    )
                    node.meta["recompute"] = (
                        torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
                    )
                    graph.output(node)
                    gm = torch.fx.GraphModule({}, graph)

                    self.assertEqual(has_recomputable_rng_ops(gm), expected)

            for dropout_p, expected in [
                (symfloat_zero, False),
                (symfloat_nonzero, True),
            ]:
                with self.subTest(op=op, symfloat_dropout_p=dropout_p):
                    args = [None] * (dropout_arg_idx + 1)
                    args[dropout_arg_idx] = dropout_p
                    node = SimpleNamespace(target=op, args=tuple(args), kwargs={})

                    self.assertEqual(is_rng_op(node), expected)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ac_rematerialize_with_sdpa_dropout_zero(self):
        from torch.nn.attention import sdpa_kernel, SDPBackend

        cases = []
        if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
            cases.append((SDPBackend.EFFICIENT_ATTENTION, torch.float32))
        if PLATFORM_SUPPORTS_FLASH_ATTENTION:
            cases.append((SDPBackend.FLASH_ATTENTION, torch.float16))
        if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
            cases.append((SDPBackend.CUDNN_ATTENTION, torch.float16))
        if not cases:
            self.skipTest("No fused SDPA backends available")
        sdpa_ops = {
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten._scaled_dot_product_cudnn_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        }

        def policy_fn(ctx, op, *args, **kwargs):
            if op in sdpa_ops:
                return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_SAVE

        context_fn = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
        )

        for backend, dtype in cases:
            with self.subTest(backend=backend, dtype=dtype):
                torch._dynamo.reset()
                q = torch.randn(
                    2, 4, 128, 64, device="cuda", dtype=dtype, requires_grad=True
                )
                k = torch.randn(
                    2, 4, 128, 64, device="cuda", dtype=dtype, requires_grad=True
                )
                v = torch.randn(
                    2, 4, 128, 64, device="cuda", dtype=dtype, requires_grad=True
                )

                def fwd_bwd_with_sdpa(q, k, v):
                    with sdpa_kernel(backend):
                        z = torch.utils.checkpoint.checkpoint(
                            lambda q, k, v: F.scaled_dot_product_attention(
                                q, k, v, dropout_p=0.0
                            ),
                            q,
                            k,
                            v,
                            use_reentrant=False,
                            context_fn=context_fn,
                        )
                        loss = z.sum()
                        dq, dk, dv = _grad(loss, (q, k, v))

                    return z.detach(), dq, dk, dv

                result_with, gm_with = self._compile_and_capture(
                    fwd_bwd_with_sdpa, True, (q, k, v)
                )
                torch._dynamo.reset()
                result_without, _ = self._compile_and_capture(
                    fwd_bwd_with_sdpa, False, (q, k, v)
                )
                eager_inputs = tuple(
                    t.detach().clone().requires_grad_(True) for t in (q, k, v)
                )
                result_eager = fwd_bwd_with_sdpa(*eager_inputs)

                for actual, expected in zip(result_with, result_without):
                    self.assertEqual(actual, expected)
                for actual, expected in zip(result_with, result_eager):
                    self.assertEqual(actual, expected)
                self.assertEqual(sum(self.count_op(gm_with, op) for op in sdpa_ops), 2)

    def test_ac_rematerialize_with_no_annotations(self):
        x = torch.randn(4, 4, requires_grad=True)

        def fwd_bwd(x):
            z = torch.utils.checkpoint.checkpoint(
                lambda a: torch.sigmoid(a + 4), x, use_reentrant=False
            )
            loss = z.sum()
            return _grad(loss, x)[0]

        result_with, gm_with = self._compile_and_capture(fwd_bwd, True, (x,))
        result_without, gm_without = self._compile_and_capture(fwd_bwd, False, (x,))

        self.assertTrue(torch.allclose(result_with, result_without))

        # autograd_backward tagging is automatic now, so remat should still work
        sigmoid_with = self.count_op(gm_with, torch.ops.aten.sigmoid.default)
        sigmoid_without = self.count_op(gm_without, torch.ops.aten.sigmoid.default)
        self.assertEqual(sigmoid_with, 2, "sigmoid should be recomputed in backward")
        self.assertEqual(sigmoid_without, 1)

    def test_ac_rematerialize_with_selective_checkpoint_policy(self):
        x = torch.randn(4, 128, requires_grad=True)
        w1 = torch.randn(128, 128, requires_grad=True)
        b1 = torch.randn(128, requires_grad=True)

        def policy_fn(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.addmm.default:
                return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

        context_fn = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
        )

        def fwd_bwd_with_policy(x, w1, b1):
            def checkpoint_fn(inp, w, b):
                linear = torch.nn.functional.linear(inp, w, b)
                return torch.relu(linear)

            result = torch.utils.checkpoint.checkpoint(
                checkpoint_fn, x, w1, b1, use_reentrant=False, context_fn=context_fn
            )
            loss = result.sum()

            dx, dw, db = _grad(loss, (x, w1, b1))
            return dx, dw, db

        result_with, gm_with = self._compile_and_capture(
            fwd_bwd_with_policy, True, (x, w1, b1)
        )
        result_without, gm_without = self._compile_and_capture(
            fwd_bwd_with_policy, False, (x, w1, b1)
        )

        torch.testing.assert_close(result_with[0], result_without[0])
        torch.testing.assert_close(result_with[1], result_without[1])
        torch.testing.assert_close(result_with[2], result_without[2])

        addmm_without = self.count_op(gm_without, torch.ops.aten.addmm.default)
        relu_without = self.count_op(gm_without, torch.ops.aten.relu.default)

        addmm_with = self.count_op(gm_with, torch.ops.aten.addmm.default)
        relu_with = self.count_op(gm_with, torch.ops.aten.relu.default)

        self.assertEqual(addmm_without, addmm_with)
        self.assertEqual(relu_with, relu_without + 1)

        recomputed_nodes = [
            n.name for n in gm_with.graph.nodes if "_recomputed" in n.name
        ]
        self.assertNotIn("addmm_recomputed", recomputed_nodes)

        self.assertTrue(
            any("relu" in name for name in recomputed_nodes),
            lambda msg: f"{msg}\nExpected relu_recomputed but got: {recomputed_nodes}",
        )

    def _compile_with_joint_graph_pass_and_capture(self, fn, inputs):
        from torch._inductor.fx_passes.joint_graph import joint_graph_passes

        captured_gm_before = None
        captured_gm_after = None

        def custom_compiler(gm, example_inputs):
            nonlocal captured_gm_before, captured_gm_after
            import copy

            captured_gm_before = copy.deepcopy(gm)
            joint_graph_passes(gm)
            captured_gm_after = gm
            return gm.forward

        backend = aot_autograd(
            fw_compiler=custom_compiler,
            bw_compiler=None,
            partition_fn=None,
        )

        with torch._dynamo.config.patch(trace_autograd_ops=True):
            compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
            result = compiled_fn(*inputs)

        return result, captured_gm_before, captured_gm_after

    def test_joint_graph_passes_view_optimization(self):
        x = torch.randn(4, 4, requires_grad=True)

        def fwd_bwd_with_views(x):
            def checkpoint_fn(a):
                b = a.view(16)
                c = b.view(4, 4)
                return torch.sigmoid(c)

            z = torch.utils.checkpoint.checkpoint(
                checkpoint_fn,
                x,
                use_reentrant=False,
            )
            loss = z.sum()

            dx = _grad(loss, x)[0]

            return dx.detach()

        result, gm_before, gm_after = self._compile_with_joint_graph_pass_and_capture(
            fwd_bwd_with_views, (x,)
        )

        result_eager = torch.autograd.grad(torch.sigmoid(x).sum(), x)[0]
        self.assertTrue(torch.allclose(result, result_eager, atol=1e-5))

        view_count_before = self.count_op(gm_before, torch.ops.aten.view.default)
        view_count_after = self.count_op(gm_after, torch.ops.aten.view.default)
        self.assertTrue(view_count_after == 0)
        self.assertTrue(view_count_before == 6)

        self.assertExpectedInline(
            gm_after.code.strip(),
            """\
def forward(self, arg0_1):
    sigmoid = torch.ops.aten.sigmoid.default(arg0_1)
    sum_1 = torch.ops.aten.sum.default(sigmoid);  sigmoid = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    sigmoid_recomputed = torch.ops.aten.sigmoid.default(arg0_1);  arg0_1 = None
    detach_3_recomputed = torch.ops.aten.detach.default(sigmoid_recomputed);  sigmoid_recomputed = None
    detach_5 = torch.ops.aten.detach.default(detach_3_recomputed);  detach_3_recomputed = None
    sigmoid_backward = torch.ops.aten.sigmoid_backward.default(expand, detach_5);  expand = detach_5 = None
    detach_6 = torch.ops.aten.detach.default(sigmoid_backward);  sigmoid_backward = None
    return (detach_6,)""",
        )

    def test_joint_graph_passes_permute_optimization(self):
        x = torch.randn(4, 4, requires_grad=True)

        def fwd_bwd_with_permute(x):
            def checkpoint_fn(a):
                b = a.permute(1, 0)
                c = b.permute(1, 0)
                return torch.sigmoid(c)

            z = torch.utils.checkpoint.checkpoint(
                checkpoint_fn,
                x,
                use_reentrant=False,
            )
            loss = z.sum()

            dx = _grad(loss, x)[0]

            return dx.detach()

        result, gm_before, gm_after = self._compile_with_joint_graph_pass_and_capture(
            fwd_bwd_with_permute, (x,)
        )

        result_eager = torch.autograd.grad(torch.sigmoid(x).sum(), x)[0]
        self.assertTrue(torch.allclose(result, result_eager, atol=1e-5))

        permute_count_before = self.count_op(gm_before, torch.ops.aten.permute.default)
        permute_count_after = self.count_op(gm_after, torch.ops.aten.permute.default)
        self.assertTrue(permute_count_after == 0)
        self.assertTrue(permute_count_before == 6)

        self.assertExpectedInline(
            gm_after.code.strip(),
            """\
def forward(self, arg0_1):
    sigmoid = torch.ops.aten.sigmoid.default(arg0_1)
    sum_1 = torch.ops.aten.sum.default(sigmoid);  sigmoid = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    sigmoid_recomputed = torch.ops.aten.sigmoid.default(arg0_1);  arg0_1 = None
    detach_3_recomputed = torch.ops.aten.detach.default(sigmoid_recomputed);  sigmoid_recomputed = None
    detach_5 = torch.ops.aten.detach.default(detach_3_recomputed);  detach_3_recomputed = None
    sigmoid_backward = torch.ops.aten.sigmoid_backward.default(expand, detach_5);  expand = detach_5 = None
    detach_6 = torch.ops.aten.detach.default(sigmoid_backward);  sigmoid_backward = None
    return (detach_6,)""",
        )

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_attr_compile_submodules_in_checkpoint_wrapper(self):
        """Compiling submodules inside a checkpointed block should not hit the
        recompile limit due to WeakKeyDictionary guards in the pack_hook."""
        from torch.utils.checkpoint import checkpoint

        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm1 = nn.RMSNorm(dim)
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.norm2 = nn.RMSNorm(dim)
                self.linear2 = nn.Linear(dim, dim, bias=False)
                self.norm3 = nn.RMSNorm(dim)
                self.linear3 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = x + self.linear1(self.norm1(x))
                x = x + self.linear2(self.norm2(x))
                x = x + self.linear3(self.norm3(x))
                return x

        class CheckpointedBlock(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block

            def forward(self, x):
                return checkpoint(self.block, x, use_reentrant=False)

        dim = 32
        block = Block(dim)

        x_ref = torch.randn(4, dim, requires_grad=True)
        ref = block(x_ref)
        ref.sum().backward()

        block_cp = Block(dim)
        block_cp.load_state_dict(block.state_dict())
        wrapped = CheckpointedBlock(block_cp)

        for _, submod in wrapped.block.named_children():
            submod.compile(backend="aot_eager")

        with torch._dynamo.config.patch(recompile_limit=2):
            x_test = x_ref.detach().clone().requires_grad_(True)
            result = wrapped(x_test)
            result.sum().backward()

        self.assertEqual(ref, result)
        self.assertEqual(x_ref.grad, x_test.grad)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_multiple_user_phase_annotations_errors(self):
        x = torch.randn(4, 4, requires_grad=True)
        w = torch.randn(4, 4, requires_grad=True)

        def fn(x, w):
            z = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sin(a @ b), x, w, use_reentrant=False
            )
            loss = z.sum()
            with torch.fx.traceback.annotate({"phase": "backward"}):
                dx, dw = _grad(loss, (x, w))
            # Non-backward computation between two backward annotations
            out = dx + dw
            with torch.fx.traceback.annotate({"phase": "backward"}):
                out = out * 2
            return out.detach()

        with self.assertRaisesRegex(RuntimeError, "backward regions annotated"):
            self._compile_and_capture(fn, True, (x, w))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_user_phase_annotation_with_extra_autograd_grad(self):
        """Only the user-annotated backward region gets rematerialization."""
        x = torch.randn(4, 4, requires_grad=True)
        w1 = torch.randn(4, 4, requires_grad=True)
        w2 = torch.randn(4, 4, requires_grad=True)

        def fn(x, w1, w2):
            z1 = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sin(a @ b), x, w1, use_reentrant=False
            )
            z2 = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sigmoid(a @ b), x, w2, use_reentrant=False
            )
            loss1 = z1.sum()
            loss2 = z2.sum()
            # Only the first backward is annotated
            with torch.fx.traceback.annotate({"phase": "backward"}):
                dx1 = _grad(loss1, (x,))
            # Second backward NOT annotated — should not get remat
            dx2 = _grad(loss2, (x,))
            return (dx1[0] + dx2[0]).detach()

        _, gm_with = self._compile_and_capture(fn, True, (x, w1, w2))

        self.assertExpectedInline(
            gm_with.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    mm = torch.ops.aten.mm.default(arg2_1, arg0_1)
    sin = torch.ops.aten.sin.default(mm);  mm = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    mm_recomputed = torch.ops.aten.mm.default(arg2_1, arg0_1)
    cos = torch.ops.aten.cos.default(mm_recomputed);  mm_recomputed = None
    mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    mm_2 = torch.ops.aten.mm.default(mul, t);  mul = t = None
    mm_3 = torch.ops.aten.mm.default(arg2_1, arg1_1);  arg2_1 = None
    sigmoid = torch.ops.aten.sigmoid.default(mm_3);  mm_3 = None
    detach_4 = torch.ops.aten.detach.default(sigmoid)
    sum_2 = torch.ops.aten.sum.default(sigmoid);  sigmoid = None
    ones_like_1 = torch.ops.aten.ones_like.default(sum_2, pin_memory = False, memory_format = torch.preserve_format);  sum_2 = None
    expand_1 = torch.ops.aten.expand.default(ones_like_1, [4, 4]);  ones_like_1 = None
    detach_5 = torch.ops.aten.detach.default(detach_4);  detach_4 = None
    sigmoid_backward = torch.ops.aten.sigmoid_backward.default(expand_1, detach_5);  expand_1 = detach_5 = None
    t_1 = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_4 = torch.ops.aten.mm.default(sigmoid_backward, t_1);  sigmoid_backward = t_1 = None
    add = torch.ops.aten.add.Tensor(mm_2, mm_4);  mm_2 = mm_4 = None
    detach_6 = torch.ops.aten.detach.default(add);  add = None
    return (detach_6,)""",
        )

    def test_chunked_loss_remat(self):
        """Chunked loss pattern: multiple backward regions from chunk_loss.backward()
        calls, but only the final x.backward() region needs remat. The pass should
        skip the chunk backwards and only rematerialize for the final one."""
        dim = 32
        chunksz = 4

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(dim, dim, bias=False)
                self.l2 = nn.Linear(dim, dim, bias=False)

            def _fn(self, x):
                return self.l2(F.gelu(self.l1(x), approximate="tanh"))

            def forward(self, x):
                return checkpoint(self._fn, x, use_reentrant=False)

        class ChunkedLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()
                self.head = nn.Linear(dim, dim, bias=False)

            def forward(self, x, y):
                x = self.block(x)
                x_detached = x.detach().requires_grad_()
                total = 0
                for start in range(0, x_detached.shape[0], chunksz):
                    end = start + chunksz
                    chunk_loss = (
                        F.mse_loss(
                            self.head(x_detached[start:end]),
                            y[start:end],
                            reduction="sum",
                        )
                        / x_detached.shape[0]
                    )
                    chunk_loss.backward()
                    total = total + chunk_loss.detach()
                x.backward(x_detached.grad)
                return total

        model = ChunkedLoss()
        x = torch.randn(12, dim)
        y = torch.randn(12, dim)

        result_with, gm_with = self._compile_and_capture(model, True, (x, y))
        result_without, gm_without = self._compile_and_capture(model, False, (x, y))

        torch.testing.assert_close(result_with, result_without)

        # Without remat, gelu appears once (forward only).
        # With remat, gelu is duplicated into the backward region.
        gelu_without = self.count_op(gm_without, torch.ops.aten.gelu.default)
        gelu_with = self.count_op(gm_with, torch.ops.aten.gelu.default)
        self.assertEqual(gelu_without, 1)
        self.assertEqual(gelu_with, 2, "gelu should be recomputed in backward")

    def test_two_backward_regions_needing_remat_errors(self):
        """Two independent backward calls that both need recompute should error."""
        dim = 32

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(dim, dim, bias=False)
                self.l2 = nn.Linear(dim, dim, bias=False)

            def _fn(self, x):
                return self.l2(F.gelu(self.l1(x), approximate="tanh"))

            def forward(self, x):
                return checkpoint(self._fn, x, use_reentrant=False)

        class TwoBackwards(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = Block()
                self.block2 = Block()

            def forward(self, x):
                y = self.block1(x)
                z = self.block2(y)
                z.sum().backward(retain_graph=True)
                y.sum().backward()
                return z.detach()

        x = torch.randn(8, dim, requires_grad=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "require recomputation",
        ):
            self._compile_and_capture(TwoBackwards(), True, (x,))


instantiate_device_type_tests(
    ActivationCheckpointingViaTagsTests, globals(), except_for="cpu", allow_xpu=True
)


class DeterministicRecomputeHelperTests(torch._dynamo.test_case.TestCase):
    """The helpers of torch._functorch._aot_autograd.deterministic_recompute, on
    their own."""

    def _partitioned(self, fn, *args):
        """The forward and backward graphs the partitioner makes for fn, and the
        values the forward saves for the backward."""
        graphs = []

        def partition(joint, inputs, *, num_fwd_outputs, **kwargs):
            fw, bw = min_cut_rematerialization_partition(
                joint, inputs, num_fwd_outputs=num_fwd_outputs, **kwargs
            )
            outs = fw.graph.find_nodes(op="output")[0].args[0][num_fwd_outputs:]
            graphs.append((fw, bw, {n for n in outs if isinstance(n, torch.fx.Node)}))
            return fw, bw

        backend = aot_autograd(fw_compiler=nop, bw_compiler=nop, partition_fn=partition)
        torch.compile(fn, backend=backend, fullgraph=True)(*args).sum().backward()
        return graphs[0]

    def test_recomputed_copies(self):
        # Each region's recompute is found by name, op and inputs: whatever of
        # it the backward needs (the first region's cos for the second mm's
        # gradient), except what a policy saves (the second mm).
        from torch._functorch._aot_autograd.deterministic_recompute import (
            _recomputed_copies,
        )

        def gn(x, w):
            return torch.sigmoid(x @ w).cos()

        def fn(x, w):
            y = checkpoint(gn, x, w, use_reentrant=False)
            return checkpoint(
                gn, y, w, use_reentrant=False, context_fn=save_mm_context_fn
            )

        x = torch.randn(4, 4, requires_grad=True)
        w = torch.randn(4, 4, requires_grad=True)
        copies, regions = _recomputed_copies(*self._partitioned(fn, x, w))
        aten = torch.ops.aten
        targets = [[n.target for n in nodes] for nodes in regions.values()]
        first = [aten.mm.default, aten.sigmoid.default, aten.cos.default]
        self.assertEqual(targets, [first, [aten.sigmoid.default]])
        self.assertEqual(copies, {n for nodes in regions.values() for n in nodes})

    def test_recomputed_copies_outside_regions(self):
        # The partitioner also recomputes cheap ops outside any region (the
        # views feeding the checkpoint): copies in no region.
        from torch._functorch._aot_autograd.deterministic_recompute import (
            _recomputed_copies,
        )

        def fn(x, w):
            y = torch.relu(x).view(16).view(4, 4)
            return checkpoint(lambda y: (y @ w).sin(), y, use_reentrant=False)

        x = torch.randn(4, 4, requires_grad=True)
        w = torch.randn(4, 4, requires_grad=True)
        copies, regions = _recomputed_copies(*self._partitioned(fn, x, w))
        in_regions = {n for nodes in regions.values() for n in nodes}
        outside = {n.name for n in copies - in_regions}
        self.assertEqual(outside, {"view", "view_1"})

    @parametrize("op", ["rand_like", "native_dropout"])
    def test_recomputed_copies_logs_mismatch(self, op):
        # On CPU the backward replays a random op from a saved RNG state with a
        # different op, so neither it nor what is computed from it is a copy.
        # The log says so once, where that starts, even when the random op has
        # two readers.
        from torch._functorch._aot_autograd import deterministic_recompute

        def gn(x, w):
            h = torch.sigmoid(x @ w)
            if op == "rand_like":
                return (h * torch.rand_like(x)).sin()
            out, mask = torch.ops.aten.native_dropout(h, 0.5, True)
            return out.sin() * mask

        def fn(x, w):
            return checkpoint(gn, x, w, use_reentrant=False)

        x = torch.randn(4, 4, requires_grad=True)
        w = torch.randn(4, 4, requires_grad=True)
        graphs = self._partitioned(fn, x, w)
        with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
            _, regions = deterministic_recompute._recomputed_copies(*graphs)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("the backward recomputes it differently", logs.output[0])
        aten = torch.ops.aten
        targets = [[n.target for n in nodes] for nodes in regions.values()]
        self.assertEqual(targets, [[aten.mm.default, aten.sigmoid.default]])

    def test_recomputed_copies_checks_op_and_args(self):
        # A backward node with a forward node's name is its copy only if it is
        # the same op on the same inputs (primals, saved values, graph
        # attributes such as h's subgraph, or other copies). The log says once
        # where that fails: at a, which the backward recomputes under another
        # name, rather than at b and c, which read it.
        from torch._functorch._aot_autograd import deterministic_recompute

        def hop(sub, x):
            return x

        def graph(ops):
            g = torch.fx.Graph()
            env = {"x": g.placeholder("x"), "y": g.placeholder("y")}
            env["sub"] = g.get_attr("sub")
            for name, target, args in ops:
                args = tuple(env[a] for a in args)
                env[name] = g.create_node("call_function", target, args, name=name)
            g.output(tuple(env[name] for name, _, _ in ops))
            root = torch.nn.Module()
            root.sub = torch.nn.Module()
            return torch.fx.GraphModule(root, g)

        sin, cos = torch.ops.aten.sin.default, torch.ops.aten.cos.default
        common = [("b", cos, ("a",)), ("c", sin, ("a",)), ("h", hop, ("sub", "x"))]
        fw = graph(
            [("same", sin, ("x",)), ("target", sin, ("x",)), ("args", sin, ("x",))]
            + [("a", sin, ("x",)), *common]
        )
        bw = graph(
            [("same", sin, ("x",)), ("target", cos, ("x",)), ("args", sin, ("y",))]
            + [
                ("a_1", sin, ("x",)),
                *(o if o[0] == "h" else (o[0], o[1], ("a_1",)) for o in common),
            ]
        )
        with self.assertLogs(deterministic_recompute.log, level="INFO") as logs:
            copies, _ = deterministic_recompute._recomputed_copies(fw, bw, set())
        self.assertEqual({n.name for n in copies}, {"same", "h"})
        logged = [re.search(r": (\w+) left", line).group(1) for line in logs.output]
        self.assertEqual(logged, ["target", "args", "a"])

    @unittest.skipIf(not TEST_CUDA, "graphsafe RNG functionalization is CUDA-only")
    def test_recomputed_copies_graphsafe_rng(self):
        # On CUDA the backward replays dropout with graphsafe_run_with_rng_state,
        # under the forward draw's name but from its own RNG state: same name,
        # different inputs, so not a copy.
        from torch._functorch._aot_autograd.deterministic_recompute import (
            _recomputed_copies,
        )

        def fn(x, w):
            return checkpoint(
                lambda x, w: F.dropout(x @ w, p=0.5).sin(), x, w, use_reentrant=False
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        w = torch.randn(4, 4, device="cuda", requires_grad=True)
        copies, regions = _recomputed_copies(*self._partitioned(fn, x, w))
        draw = torch._prims.rng_prims.graphsafe_run_with_rng_state
        self.assertFalse([n for n in copies if n.target is draw])
        targets = [[n.target for n in nodes] for nodes in regions.values()]
        self.assertEqual(targets, [[torch.ops.aten.mm.default]])

    def _outline(self, fn, args, names):
        """Replace the nodes of fn's graph named ``names`` with a call of one
        subgraph computing them, as the pass does in each graph, and check that
        the graph still computes fn. Returns the graph, or the reason the call
        can't be placed."""
        from torch._functorch._aot_autograd.deterministic_recompute import (
            _place,
            _slice_module,
            _splice_slice_call,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        gm = make_fx(fn, tracing_mode="fake")(*args)
        index = {n: i for i, n in enumerate(gm.graph.nodes)}
        nodes = [n for n in gm.graph.nodes if n.name in names]
        inputs = [a for n in nodes for a in n.all_input_nodes if a not in nodes]
        inputs = list(dict.fromkeys(inputs))
        outputs = [n for n in nodes if any(u not in nodes for u in n.users)]
        last_input = max(inputs, key=index.__getitem__)
        point, moved, reason = _place(nodes, set(nodes), inputs, last_input, index)
        if reason is not None:
            return reason
        _splice_slice_call(
            gm,
            _slice_module(gm, inputs, nodes, outputs),
            "piece",
            [n.meta["val"] for n in outputs],
            operands=inputs,
            replace={n.name: n for n in outputs},
            slice_index={n.name: i for i, n in enumerate(outputs)},
            erase_names=set(names),
            insert_after=point,
            move_below=moved,
        )
        gm.graph.lint()
        gm.recompile()
        self.assertEqual(gm(*args), fn(*args))
        return gm

    def _ops(self, gm):
        return [n.target for n in gm.graph.nodes if n.op == "call_function"]

    def test_outline_before_first_reader(self):
        # The call goes right before the first node that reads one of the nodes
        # it replaces, so it computes them no earlier than the first of them is
        # needed.
        from torch._higher_order_ops import invoke_subgraph

        def fn(x, w):
            y = w * 2
            return x.sin().cos() + y

        args = (torch.randn(4), torch.randn(4))
        gm = self._outline(fn, args, {"sin", "cos"})
        aten = torch.ops.aten
        ops = [aten.mul.Tensor, invoke_subgraph, operator.getitem, aten.add.Tensor]
        self.assertEqual(self._ops(gm), ops)

    def test_outline_moves_readers(self):
        # The call needs exp, which comes after mul reads sin, so the call goes
        # after exp and mul moves below it.
        from torch._higher_order_ops import invoke_subgraph

        def fn(x, w):
            a = x.sin()
            return a * 3, a + w.exp()

        args = (torch.randn(4), torch.randn(4))
        gm = self._outline(fn, args, {"sin", "add"})
        aten = torch.ops.aten
        getitem = operator.getitem
        ops = [aten.exp.default, invoke_subgraph, getitem, getitem, aten.mul.Tensor]
        self.assertEqual(self._ops(gm), ops)

    @parametrize("hops", [1, 2])
    def test_outline_rejects_path_back(self, hops):
        # exp (and neg) read sin and are read by add: no single call can replace
        # sin and add.
        def fn(x):
            a = x.sin()
            b = a.exp().neg() if hops == 2 else a.exp()
            return a + b

        reason = self._outline(fn, (torch.randn(4),), {"sin", "add"})
        last = "neg" if hops == 2 else "exp"
        self.assertEqual(reason, f"its input {last} depends on its own recompute")

    def test_outline_moves_view_it_copies(self):
        # A view the call copies in rather than returns (t) reads a replaced
        # node (sin), so it moves below the call, while add, replaced, still
        # reads it above the call until both are erased.
        from torch._functorch._aot_autograd.deterministic_recompute import (
            _place,
            _slice_module,
            _splice_slice_call,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(x, w):
            b = x.sin().t() + 1
            return b * (x @ w)

        args = (torch.randn(4, 4), torch.randn(4, 4))
        gm = make_fx(fn, tracing_mode="fake")(*args)
        n = {n.name: n for n in gm.graph.nodes}
        nodes = [n["sin"], n["t"], n["add"], n["mul"]]
        inputs = [n["x_1"], n["mm"]]
        index = {node: i for i, node in enumerate(gm.graph.nodes)}
        replaced = {n["sin"], n["add"], n["mul"]}
        point, moved, reason = _place(nodes, replaced, inputs, n["mm"], index)
        self.assertEqual((point, moved, reason), (n["mm"], [n["t"]], None))
        _splice_slice_call(
            gm,
            _slice_module(gm, inputs, nodes, [n["mul"]]),
            "piece",
            [n["mul"].meta["val"]],
            operands=inputs,
            replace={"mul": n["mul"]},
            slice_index={"mul": 0},
            erase_names={"sin", "t", "add", "mul"},
            insert_after=point,
            move_below=moved,
        )
        gm.graph.lint()
        gm.recompile()
        self.assertEqual(gm(*args), fn(*args))

    def test_outline_keeps_random_readers(self):
        # Moving rand_like below the call could reorder it with another random
        # op, which changes what each draws.
        def fn(x, w):
            a = x.sin()
            return torch.rand_like(a) * a, a + w.exp()

        reason = self._outline(fn, (torch.randn(4), torch.randn(4)), {"sin", "add"})
        self.assertEqual(reason, "it would move the random op rand_like")

    def test_rng_and_inplace_writes(self):
        # Outlining can't re-run a random op (it would draw again), and has to
        # keep an in-place write in order with the reads and writes of the
        # storage it writes: a write into a view writes its base, and an op with
        # side effects but no write in its schema (resize_storage_bytes_ frees
        # its tensor's storage) writes all of its tensor inputs.
        from torch._functorch._aot_autograd.deterministic_recompute import (
            _is_rng,
            _writes,
        )
        from torch._inductor.fx_utils import get_node_storage
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(x, y, z):
            x[0].add_(y[0].sin())
            torch.ops.inductor.resize_storage_bytes_(z, 0)
            _, r = torch._prims.rng_prims.run_and_save_rng_state(
                torch.ops.aten.rand_like.default, y
            )
            return torch.rand_like(y) + r

        args = [torch.randn(4, 4) for _ in range(3)]
        gm = make_fx(fn, tracing_mode="fake")(*args)
        nodes = {n.name: n for n in gm.graph.nodes}
        rng = [n.name for n in gm.graph.nodes if _is_rng(n)]
        self.assertEqual(rng, ["run_and_save_rng_state", "rand_like"])
        writes = {n.name: _writes(n) for n in gm.graph.nodes if _writes(n)}
        storage = {name: get_node_storage(nodes[name]) for name in ("x_1", "z_1")}
        expected = {"add_": {storage["x_1"]}, "resize_storage_bytes_": {storage["z_1"]}}
        self.assertEqual(writes, expected)

    def test_unbound_symbols(self):
        # Inductor binds a symbol in a subgraph from a SymInt input that is the
        # symbol, and, under size_asserts, from an input size or stride that is:
        # x.view(s0 * s1, s2) binds neither s0 nor s1, while x's sizes or the
        # SymInts s0 and s1 do, and y[:, 0]'s stride binds y's second size.
        # Without size_asserts only the SymInts bind, and nor do the sizes of
        # z, of a static size 0.
        from torch._functorch._aot_autograd.deterministic_recompute import (
            _unbound_symbols,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(x, w, y, z):
            n0, n1 = x.shape[0], x.shape[1]
            return (x.view(n0 * n1, -1) @ w).sin(), y[:, 0].cos(), z.sum(1).sin()

        args = (torch.randn(4, 6, 8), torch.randn(8, 10), torch.randn(3, 5))
        args += (torch.randn(7, 0),)
        gm = make_fx(fn, tracing_mode="symbolic")(*args)
        n = {n.name: n for n in gm.graph.nodes}
        s0, s1, s2 = (d.node.expr for d in n["x_1"].meta["val"].shape)
        s3 = n["w_1"].meta["val"].shape[1].node.expr
        sizes = n["sym_size_int"], n["sym_size_int_1"]
        x, w, view, body = n["x_1"], n["w_1"], n["view"], [n["mm"], n["sin"]]
        with torch._inductor.config.patch(size_asserts=True):
            self.assertEqual(_unbound_symbols([view, w], body), {s0, s1})
            self.assertEqual(_unbound_symbols([x, view, w], body), set())
            self.assertEqual(_unbound_symbols([view, w, *sizes], body), set())
            self.assertEqual(_unbound_symbols([n["select"]], [n["cos"]]), set())
            z, z_sum = n["z_1"], n["sum_1"]
            s4 = z.meta["val"].shape[0].node.expr
            self.assertEqual(_unbound_symbols([z], [z_sum, n["sin_1"]]), {s4})
        with torch._inductor.config.patch(size_asserts=False):
            self.assertEqual(_unbound_symbols([x, view, w, *sizes], body), {s2, s3})

    def test_accumulates(self):
        # Floating-point atomics over repeated indices and scans can accumulate
        # in a different order each run; integer accumulation, an overwriting
        # index_put and an amax scatter can't.
        from torch._functorch._aot_autograd.deterministic_recompute import _accumulates
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(x, i, i2d):
            zeros, n = torch.zeros(2, 8), (x > 0).long()
            return (
                zeros.index_add(0, i, x),
                zeros.long().index_add(0, i, n),
                zeros.index_put((i,), x, accumulate=True),
                torch.zeros(4, 8).index_put((i.argsort(),), x),
                zeros.scatter_reduce(0, i2d, x, "sum"),
                zeros.scatter_reduce(0, i2d, x, "amax"),
                x.cumsum(0),
                n.cumsum(0),
                x.cumprod(0),
                x.logcumsumexp(0),
            )

        i = torch.tensor([0, 1, 0, 1])
        i2d = torch.tensor([[0] * 8, [1] * 8] * 2)
        gm = make_fx(fn)(torch.randn(4, 8), i, i2d)
        names = [n.name for n in gm.graph.nodes if _accumulates(n)]
        expected = ["index_add", "index_put", "scatter_reduce", "cumsum", "cumprod"]
        self.assertEqual(names, [*expected, "logcumsumexp"])


instantiate_parametrized_tests(DeterministicRecomputeHelperTests)


class ActivationCheckpointingNonStrictTracerTests(torch._dynamo.test_case.TestCase):
    """Tests for non-strict tracing flag interaction with checkpoint."""

    @staticmethod
    def _count_backward_regions(gm):
        regions = 0
        in_backward = False
        for node in gm.graph.nodes:
            is_backward = bool(node.meta.get("autograd_backward", False))
            if is_backward and not in_backward:
                regions += 1
            in_backward = is_backward
        return regions

    def test_backward_nodes_have_seq_nr_under_non_strict(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return checkpoint(
                    lambda x: torch.sin(x @ self.w), x, use_reentrant=False
                )

        gm = self._trace_train_step(Model(), torch.randn(2, 4))
        forward_seq_nrs = {
            node.meta["seq_nr"]
            for node in gm.graph.nodes
            if node.op == "call_function"
            and not node.meta.get("autograd_backward", False)
            and "seq_nr" in node.meta
        }
        backward_seq_nrs = {
            node.meta["seq_nr"]
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.meta.get("autograd_backward", False)
            and "seq_nr" in node.meta
        }

        self.assertTrue(forward_seq_nrs)
        self.assertTrue(backward_seq_nrs)
        self.assertSetEqual(backward_seq_nrs, forward_seq_nrs)

    def test_patch_autograd_grad_requires_non_strict_tracing(self):
        x = torch.randn(2, 4, requires_grad=True)
        loss = torch.sin(x).sum()

        with torch.compiler._patch_autograd_grad():
            with self.assertRaisesRegex(
                AssertionError,
                "_patch_autograd_grad\\(\\) must be used under "
                "_non_strict_tracing_context\\(\\)",
            ):
                torch.autograd.grad(loss, (x,))

    def test_patch_engine_backward_requires_non_strict_tracing(self):
        x = torch.randn(2, 4, requires_grad=True)
        loss = torch.sin(x).sum()

        with torch.compiler._patch_engine_backward():
            with self.assertRaisesRegex(
                AssertionError,
                "_patch_engine_backward\\(\\) must be used under "
                "_non_strict_tracing_context\\(\\)",
            ):
                loss.backward()

    def test_patch_autograd_grad_does_not_leak_backward_tag(self):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta

        x = torch.randn(2, 4, requires_grad=True)

        def fn(x):
            with torch.fx.traceback.annotate({"ac_region_id": 0}):
                y = torch.sin(x)
                torch.autograd.grad(y.sum(), (x,))
                return torch.neg(y)

        with (
            torch.compiler._non_strict_tracing_context(),
            torch.compiler._patch_autograd_grad(),
            preserve_node_meta(),
        ):
            gm = make_fx(fn)(x)

        backward_nodes = [
            node for node in gm.graph.nodes if node.meta.get("autograd_backward", False)
        ]
        self.assertTrue(backward_nodes)

        neg_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.neg.default
        )
        self.assertEqual(len(neg_nodes), 1)
        self.assertNotIn("autograd_backward", neg_nodes[0].meta)
        self.assertEqual(neg_nodes[0].meta.get("custom", {}), {"ac_region_id": 0})

    def test_patch_engine_backward_does_not_leak_backward_tag(self):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta

        x = torch.randn(2, 4, requires_grad=True)

        def fn(x):
            with torch.fx.traceback.annotate({"ac_region_id": 0}):
                y = torch.sin(x)
                y.sum().backward()
                return torch.neg(y)

        with (
            torch.compiler._non_strict_tracing_context(),
            torch.compiler._patch_engine_backward(),
            preserve_node_meta(),
        ):
            gm = make_fx(fn)(x)

        backward_nodes = [
            node for node in gm.graph.nodes if node.meta.get("autograd_backward", False)
        ]
        self.assertTrue(backward_nodes)

        neg_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.neg.default
        )
        self.assertEqual(len(neg_nodes), 1)
        self.assertNotIn("autograd_backward", neg_nodes[0].meta)
        self.assertEqual(neg_nodes[0].meta.get("custom", {}), {"ac_region_id": 0})

    def test_patch_autograd_grad_mlp_has_single_contiguous_backward_region(self):
        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim * 2)
                self.fc2 = nn.Linear(dim * 2, dim)

            def forward(self, x):
                return x + self.fc2(torch.relu(self.fc1(x)))

        class Model(nn.Module):
            def __init__(self, dim=32, depth=6):
                super().__init__()
                self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])

            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x

        gm = self._trace_train_step(Model(), torch.randn(4, 16, 32))
        self.assertEqual(self._count_backward_regions(gm), 1)

    def _trace_train_step(self, mod, x):
        import torch.utils._pytree as pytree
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta
        from torch.nn.utils import stateless

        params = dict(mod.named_parameters())
        flat_params, params_spec = pytree.tree_flatten(params)
        params_len = len(flat_params)

        def train_step(*all_args):
            p = pytree.tree_unflatten(list(all_args[:params_len]), params_spec)
            with stateless._reparametrize_module(mod, p):
                loss = mod(all_args[params_len]).sum()
                return torch.autograd.grad(loss, all_args[:params_len])

        full_args = (*flat_params, x)

        with (
            torch.compiler._non_strict_tracing_context(),
            torch.compiler._patch_autograd_grad(),
            preserve_node_meta(),
        ):
            return make_fx(train_step)(*full_args)

    def test_checkpoint_traces_through_eager_ac_under_non_strict(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return checkpoint(
                    lambda x: torch.sin(x @ self.w), x, use_reentrant=False
                )

        gm = self._trace_train_step(Model(), torch.randn(2, 4))

        # Everything is recomputed in backward: mm_1 is the recomputed mm.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg1_1, arg0_1)
    sin = torch.ops.aten.sin.default(mm);  mm = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [2, 4]);  ones_like = None
    mm_1 = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg0_1 = None
    alias = torch.ops.aten.alias.default(mm_1);  mm_1 = None
    alias_1 = torch.ops.aten.alias.default(alias);  alias = None
    cos = torch.ops.aten.cos.default(alias_1);  alias_1 = None
    mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None
    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_2 = torch.ops.aten.mm.default(t, mul);  t = mul = None
    return (mm_2,)""",
        )

    @torch._functorch.config.patch(deterministic_recompute=True)
    def test_deterministic_recompute_rejected_under_non_strict(self):
        # The recompute is traced as separate ops here, so it can't be shared.
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return checkpoint(
                    lambda x: torch.sin(x @ self.w), x, use_reentrant=False
                )

        with self.assertRaisesRegex(NotImplementedError, "train-step tracing"):
            self._trace_train_step(Model(), torch.randn(2, 4))

    def test_sac_traces_through_eager_ac_under_non_strict(self):
        def policy_fn(ctx, func, *args, **kwargs):
            if func == torch.ops.aten.mm.default:
                return CheckpointPolicy.PREFER_RECOMPUTE
            return CheckpointPolicy.MUST_SAVE

        def context_fn():
            return create_selective_checkpoint_contexts(policy_fn)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                def fn(x):
                    return torch.sin(x @ self.w)

                return checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        gm = self._trace_train_step(Model(), torch.randn(2, 4))

        # mm is PREFER_RECOMPUTE so it gets recomputed in backward (mm_1).
        # sin is MUST_SAVE so its output is saved via alias under make_fx's
        # default detach decomposition.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg1_1, arg0_1)
    sin = torch.ops.aten.sin.default(mm);  mm = None
    alias = torch.ops.aten.alias.default(sin);  alias = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [2, 4]);  ones_like = None
    mm_1 = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg0_1 = None
    alias_1 = torch.ops.aten.alias.default(mm_1);  mm_1 = None
    alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    cos = torch.ops.aten.cos.default(alias_2);  alias_2 = None
    mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None
    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_2 = torch.ops.aten.mm.default(t, mul);  t = mul = None
    return (mm_2,)""",
        )

    def test_checkpoint_with_rng_op_under_non_strict(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                def fn(x):
                    return torch.nn.functional.dropout(torch.sin(x @ self.w), p=0.5)

                return checkpoint(fn, x, use_reentrant=False, preserve_rng_state=True)

        mod = Model()
        x = torch.randn(2, 4)

        import torch.utils._pytree as pytree
        from torch.nn.utils import stateless

        params = dict(mod.named_parameters())
        flat_params, params_spec = pytree.tree_flatten(params)
        params_len = len(flat_params)

        def train_step(*all_args):
            p = pytree.tree_unflatten(list(all_args[:params_len]), params_spec)
            with stateless._reparametrize_module(mod, p):
                loss = mod(all_args[params_len]).sum()
                return torch.autograd.grad(loss, all_args[:params_len])

        full_args = (*flat_params, x)

        torch.manual_seed(42)
        with torch.compiler._non_strict_tracing_context():
            gm = make_fx(train_step)(*full_args)

        # The traced graph has two independent bernoulli_ calls (forward and
        # recomputed). At replay time these produce different dropout masks
        # because get/set_rng_state are silently dropped by make_fx, so the
        # traced graph is NOT bitwise equivalent to eager. This is a known
        # limitation of tracing through eager checkpoint.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg1_1, arg0_1)
    sin = torch.ops.aten.sin.default(mm);  mm = None
    empty_like = torch.ops.aten.empty_like.default(sin)
    bernoulli_ = torch.ops.aten.bernoulli_.float(empty_like);  empty_like = None
    div_ = torch.ops.aten.div_.Scalar(bernoulli_, 0.5);  bernoulli_ = None
    mul = torch.ops.aten.mul.Tensor(sin, div_);  sin = div_ = None
    sum_1 = torch.ops.aten.sum.default(mul);  mul = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [2, 4]);  ones_like = None
    mm_1 = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg0_1 = None
    alias = torch.ops.aten.alias.default(mm_1)
    sin_1 = torch.ops.aten.sin.default(mm_1);  mm_1 = None
    empty_like_1 = torch.ops.aten.empty_like.default(sin_1);  sin_1 = None
    bernoulli__1 = torch.ops.aten.bernoulli_.float(empty_like_1);  empty_like_1 = None
    div__1 = torch.ops.aten.div_.Scalar(bernoulli__1, 0.5);  bernoulli__1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, div__1);  expand = div__1 = None
    alias_1 = torch.ops.aten.alias.default(alias);  alias = None
    cos = torch.ops.aten.cos.default(alias_1);  alias_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(mul_1, cos);  mul_1 = cos = None
    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_2 = torch.ops.aten.mm.default(t, mul_2);  t = mul_2 = None
    return (mm_2,)""",
        )


class ActivationCheckpointingNestedCompileTests(torch._dynamo.test_case.TestCase):
    @requires_gpu_and_triton
    def test_checkpoint_recompute_preserves_nested_fx_trace_policy(self):
        from torch._guards import tracing, TracingContext
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta

        compiled_f = torch.compile(lambda x: x.sin().cos(), fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        @contextlib.contextmanager
        def skip_nested_compile():
            prev = torch._dynamo.config.error_on_nested_fx_trace
            torch._dynamo.config.error_on_nested_fx_trace = False
            try:
                yield
            finally:
                torch._dynamo.config.error_on_nested_fx_trace = prev

        class M(torch.nn.Module):
            def forward(self, x):
                return checkpoint(self.block, x, use_reentrant=False)

            def block(self, x):
                return compiled_f(x)

        m = getattr(M(), device_type)()
        x = torch.randn(8, device=device_type, requires_grad=True)

        def fn(x):
            y = m(x).sum()
            (gx,) = torch.autograd.grad(y, (x,))
            return y.detach(), gx

        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=torch.fx.experimental.symbolic_shapes.ShapeEnv(),
        )
        fx_x = fake_mode.from_tensor(x, static_shapes=True)
        if x.requires_grad and not fx_x.requires_grad:
            fx_x.requires_grad_(True)

        ctx = TracingContext(fake_mode)

        with (
            fake_mode,
            tracing(ctx),
            preserve_node_meta(),
            skip_nested_compile(),
            torch.compiler._non_strict_tracing_context(),
        ):
            gm = make_fx(fn)(fx_x)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sin = torch.ops.aten.sin.default(x_1)
    cos = torch.ops.aten.cos.default(sin);  sin = None
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format)
    expand = torch.ops.aten.expand.default(ones_like, [8]);  ones_like = None
    alias = torch.ops.aten.alias.default(x_1)
    sin_1 = torch.ops.aten.sin.default(x_1);  x_1 = None
    alias_1 = torch.ops.aten.alias.default(sin_1);  sin_1 = None
    alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sin_2 = torch.ops.aten.sin.default(alias_2);  alias_2 = None
    neg = torch.ops.aten.neg.default(sin_2);  sin_2 = None
    mul = torch.ops.aten.mul.Tensor(expand, neg);  expand = neg = None
    alias_3 = torch.ops.aten.alias.default(alias);  alias = None
    cos_1 = torch.ops.aten.cos.default(alias_3);  alias_3 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, cos_1);  mul = cos_1 = None
    alias_4 = torch.ops.aten.alias.default(sum_1);  sum_1 = None
    return (alias_4, mul_1)""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
