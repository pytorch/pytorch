# Owner(s): ["module: dynamo"]

import copy
import functools
import sys
import warnings
from typing import Any, TYPE_CHECKING

import torch
import torch._inductor.test_case
import torch.fx.traceback as fx_traceback
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import _testing_capture_invoke_subgraph_inductor_compile_gms
from torch._functorch._aot_autograd.autograd_cache import BundledCompiledForward
from torch._guards import detect_fake_mode
from torch._higher_order_ops.invoke_subgraph import get_invoke_subgraph_compile_options
from torch._inductor.output_code import RegionalOutputCode
from torch._inductor.test_case import run_tests
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.fx._graph_pickler import GraphPickler
from torch.fx.passes.regional_inductor import regional_inductor
from torch.fx.passes.regional_inductor_invoke_subgraph import (
    regional_inductor_invoke_subgraph,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfTorchDynamo,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


if TYPE_CHECKING:
    from torch._inductor.compile_fx import _CompileFxKwargs


# Open questions / follow-ups
# 1) CSE behavior with meta custom nodes
#   Common subexpression elimination may not differentiate between distinct meta
#   custom nodes and could remove expressions, which might confuse users.
#
# 2) SAC: recompute vs. forward size
#   If the recomputed forward is smaller than the original forward, do we end up
#   compiling only the smaller region?
#
# 3) fx_traceback.annotate nesting
#   How does nesting behave? Are there any ordering requirements?
#
# 4) Planned uses for annotations
#   a) compile flex
#   b) streams
#   c) nn.Module info to organize MoE runtime
#   d) pipeline-parallel stages
#   e) rename graph nodes for easier debugging
#   f) disallow nested regional compile


def aot_eager_regional_inductor(
    serialize=False, on_invoke_subgraph=False, captured_gms=None, partitioner=None
):
    def regional_inductor_fn(gm, *args, **kwargs):
        if captured_gms is not None:
            captured_gms.append(copy.deepcopy(gm))

        if on_invoke_subgraph:
            return regional_inductor_invoke_subgraph(gm, *args, **kwargs)
        else:
            return regional_inductor(gm, *args, **kwargs)

    kwargs = {}
    if partitioner:
        kwargs["partition_fn"] = partitioner
    if serialize:

        def regional_inductor_pickle(gm, *example_args):
            with torch._functorch.config.patch(force_autograd_cache=True):
                result = regional_inductor_fn(gm, *example_args)
            serialized = GraphPickler.dumps(result)

            fake_mode = detect_fake_mode(example_args)
            assert fake_mode is not None
            # Serialize and deserialize the result to confirm pickling works
            # Use a fresh tracing context on the new process
            context = torch._guards.TracingContext(fake_mode)
            with torch._guards.tracing(context):
                result = GraphPickler.loads(serialized, fake_mode)
                if isinstance(result, torch.fx.GraphModule):
                    result.recompile()
                elif isinstance(result, RegionalOutputCode):
                    result._graph_module.recompile()
                else:
                    raise RuntimeError(f"Unexpected type: {type(result)}")
                return result

        return aot_autograd(
            fw_compiler=regional_inductor_pickle,
            bw_compiler=regional_inductor_pickle,
            **kwargs,
        )

    return aot_autograd(
        fw_compiler=regional_inductor_fn,
        bw_compiler=regional_inductor_fn,
        **kwargs,
    )


@skipIfTorchDynamo("Not a suitable dynamo wrapped test")
@instantiate_parametrized_tests
class RegionalInductorTests(torch._inductor.test_case.TestCase):
    @parametrize("serialize", [False, True])
    def test_simple(self, serialize):
        def fn(x, y):
            sin = torch.sin(x)

            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1

            return torch.sin(add)

        opt_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(serialize=serialize), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Check that inductor compilation is called twice
        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x, y))
        self.assertEqual(len(codes), 2)

    def test_boxed_calling_convention(self):
        def fn(x, y):
            sin = torch.sin(x)

            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1

            return torch.sin(add)

        opt_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(serialize=False), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Check that inductor compilation is called twice
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x, y))

        msgs = [str(warn.message) for warn in w]
        self.assertTrue(
            not any(
                "Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments"
                in m
                for m in msgs
            )
        )

    @parametrize("serialize", [False, True])
    def test_repeated_blocks(self, serialize):
        def fn(x, y):
            sin = torch.sin(x)

            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1

            return torch.sin(add)

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = fn(x, y)
                return fn(a, y)

        mod = Mod()

        opt_mod = torch.compile(
            mod,
            backend=aot_eager_regional_inductor(serialize=serialize),
            fullgraph=True,
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Check that inductor compilation is called 4 times
        # there will be 2 partitions in the fwd and 2 in the bwd, totalling 4
        _, codes = run_fw_bw_and_get_code(lambda: opt_mod(x, y))
        self.assertEqual(len(codes), 4)

    @parametrize("serialize", [False, True])
    def test_invoke_subgraph(self, serialize):
        # Checks that get_attr nodes custom metadata is propagated
        @torch.compiler.nested_compile_region
        def gn(x):
            return torch.sin(x)

        def fn(x):
            x = x + 1
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                z = gn(x)
            return torch.sigmoid(z)

        opt_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(serialize=serialize), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)

        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
        self.assertEqual(len(codes), 2)

    @parametrize("serialize", [False, True])
    def test_invoke_subgraph_inner(self, serialize):
        # Checks that the inductor regions are searched recursively.

        @torch.compiler.nested_compile_region
        def gn(x):
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                return torch.sin(x)

        def fn(x):
            x = x + 1
            x = gn(x)
            x = x + 1
            x = gn(x)
            return torch.sigmoid(x)

        opt_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(serialize=serialize), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)

        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
        # the invoke_subgraph is called twice - but the inside code is compiled
        # once - so in total 2 (1 fwd + 1 bwd)
        self.assertEqual(len(codes), 2)

    @requires_cuda_and_triton
    @parametrize("serialize", [False, True])
    def test_flex_attention(self, serialize):
        def _squared(score, b, h, m, n):
            return score * score

        def mask_mod(b, h, q, k):
            return q >= 0

        a = 12
        b = 64
        block_mask = create_block_mask(mask_mod, None, None, a * b, a * b)

        def fn(x):
            x = torch.sin(x)
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                x = flex_attention(x, x, x, block_mask=block_mask, score_mod=_squared)
            return torch.cos(x)

        x = torch.randn(
            1,
            1,
            a * b,
            b,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )

        opt_fn = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(serialize),
            fullgraph=True,
        )

        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
        # flex in forward and flex_backward in backward
        self.assertEqual(len(codes), 2)

    @parametrize("serialize", [False, True])
    def test_max_autotune_no_cudagraphs(self, serialize):
        """Test that max-autotune-no-cudagraphs options are properly applied via annotations."""
        import torch._inductor.config as inductor_config

        def fn(x, y):
            sin = torch.sin(x)

            # Use annotation API to specify inductor configs
            with fx_traceback.annotate(
                {
                    "compile_with_inductor": {
                        "inductor_configs": {
                            "max_autotune": True,
                            "triton.cudagraphs": False,
                        }
                    }
                }
            ):
                mul = sin * y
                add = mul + 1

            return torch.sin(add)

        # Hook to verify options
        original_compile = torch._inductor.standalone_compile
        captured_options = []

        def verify_options(*args, **kwargs):
            options = kwargs.get("options", {})
            captured_options.append(options)

            # Verify config is set as expected from explicit options
            assert inductor_config.max_autotune, "max_autotune should be True"
            assert not inductor_config.triton.cudagraphs, (
                "triton.cudagraphs should be False"
            )

            return original_compile(*args, **kwargs)

        torch._inductor.standalone_compile = verify_options

        try:
            # Use backend without options - they come from annotations
            backend = aot_eager_regional_inductor(serialize=serialize)

            opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
            x = torch.randn(10, requires_grad=True)
            y = torch.randn(10, requires_grad=True)

            # Run and check that options were passed
            _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x, y))
            self.assertEqual(len(codes), 2)

            # Verify that compilation happened
            self.assertTrue(
                len(captured_options) > 0, "Compilation should have occurred"
            )

        finally:
            torch._inductor.standalone_compile = original_compile

    def test_annotation_inductor_configs(self):
        """Test that inductor_configs can be passed through annotation API."""
        import torch._inductor.config as inductor_config

        def fn_with_annotation_configs(x, y):
            # New annotation format with inductor_configs
            with fx_traceback.annotate(
                {
                    "compile_with_inductor": {
                        "inductor_configs": {
                            "max_autotune": True,
                            "triton.cudagraphs": False,
                        }
                    }
                }
            ):
                return torch.matmul(x, y) + 1

        # Capture config during compilation
        config_snapshots = []

        original_compile = torch._inductor.standalone_compile

        def capture_config(*args, **kwargs):
            config_snapshots.append(
                {
                    "max_autotune": inductor_config.max_autotune,
                    "triton.cudagraphs": inductor_config.triton.cudagraphs,
                }
            )
            return original_compile(*args, **kwargs)

        torch._inductor.standalone_compile = capture_config

        try:
            backend = aot_eager_regional_inductor()

            opt_fn = torch.compile(
                fn_with_annotation_configs, backend=backend, fullgraph=True
            )
            x = torch.randn(32, 32, requires_grad=True)
            y = torch.randn(32, 32, requires_grad=True)

            # Run forward and backward
            result = opt_fn(x, y)
            result.sum().backward()

            self.assertTrue(len(config_snapshots) > 0, "No compilation occurred")

            for snapshot in config_snapshots:
                self.assertEqual(snapshot["max_autotune"], True)
                self.assertEqual(snapshot["triton.cudagraphs"], False)

        finally:
            torch._inductor.standalone_compile = original_compile

    def test_invalid_inductor_config(self):
        """Test that invalid inductor config keys are caught with a clear error."""

        def fn(x, y):
            with fx_traceback.annotate(
                {
                    "compile_with_inductor": {
                        "inductor_configs": {
                            "invalid_config_key": True,
                        }
                    }
                }
            ):
                return x * y + 1

        backend = aot_eager_regional_inductor()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Invalid inductor config key 'invalid_config_key'",
        ):
            opt_fn(x, y)

    @requires_cuda_and_triton
    @parametrize("serialize", [False, True])
    def test_selective_ac_flex(self, serialize):
        class FlexAttentionModule(torch.nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads

                # In-projections (query, key, value)
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size)

                # Out-projection
                self.out_proj = torch.nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                batch_size, seq_len, _ = x.size()

                # Project queries, keys, and values
                q = (
                    self.q_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )

                # Apply flex attention
                with torch.fx.traceback.annotate({"compile_with_inductor": 0}):
                    attn_output = flex_attention(
                        q,
                        k,
                        v,
                    )

                # Reshape output
                attn_output = (
                    attn_output.transpose(1, 2)
                    .contiguous()
                    .view(batch_size, seq_len, self.hidden_size)
                )

                # Out projection
                output = self.out_proj(attn_output)

                return output

        from torch.utils.checkpoint import (
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        ops_to_save = [
            torch.ops.aten.mm.default,
        ]
        context_fn = functools.partial(
            create_selective_checkpoint_contexts, ops_to_save
        )

        # Define a model that uses FlexAttention with selective activation checkpointing
        class SacModule(torch.nn.Module):
            def __init__(self, hidden_size, num_heads, context_fn):
                super().__init__()
                self.flex_attn = FlexAttentionModule(hidden_size, num_heads)
                self.context_fn = context_fn

            def forward(self, x):
                def flex_attn_fn(x):
                    return self.flex_attn(x)

                output = checkpoint(
                    flex_attn_fn,
                    x,
                    use_reentrant=False,
                    context_fn=self.context_fn,
                )

                return output

        flex_module = SacModule(hidden_size=512, num_heads=8, context_fn=context_fn).to(
            "cuda", dtype=torch.bfloat16
        )
        x = torch.ones(8, 1024, 512, device="cuda", dtype=torch.bfloat16)
        compiled_module = torch.compile(
            flex_module, backend=aot_eager_regional_inductor(), fullgraph=True
        )

        _, codes = run_fw_bw_and_get_code(lambda: compiled_module(x))
        # flex in forward and flex_backward in backward
        self.assertEqual(len(codes), 2)

    def test_refcounts(self):
        """Tests that activations can be cleared before the end of graph"""

        class RefcountCheckPassed(Exception):
            pass

        def regional_inductor_with_refcounting(gm, *example_args):
            fn = regional_inductor(gm, *example_args)
            assert fn._boxed_call

            def run(args: Any) -> Any:
                assert type(args) is list

                # NOTE: sys.getrefcount adds a temporary reference to the object
                # So sys.getrefcount(x) == 2 actually means we hold the single reference to x
                # There should be one activation for `fn`.
                self.assertTrue(
                    2 in [sys.getrefcount(args[i]) for i in range(len(args))]
                )
                return fn(args)

            run._boxed_call = True  # type: ignore[attr-defined]
            return run

        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):  # tensor of 1s
                act = x + x  # tensor of 2s
                ctx.save_for_backward(act)
                return act

            @staticmethod
            def backward(ctx, grad_output):  # tensor of 1s
                saved_act = ctx.saved_tensors  # tensor of 2s
                return saved_act[0] + grad_output  # tensor of 3s

        @torch.compile(
            backend=aot_autograd(
                fw_compiler=regional_inductor,
                bw_compiler=regional_inductor_with_refcounting,
            ),
            fullgraph=True,
        )
        def fn(x):
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                return MyAutogradFunction.apply(x)

        x = torch.ones(10, 10, requires_grad=True)

        fn(x).sum().backward()
        self.assertEqual(x.grad, x * 3)


@skipIfTorchDynamo("Not a suitable dynamo wrapped test")
@torch._dynamo.config.patch("enable_invoke_subgraph_regional_compile", True)
@instantiate_parametrized_tests
class RegionalInductorInvokeSubgraphTests(torch._inductor.test_case.TestCase):
    def test_custom_decomposition(self):
        # Test that custom decompositions are applied to the subgraph.

        def my_add_decomp(a, b):
            return a.sin()

        nested_config = get_invoke_subgraph_compile_options(
            decompositions={
                torch.ops.aten.add.Tensor: my_add_decomp,
                torch.ops.aten.add.default: my_add_decomp,
            }
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def g(y):
            return y + 1

        def fn(x, y):
            y = x + 1  # this should still be add
            add = g(y)  # this should be decomposed to y.sin()
            return add * 2

        opt_mod = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(
                serialize=False, on_invoke_subgraph=True
            ),
            fullgraph=True,
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        with _testing_capture_invoke_subgraph_inductor_compile_gms() as captured_gms:
            # Check that inductor compilation is called 2 times only
            # So there's not double-compilation like when we using fx.annotate.
            result, codes = run_fw_bw_and_get_code(lambda: opt_mod(x, y))
            self.assertEqual(len(codes), 2)
            self.assertEqual(
                result, 2 * torch.sin(x + 1)
            )  # note that the g() should be decomposed to sin()

            self.assertEqual(len(captured_gms), 2)
            # dynamo captured forward graph
            self.assertExpectedInline(
                captured_gms[0].code.strip(),
                """\
def forward(self, primals_0):
    sin = torch.ops.aten.sin.default(primals_0);  primals_0 = None
    return (sin,)""",
                ignore_comments=True,
                ignore_empty_lines=True,
            )
            # dynamo captured backward graph
            self.assertExpectedInline(
                captured_gms[1].code.strip(),
                """\
def forward(self, tangents_0):
    clone = torch.ops.aten.clone.default(tangents_0);  tangents_0 = None
    return (clone,)""",
                ignore_comments=True,
                ignore_empty_lines=True,
            )

    @parametrize("serialize", [False])  # , True
    def test_simple(self, serialize):
        nested_config = get_invoke_subgraph_compile_options()

        @torch.compiler.nested_compile_region(options=nested_config)
        def g(sin, y):
            mul = sin * y
            add = mul + 1
            return add

        def fn(x, y):
            sin = torch.sin(x)
            add = g(sin, y)
            return torch.sin(add)

        opt_fn = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(
                serialize=serialize, on_invoke_subgraph=True
            ),
            fullgraph=True,
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Check that inductor compilation is called twice
        result, codes = run_fw_bw_and_get_code(lambda: opt_fn(x, y))
        self.assertEqual(len(codes), 2)
        self.assertEqual(result, fn(x, y))

    @parametrize("serialize", [False])  # , True
    def test_two_graphs(self, serialize):
        nested_config = get_invoke_subgraph_compile_options()

        @torch.compiler.nested_compile_region(options=nested_config)
        def g1(sin, y):
            mul = sin * y
            add = mul + 1
            return add

        @torch.compiler.nested_compile_region(options=nested_config)
        def g2(x):
            return x / 3

        def fn(x, y):
            sin = torch.sin(x)
            add = g1(sin, y)
            div = g2(add)
            return div

        opt_fn = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(
                serialize=serialize, on_invoke_subgraph=True
            ),
            fullgraph=True,
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Check that inductor compilation is called 4 times, twice for each nested region
        result, codes = run_fw_bw_and_get_code(lambda: opt_fn(x, y))
        self.assertEqual(len(codes), 4)
        self.assertEqual(result, fn(x, y))

    @parametrize("serialize", [False])  # , True
    def test_unbacked_expr_input(self, serialize):
        # https://github.com/pytorch/pytorch/issues/167012

        nested_config = get_invoke_subgraph_compile_options()

        @torch.compiler.nested_compile_region(options=nested_config)
        def gn(x):
            return x + 1

        def fn(c):
            d = torch.concat([c, c], dim=0)
            d = gn(d)
            return d

        c = torch.randn((64, 32), requires_grad=True)
        torch._dynamo.decorators.mark_unbacked(c, 0)

        opt_fn = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(
                serialize=serialize,
                on_invoke_subgraph=True,
            ),
            fullgraph=True,
        )

        result, codes = run_fw_bw_and_get_code(lambda: opt_fn(c))
        # self.assertEqual(len(codes), 2)
        self.assertEqual(result, fn(c))

    @parametrize("serialize", [False])
    def test_repeated_blocks(self, serialize):
        nested_config = get_invoke_subgraph_compile_options()

        @torch.compiler.nested_compile_region(options=nested_config)
        def g(sin, y):
            mul = sin * y
            add = mul + 1
            return add

        def fn(x, y):
            sin = torch.sin(x)
            add = g(sin, y)
            return torch.sin(add)

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = fn(x, y)
                return fn(a, y)

        mod = Mod()

        captured_gms = []
        opt_mod = torch.compile(
            mod,
            backend=aot_eager_regional_inductor(
                serialize=serialize, on_invoke_subgraph=True, captured_gms=captured_gms
            ),
            fullgraph=True,
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        with (
            _testing_capture_invoke_subgraph_inductor_compile_gms() as inner_captured_gms
        ):
            # Check that inductor compilation is called 2 times only
            # So there's not double-compilation like when we using fx.annotate.
            result, codes = run_fw_bw_and_get_code(lambda: opt_mod(x, y))
            self.assertEqual(len(codes), 2)
            self.assertEqual(result, mod(x, y))

            self.assertEqual(len(captured_gms), 2)
            # inductor compiled forward graph
            self.assertExpectedInline(
                inner_captured_gms[0].code.strip(),
                """\
def forward(self, primals_0, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_0, primals_1)
    add = torch.ops.aten.add.Tensor(mul, 1);  mul = None
    return (add, primals_0, primals_1)""",
                ignore_comments=True,
                ignore_empty_lines=True,
            )
            # inductor compiled backward graph
            self.assertExpectedInline(
                inner_captured_gms[1].code.strip(),
                """\
def forward(self, primals_0, primals_1, tangents_0):
    mul_1 = torch.ops.aten.mul.Tensor(tangents_0, primals_0);  primals_0 = None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_0, primals_1);  tangents_0 = primals_1 = None
    return (mul_2, mul_1)""",
                ignore_comments=True,
                ignore_empty_lines=True,
            )

        # Graph modules compiled by aot_eager_regional_inductor backend
        self.assertExpectedInline(
            captured_gms[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    sin = torch.ops.aten.sin.default(primals_1)
    partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
    invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', sin, primals_2);  partitioned_fw_subgraph_0_0 = sin = None
    getitem_9 = invoke_subgraph_4[2]
    getitem_8 = invoke_subgraph_4[1]
    getitem = invoke_subgraph_4[0];  invoke_subgraph_4 = None
    sin_1 = torch.ops.aten.sin.default(getitem)
    sin_2 = torch.ops.aten.sin.default(sin_1)
    partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
    invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', sin_2, primals_2);  partitioned_fw_subgraph_0_1 = sin_2 = primals_2 = None
    getitem_11 = invoke_subgraph_6[2]
    getitem_10 = invoke_subgraph_6[1]
    getitem_1 = invoke_subgraph_6[0];  invoke_subgraph_6 = None
    sin_3 = torch.ops.aten.sin.default(getitem_1)
    return (sin_3, primals_1, getitem_9, getitem_8, getitem, sin_1, getitem_11, getitem_10, getitem_1)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        self.assertExpectedInline(
            captured_gms[1].code.strip(),
            """\
def forward(self, primals_1, getitem_9, getitem_8, getitem, sin_1, getitem_11, getitem_10, getitem_1, tangents_1):
    cos = torch.ops.aten.cos.default(getitem_1);  getitem_1 = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, cos);  tangents_1 = cos = None
    partitioned_bw_subgraph_0_1 = self.partitioned_bw_subgraph_0_0
    invoke_subgraph_7 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_1, 'partitioned_bw_subgraph_0_0', getitem_10, getitem_11, mul);  partitioned_bw_subgraph_0_1 = getitem_10 = getitem_11 = mul = None
    getitem_2 = invoke_subgraph_7[0]
    getitem_3 = invoke_subgraph_7[1];  invoke_subgraph_7 = None
    cos_1 = torch.ops.aten.cos.default(sin_1);  sin_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(getitem_2, cos_1);  getitem_2 = cos_1 = None
    cos_2 = torch.ops.aten.cos.default(getitem);  getitem = None
    mul_2 = torch.ops.aten.mul.Tensor(mul_1, cos_2);  mul_1 = cos_2 = None
    partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0
    invoke_subgraph_5 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_8, getitem_9, mul_2);  partitioned_bw_subgraph_0_0 = getitem_8 = getitem_9 = mul_2 = None
    getitem_5 = invoke_subgraph_5[0]
    getitem_6 = invoke_subgraph_5[1];  invoke_subgraph_5 = None
    add = torch.ops.aten.add.Tensor(getitem_3, getitem_6);  getitem_3 = getitem_6 = None
    cos_3 = torch.ops.aten.cos.default(primals_1);  primals_1 = None
    mul_3 = torch.ops.aten.mul.Tensor(getitem_5, cos_3);  getitem_5 = cos_3 = None
    return (mul_3, add)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    @parametrize("serialize", [False])  # , True
    def test_invoke_subgraph_inner(self, serialize):
        # Checks that the inductor regions are searched recursively.

        nested_config = get_invoke_subgraph_compile_options()

        @torch.compiler.nested_compile_region(options=nested_config)
        def g(y):
            return y * 2

        @torch.compiler.nested_compile_region
        def gn(x):
            x = g(x)
            return torch.sin(x)

        def fn(x):
            x = x + 1
            x = gn(x)
            x = x + 1
            x = gn(x)
            return torch.sigmoid(x)

        opt_fn = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(
                serialize=serialize, on_invoke_subgraph=True
            ),
            fullgraph=True,
        )
        x = torch.randn(10, requires_grad=True)

        with _testing_capture_invoke_subgraph_inductor_compile_gms() as captured_gms:
            _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
            # the invoke_subgraph is called twice - but the inside code is compiled
            # once - so in total 2 (1 fwd + 1 bwd)
            self.assertEqual(len(codes), 2)

            self.assertEqual(len(captured_gms), 2)
            # inductor compiled forward graph
            self.assertExpectedInline(
                captured_gms[0].code.strip(),
                """\
def forward(self, arg0_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
    return (mul,)""",
                ignore_comments=True,
                ignore_empty_lines=True,
            )
            # inductor compiled backward graph
            self.assertExpectedInline(
                captured_gms[1].code.strip(),
                """\
def forward(self, arg0_1, arg1_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    return (mul_1, mul)""",
                ignore_comments=True,
                ignore_empty_lines=True,
            )

    @requires_cuda_and_triton
    @parametrize("serialize", [False])  # , True
    def test_flex_attention(self, serialize):
        def _squared(score, b, h, m, n):
            return score * score

        def mask_mod(b, h, q, k):
            return q >= 0

        a = 12
        b = 64
        block_mask = create_block_mask(mask_mod, None, None, a * b, a * b)

        # must decompose aten.zeros.default, otherwise inductor complain
        nested_config = get_invoke_subgraph_compile_options(
            decompositions=torch._decomp.core_aten_decompositions()
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def f_flex_attention(x, y, z, block_mask, score_mod):
            x = flex_attention(x, y, z, block_mask=block_mask, score_mod=score_mod)
            return x

        def fn(x):
            x = torch.sin(x)
            x = f_flex_attention(x, x, x, block_mask=block_mask, score_mod=_squared)
            return torch.cos(x)

        x = torch.randn(
            1,
            1,
            a * b,
            b,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )

        opt_fn = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(serialize, on_invoke_subgraph=True),
            fullgraph=True,
        )

        with _testing_capture_invoke_subgraph_inductor_compile_gms() as captured_gms:
            res, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
            # flex in forward and flex_backward in backward
            self.assertEqual(len(codes), 2)
            true_res = fn(x)
            self.assertEqual(res, true_res)

            self.assertEqual(len(captured_gms), 2)
            # inductor compiled forward graph
            self.assertExpectedInline(
                captured_gms[0].code.strip(),
                """\
def forward(self, primals_0, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8):
    sdpa_score0 = self.sdpa_score0
    sdpa_mask0 = self.sdpa_mask0
    flex_attention = torch.ops.higher_order.flex_attention(primals_0, primals_0, primals_0, sdpa_score0, (768, 768, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, 128, 128, sdpa_mask0), 0.125, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), ());  sdpa_score0 = sdpa_mask0 = None
    getitem = flex_attention[0]
    getitem_1 = flex_attention[1];  flex_attention = None
    alias = torch.ops.aten.alias.default(getitem)
    alias_1 = torch.ops.aten.alias.default(getitem_1);  getitem_1 = None
    alias_2 = torch.ops.aten.alias.default(alias);  alias = None
    alias_3 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    return (getitem, primals_0, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, alias_2, alias_3)""",  # noqa: B950
                ignore_comments=True,
                ignore_empty_lines=True,
            )
            # inductor compiled backward graph
            self.assertExpectedInline(
                captured_gms[1].code.strip(),
                """\
def forward(self, primals_0, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, alias_2, alias_3, tangents_0):
    full_10 = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    fw_graph0 = self.fw_graph0
    joint_graph0 = self.joint_graph0
    mask_graph0 = self.mask_graph0
    flex_attention_backward = torch.ops.higher_order.flex_attention_backward(primals_0, primals_0, primals_0, alias_2, alias_3, tangents_0, full_10, fw_graph0, joint_graph0, (768, 768, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, 128, 128, mask_graph0), 0.125, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), ());  primals_0 = alias_2 = alias_3 = tangents_0 = full_10 = fw_graph0 = joint_graph0 = primals_1 = primals_2 = primals_3 = primals_4 = primals_5 = primals_6 = primals_7 = primals_8 = mask_graph0 = None
    getitem_3 = flex_attention_backward[0]
    getitem_4 = flex_attention_backward[1]
    getitem_5 = flex_attention_backward[2];  flex_attention_backward = None
    add = torch.ops.aten.add.Tensor(getitem_3, getitem_4);  getitem_3 = getitem_4 = None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_5);  add = getitem_5 = None
    return (add_1, None, None, None, None, None, None, None, None)""",  # noqa: B950
                ignore_comments=True,
                ignore_empty_lines=True,
            )

    @parametrize("serialize", [False])  # , True
    def test_max_autotune_no_cudagraphs(self, serialize):
        """Test that max-autotune-no-cudagraphs options are properly applied inductor_config_patches."""
        import torch._inductor.config as inductor_config

        nested_config = get_invoke_subgraph_compile_options(
            inductor_config_patches={
                "max_autotune": True,
                "triton.cudagraphs": False,
            }
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def g(sin, y):
            mul = sin * y
            add = mul + 1
            return add

        def fn(x, y):
            sin = torch.sin(x)
            add = g(sin, y)
            return torch.sin(add)

        # Hook to verify options
        original_compile = torch._inductor.compile_fx._compile_fx_inner
        captured_options = []

        def verify_options(*args, **kwargs):
            options = kwargs.get("inductor_config_patches", {})
            captured_options.append(options)

            # Verify config is set as expected from explicit options
            assert torch._inductor.config.max_autotune, "max_autotune should be True"
            assert not inductor_config.triton.cudagraphs, (
                "triton.cudagraphs should be False"
            )

            return original_compile(*args, **kwargs)

        torch._inductor.compile_fx._compile_fx_inner = verify_options

        try:
            # Use backend without options - they come from annotations
            backend = aot_eager_regional_inductor(
                serialize=serialize, on_invoke_subgraph=True
            )

            opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
            x = torch.randn(10, requires_grad=True)
            y = torch.randn(10, requires_grad=True)

            # Run and check that options were passed
            _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x, y))
            self.assertEqual(len(codes), 2)

            # Verify that compilation happened
            self.assertTrue(
                len(captured_options) > 0, "Compilation should have occurred"
            )

        finally:
            torch._inductor.compile_fx._compile_fx_inner = original_compile

    def test_invalid_inductor_config(self):
        """Test that invalid inductor config keys are caught with a clear error."""

        with self.assertRaisesRegex(
            ValueError,
            "Invalid inductor config key 'invalid_config_key'",
        ):
            get_invoke_subgraph_compile_options(
                inductor_config_patches={
                    "invalid_config_key": True,
                }
            )

    @requires_cuda_and_triton
    @parametrize("serialize", [False])  # , True
    def test_selective_ac_flex(self, serialize):
        # must decompose the following fallback ops in inductor
        # e.g. AssertionError: both a fallback and a decomp for same op: aten.zeros.default
        decomp_table = torch._decomp.core_aten_decompositions()
        decomp_table.update(
            torch._decomp.get_decompositions(
                {
                    torch.ops.aten.arange.start_step,
                    torch.ops.aten._to_copy.default,
                }
            )
        )
        nested_config = get_invoke_subgraph_compile_options(decompositions=decomp_table)

        @torch.compiler.nested_compile_region(options=nested_config)
        def f_flex_attention(x, y, z):
            x = flex_attention(x, y, z)
            return x

        class FlexAttentionModule(torch.nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads

                # In-projections (query, key, value)
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size)

                # Out-projection
                self.out_proj = torch.nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                batch_size, seq_len, _ = x.size()

                # Project queries, keys, and values
                q = (
                    self.q_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )

                # Apply flex attention
                attn_output = f_flex_attention(
                    q,
                    k,
                    v,
                )

                # Reshape output
                attn_output = (
                    attn_output.transpose(1, 2)
                    .contiguous()
                    .view(batch_size, seq_len, self.hidden_size)
                )

                # Out projection
                output = self.out_proj(attn_output)

                return output

        from torch.utils.checkpoint import (
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        ops_to_save = [
            torch.ops.aten.mm.default,
        ]
        context_fn = functools.partial(
            create_selective_checkpoint_contexts, ops_to_save
        )

        # Define a model that uses FlexAttention with selective activation checkpointing
        class SacModule(torch.nn.Module):
            def __init__(self, hidden_size, num_heads, context_fn):
                super().__init__()
                self.flex_attn = FlexAttentionModule(hidden_size, num_heads)
                self.context_fn = context_fn

            def forward(self, x):
                def flex_attn_fn(x):
                    return self.flex_attn(x)

                output = checkpoint(
                    flex_attn_fn,
                    x,
                    use_reentrant=False,
                    context_fn=self.context_fn,
                )

                return output

        flex_module = SacModule(hidden_size=512, num_heads=8, context_fn=context_fn).to(
            "cuda", dtype=torch.bfloat16
        )
        x = torch.ones(8, 1024, 512, device="cuda", dtype=torch.bfloat16)
        compiled_module = torch.compile(
            flex_module,
            backend=aot_eager_regional_inductor(serialize, on_invoke_subgraph=True),
            fullgraph=True,
        )

        res, codes = run_fw_bw_and_get_code(lambda: compiled_module(x))
        # flex in forward and flex_backward in backward
        self.assertEqual(len(codes), 2)
        true_res = flex_module(x)
        self.assertEqual(res, true_res)

    @parametrize("serialize", [False])  # True,
    def test_invoke_subgraph_regional_compile_decomposition(self, serialize):
        def my_sin_decomp(x):
            return torch.cos(x)

        decompositions = {torch.ops.aten.sin.default: my_sin_decomp}
        nested_config = get_invoke_subgraph_compile_options(
            decompositions=decompositions
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def gn_with_backend(x):
            return torch.sin(x)

        @torch.compiler.nested_compile_region
        def gn_without_backend(x):
            return torch.sin(x)

        def fn(x):
            return gn_with_backend(x) + gn_without_backend(x)

        backend = aot_eager_regional_inductor(
            serialize=serialize, on_invoke_subgraph=True
        )

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        res, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
        self.assertEqual(len(codes), 2)
        true_res = torch.sin(x) + torch.cos(x)
        self.assertEqual(res, true_res)

    @torch._dynamo.config.patch("enable_invoke_subgraph_regional_compile", True)
    @parametrize("serialize", [False])  # True,
    def test_invoke_subgraph_regional_compile(self, serialize):
        call_test_partitioner_ct = 0
        original_mincut_partitioner = (
            torch._functorch.partitioners.min_cut_rematerialization_partition
        )

        def test_partitioner(
            *args, **kwargs
        ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
            nonlocal call_test_partitioner_ct
            call_test_partitioner_ct += 1
            return original_mincut_partitioner(*args, **kwargs)

        # pyrefly: ignore [not-iterable]
        if serialize:
            # Callable cannot be serialized
            torch._functorch.partitioners.default_partition = test_partitioner
            partitioner = "default_partition"
        else:
            partitioner = test_partitioner

        config_patches = {
            "max_autotune": True,
            "triton.cudagraphs": False,
        }
        decompositions = {}
        nested_config = get_invoke_subgraph_compile_options(
            config_patches, decompositions, partitioner
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def gn_with_backend(x):
            return torch.sin(x)

        @torch.compiler.nested_compile_region
        def gn_without_backend(x):
            return torch.cos(x)

        def fn(x):
            return gn_with_backend(x) + gn_without_backend(x)

        backend = aot_eager_regional_inductor(
            serialize=serialize, on_invoke_subgraph=True
        )
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        try:
            x = torch.randn(8, 8, requires_grad=True)
            # opt_fn(x)
            res, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
            self.assertEqual(len(codes), 2)
            self.assertEqual(call_test_partitioner_ct, 1)
            true_res = fn(x)
            self.assertEqual(res, true_res)
        finally:
            torch._functorch.partitioners.min_cut_rematerialization_partition = (
                original_mincut_partitioner
            )

    def test_refcounts(self):
        """Tests that activations can be cleared before the end of graph"""

        class RefcountCheckPassed(Exception):
            pass

        def regional_inductor_with_refcounting(gm, *example_args):
            fn = regional_inductor_invoke_subgraph(gm, *example_args)
            assert fn._boxed_call

            def run(args: Any) -> Any:
                assert type(args) is list

                # NOTE: sys.getrefcount adds a temporary reference to the object
                # So sys.getrefcount(x) == 2 actually means we hold the single reference to x
                # There should be one activation for `fn`.
                self.assertTrue(
                    2 in [sys.getrefcount(args[i]) for i in range(len(args))]
                )
                return fn(args)

            run._boxed_call = True  # type: ignore[attr-defined]
            return run

        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):  # tensor of 1s
                act = x + x  # tensor of 2s
                ctx.save_for_backward(act)
                return act

            @staticmethod
            def backward(ctx, grad_output):  # tensor of 1s
                saved_act = ctx.saved_tensors  # tensor of 2s
                return saved_act[0] + grad_output  # tensor of 3s

        nested_config = get_invoke_subgraph_compile_options()

        @torch.compiler.nested_compile_region(options=nested_config)
        def autograd_apply(x):
            return MyAutogradFunction.apply(x)

        @torch.compile(
            backend=aot_autograd(
                fw_compiler=regional_inductor_invoke_subgraph,
                bw_compiler=regional_inductor_with_refcounting,
            ),
            fullgraph=True,
        )
        def fn(x):
            return autograd_apply(x)

        x = torch.ones(10, 10, requires_grad=True)

        fn(x).sum().backward()
        self.assertEqual(x.grad, x * 3)


@skipIfTorchDynamo("Not a suitable dynamo wrapped test")
class TestRegionalOutputCode(torch._inductor.test_case.TestCase):
    """Tests for RegionalOutputCode and BundledAOTAutogradResult."""

    def test_regional_output_code_serialization(self):
        """Test that RegionalOutputCode can be serialized and deserialized."""

        def fn(x, y):
            sin = torch.sin(x)
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Compile with regional inductor
        with (
            torch.fx.traceback.preserve_node_meta(enable=False),
            torch._functorch.config.patch(force_autograd_cache=True),
        ):
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.proxy_tensor import make_fx

            fake_mode = FakeTensorMode()
            with fake_mode:
                fake_x = fake_mode.from_tensor(x)
                fake_y = fake_mode.from_tensor(y)
                gm = make_fx(fn)(fake_x, fake_y)

            # Run regional_inductor on the graph
            output_code = regional_inductor(gm, fake_x, fake_y)

        # Create RegionalOutputCode
        self.assertIsInstance(output_code, RegionalOutputCode)

        # Test that we can call it
        self.assertIsNotNone(output_code._graph_module)

        # Serialize
        output_code.prepare_for_serialization()
        self.assertIsNone(output_code._graph_module)
        self.assertIsNotNone(output_code._serialized_graph_module)

        # Deserialize via post_compile
        from torch._inductor.output_code import CompiledFxGraphConstants

        fx_config: _CompileFxKwargs = {"is_backward": False}
        output_code.post_compile(
            [fake_x, fake_y], CompiledFxGraphConstants(), fx_config
        )
        self.assertIsNotNone(output_code._graph_module)
        self.assertIsInstance(output_code._graph_module, torch.fx.GraphModule)

        # Test that deserialized graph works
        with fake_mode:
            result = output_code([fake_x, fake_y])
            self.assertIsNotNone(result)

    def test_regional_output_code_with_backward(self):
        """Test RegionalOutputCode with both forward and backward compilation."""

        def fn(x, y):
            sin = torch.sin(x)
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Compile with regional inductor backend
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx

        fake_mode = FakeTensorMode()
        with fake_mode:
            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)

            # Create forward graph
            with (
                torch.fx.traceback.preserve_node_meta(enable=False),
                torch._functorch.config.patch(force_autograd_cache=True),
            ):
                gm = make_fx(fn)(fake_x, fake_y)
                fw_code = regional_inductor(gm, fake_x, fake_y)

        # Create forward output code
        self.assertIsInstance(fw_code, RegionalOutputCode)

        # Verify it can be called
        with fake_mode:
            result = fw_code([fake_x, fake_y])
            self.assertIsNotNone(result)

        # Test serialization round-trip
        fw_code.prepare_for_serialization()

        # Deserialize via post_compile

        from torch._inductor.output_code import CompiledFxGraphConstants

        fx_config: _CompileFxKwargs = {"is_backward": False}
        fw_code.post_compile([fake_x, fake_y], CompiledFxGraphConstants(), fx_config)

        with fake_mode:
            result2 = fw_code([fake_x, fake_y])
            self.assertIsNotNone(result2)

    def test_regional_compiled_forward_backward(self):
        """Test BundledCompiledForward and BundledCompiledBackward with RegionalOutputCode."""

        def fn(x):
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                return torch.sin(x) * 2

        x = torch.randn(5, requires_grad=True)

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx

        fake_mode = FakeTensorMode()
        with fake_mode:
            fake_x = fake_mode.from_tensor(x)

            with (
                torch.fx.traceback.preserve_node_meta(enable=False),
                torch._functorch.config.patch(force_autograd_cache=True),
            ):
                gm = make_fx(fn)(fake_x)
                compiled_gm = regional_inductor(gm, fake_x)

        # Create forward using the generic BundledCompiledForward
        self.assertIsInstance(compiled_gm, RegionalOutputCode)
        fw_compiled = BundledCompiledForward[RegionalOutputCode](result=compiled_gm)

        # Test pre_save
        fw_compiled.pre_save()
        # After pre_save, fw_compiled.result is a copy with serialized graph
        self.assertIsNotNone(fw_compiled.result._serialized_graph_module)
        self.assertIsNone(
            fw_compiled.result._graph_module
        )  # Should be cleared after serialization

        # Test load (doesn't deserialize yet)
        loaded_code = fw_compiled.load([fake_x])
        self.assertIsNone(loaded_code._graph_module)  # Not yet deserialized
        self.assertIsNotNone(loaded_code._serialized_graph_module)

        fx_config: _CompileFxKwargs = {"is_backward": False}
        post_compiled = fw_compiled.post_compile(loaded_code, fx_config)
        self.assertIsNotNone(post_compiled)
        self.assertIsNotNone(post_compiled._graph_module)  # Now deserialized


if __name__ == "__main__":
    run_tests()
