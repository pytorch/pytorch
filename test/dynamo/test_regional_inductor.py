# Owner(s): ["module: dynamo"]

import functools
from typing import TYPE_CHECKING

import torch
import torch._inductor.test_case
import torch.fx.traceback as fx_traceback
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.autograd_cache import BundledCompiledForward
from torch._guards import detect_fake_mode
from torch._inductor.output_code import RegionalOutputCode
from torch._inductor.test_case import run_tests
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.fx._graph_pickler import GraphPickler
from torch.fx.passes.regional_inductor import regional_inductor
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


def aot_eager_regional_inductor(serialize=False):
    if serialize:

        def regional_inductor_pickle(gm, *example_args):
            result = regional_inductor(gm, *example_args)
            serialized = GraphPickler.dumps(result)

            fake_mode = detect_fake_mode(example_args)
            assert fake_mode is not None
            # Serialize and deserialize the result to confirm pickling works
            # Use a fresh tracing context on the new process
            context = torch._guards.TracingContext(fake_mode)
            with torch._guards.tracing(context):
                result = GraphPickler.loads(serialized, fake_mode)
                assert isinstance(result, torch.fx.GraphModule)
                result.recompile()
                return result

        return aot_autograd(
            fw_compiler=regional_inductor_pickle,
            bw_compiler=regional_inductor_pickle,
        )

    return aot_autograd(
        fw_compiler=regional_inductor,
        bw_compiler=regional_inductor,
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
        with torch.fx.traceback.preserve_node_meta(enable=False):
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.proxy_tensor import make_fx

            fake_mode = FakeTensorMode()
            with fake_mode:
                fake_x = fake_mode.from_tensor(x)
                fake_y = fake_mode.from_tensor(y)
                gm = make_fx(fn)(fake_x, fake_y)

            # Run regional_inductor on the graph
            result_gm = regional_inductor(gm, fake_x, fake_y)

        # Create RegionalOutputCode
        output_code = RegionalOutputCode(result_gm)

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
            with torch.fx.traceback.preserve_node_meta(enable=False):
                gm = make_fx(fn)(fake_x, fake_y)
                forward_gm = regional_inductor(gm, fake_x, fake_y)

        # Create forward output code
        fw_code = RegionalOutputCode(forward_gm)

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

            with torch.fx.traceback.preserve_node_meta(enable=False):
                gm = make_fx(fn)(fake_x)
                compiled_gm = regional_inductor(gm, fake_x)

        # Create forward using the generic BundledCompiledForward
        fw_code = RegionalOutputCode(compiled_gm)
        fw_compiled = BundledCompiledForward[RegionalOutputCode](result=fw_code)

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
