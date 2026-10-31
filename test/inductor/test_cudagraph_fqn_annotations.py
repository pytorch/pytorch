# Owner(s): ["module: inductor"]
"""Tests for CUDA graph kernel FQN annotation.

nn.Module fully-qualified names are attached to CUDA graph kernel nodes on two
paths:

  1. Inductor, driven by ``triton.cudagraph_kernel_annotations``: the generated
     wrapper wraps kernel calls in ``mark_kernels(fqn)``.
  2. Standalone ``torch.cuda.CUDAGraph`` capture via
     ``register_fqn_annotation_hooks``.

Annotations are keyed by graph node ``tools_id``; each value is a list of
``{"name": "<fqn>"}`` dicts.  ``TestFqnAnnotationCodegen`` checks the generated
wrapper code and only needs CUDA + Triton.  ``TestCudagraphFqnAnnotations``
checks the recorded annotations and needs ``cudaGraphNodeGetToolsId``
(CUDA >= 13.1).
"""

import json
import re
import unittest

import torch
import torch.nn as nn
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.cuda._graph_annotations import (
    _HAS_CUDA_BINDINGS,
    _is_tools_id_unavailable,
    _reset_kernel_annotations,
    get_kernel_annotations,
    register_fqn_annotation_hooks,
    save_kernel_annotations,
)
from torch.testing._internal.common_utils import skipIfRocm, TemporaryFileName
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


# --- Fixtures ---
# Llama-shaped hierarchy exercising deep dotted FQNs:
#   L.model.layers.N.fc1.linear, L.logits


class LeafModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.linear(x)  # GEMM (cuBLAS addmm)
        h = torch.nn.functional.silu(h) * h + x  # pointwise -> triton fused kernel
        return h


class LayerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = LeafModule(dim)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h = self.fc1(x)
        return h * self.scale + x  # mul + scale + add (residual)


class InnerModel(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([LayerBlock(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OuterModel(nn.Module):
    def __init__(self, dim=64, num_layers=4):
        super().__init__()
        self.model = InnerModel(dim, num_layers)
        self.logits = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.logits(self.model(x))


# CNN hierarchy exercising ModuleList indices: L.networks.N.conv, L.classifier


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(1, 16, 32, 32))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return (self.relu(self.conv(x)) + self.bias) * self.scale


class CNNEnsemble(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.networks = nn.ModuleList(
            [SimpleCNN(in_channels=3)]
            + [SimpleCNN(in_channels=16) for _ in range(n - 1)]
        )
        self.classifier = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        for net in self.networks:
            x = net(x)
        return self.classifier(x.view(x.size(0), -1))


# Attention model for testing exact per-kernel annotation attribution.
# SharedMaskAttn shares a causal mask across layers; without the
# collect_compute_fx_nodes fix, the mask kernel claims qkv ops it only reads.


class _AttnLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, D // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        o = torch.nn.functional.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=mask
        )
        o = o.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.proj(o)


class SharedMaskAttn(nn.Module):
    def __init__(self, n_layers: int = 2, dim: int = 32, n_heads: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_AttnLayer(dim, n_heads) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[1]
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        for layer in self.layers:
            x = x + layer(x, mask)
        return x


class GraphBreakLayers(nn.Module):
    """Structurally identical layers split into separate graphs, so the FX
    graph cache sees identical graphs that differ only in nn_module_stack."""

    def __init__(self, dim=32):
        super().__init__()
        self.layers = nn.ModuleList([LayerBlock(dim) for _ in range(2)])

    def forward(self, x):
        x = self.layers[0](x)
        torch._dynamo.graph_break()
        return self.layers[1](x)


def _all_fqn_strings(annotations) -> list:
    return [
        ann["name"]
        for ann_list in annotations.values()
        for ann in ann_list
        if isinstance(ann, dict) and ann.get("name")
    ]


def _expected_shared_mask_attn_fqns(num_layers: int) -> set[str]:
    expected: set[str] = set()
    for i in range(num_layers):
        expected |= {
            f"L.layers.{i}.qkv.addmm",
            f"L.layers.{i}.where + L.layers.{i}.full_default",
            f"L.layers.{i}._scaled_dot_product_efficient_attention",
            f"L.layers.{i}.proj.mm_default",
            f"L.layers.{i}.proj.add_tensor",
        }
    return expected


def _assert_per_layer_fused_and_extern(test, all_strs, num_layers, prefix):
    """Every layer has a fused-Triton ('+'-joined) and an extern annotation,
    and every annotation carries the 'L.' prefix."""
    test.assertEqual([s for s in all_strs if not s.startswith("L.")], [])
    for i in range(num_layers):
        layer_strs = [s for s in all_strs if f"{prefix}{i}." in s]
        test.assertTrue(
            [s for s in layer_strs if " + " in s],
            f"layer {i} has no fused-Triton annotation; saw: {layer_strs}",
        )
        test.assertTrue(
            [s for s in layer_strs if " + " not in s],
            f"layer {i} has no extern annotation; saw: {layer_strs}",
        )


@unittest.skipUnless(HAS_CUDA_AND_TRITON, "Requires CUDA + Triton")
class TestFqnAnnotationCodegen(TestCase):
    """Checks the mark_kernels() calls in the generated wrapper code."""

    def _wrapper_code(self, model, *inputs, annotate=True, fullgraph=True) -> str:
        with (
            config.patch({"triton.cudagraph_kernel_annotations": annotate}),
            torch.no_grad(),
        ):
            compiled = torch.compile(model, fullgraph=fullgraph)
            _, codes = run_and_get_code(compiled, *inputs)
        return "\n".join(codes)

    def _mark_kernels_args(self, model, *inputs, **kwargs) -> list[str]:
        code = self._wrapper_code(model, *inputs, **kwargs)
        return re.findall(r"mark_kernels\('([^']*)'\)", code)

    def test_each_layer_has_fused_triton_and_extern_annotation(self):
        model = OuterModel(dim=64, num_layers=2).cuda()
        x = torch.randn(1, 64, device="cuda")
        all_strs = self._mark_kernels_args(model, x)
        _assert_per_layer_fused_and_extern(self, all_strs, 2, "L.model.layers.")

    def test_every_kernel_call_annotated(self):
        # Kernels that only serve an extern op, such as the channels-last
        # copies of a convolution's input and weight, carry its FQN too.
        model = CNNEnsemble(n=2).cuda()
        x = torch.randn(4, 3, 32, 32, device="cuda")
        lines = self._wrapper_code(model, x).splitlines()
        unannotated = []
        for i, line in enumerate(lines):
            if not re.search(r"\.run\(|= extern_kernels\.|^\s+extern_kernels\.", line):
                continue
            indent = len(line) - len(line.lstrip())
            parent = next(
                p
                for p in reversed(lines[:i])
                if p.strip() and len(p) - len(p.lstrip()) < indent
            )
            if "mark_kernels(" not in parent:
                unannotated.append(line.strip())
        self.assertEqual(unannotated, [])

    def test_no_annotations_when_flag_off(self):
        model = OuterModel(dim=64, num_layers=2).cuda()
        x = torch.randn(1, 64, device="cuda")
        self.assertEqual(self._mark_kernels_args(model, x, annotate=False), [])

    @skipIfRocm(msg="expects the efficient-attention SDPA backend")
    def test_kernel_annotations_match_expected(self):
        model = SharedMaskAttn(n_layers=2, dim=32, n_heads=2).cuda()
        x = torch.randn(1, 16, 32, device="cuda")
        all_strs = set(self._mark_kernels_args(model, x))
        self.assertEqual(all_strs, _expected_shared_mask_attn_fqns(2))

    def test_fx_graph_cache_keeps_per_layer_fqns(self):
        # layers.0 and layers.1 compile to identical graphs that differ only in
        # nn_module_stack. With annotations on, that must be in the FX graph
        # cache key, or layers.1 reuses layers.0's code and its FQNs.
        model = GraphBreakLayers(dim=32).cuda()
        x = torch.randn(1, 32, device="cuda")
        with functorch_config.patch(enable_autograd_cache=False):
            # Control: with annotations off, layers.1 hits layers.0's entry.
            counters.clear()
            self._mark_kernels_args(model, x, annotate=False, fullgraph=False)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

            counters.clear()
            all_strs = self._mark_kernels_args(model, x, fullgraph=False)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        for i in range(2):
            self.assertTrue(
                any(s.startswith(f"L.layers.{i}.") for s in all_strs),
                f"no annotation for layers.{i}; saw {sorted(set(all_strs))}",
            )

    def test_user_triton_kernel_not_annotated_with_stale_fqn(self):
        # A user Triton kernel after a SIMD kernel must not inherit its FQN.
        import triton
        import triton.language as tl

        @triton.jit
        def add_one(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
            offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

        class Act(nn.Module):
            def forward(self, x):
                return x.relu()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = Act()

            def forward(self, x):
                y = self.act(x)
                out = torch.empty_like(y)
                add_one[(y.numel(),)](y, out, y.numel(), BLOCK_SIZE=16)
                return out

        x = torch.randn(64, device="cuda")
        # Only the relu SIMD kernel is annotated, not add_one.
        self.assertEqual(len(self._mark_kernels_args(Model(), x)), 1)


@unittest.skipUnless(
    torch.cuda.is_available() and _HAS_CUDA_BINDINGS and not _is_tools_id_unavailable(),
    "Requires CUDA with cudaGraphNodeGetToolsId (CUDA >= 13.1)",
)
class TestCudagraphFqnAnnotations(TestCase):
    """Checks the annotations recorded on captured CUDA graph nodes."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        _reset_kernel_annotations()

    def tearDown(self):
        _reset_kernel_annotations()
        torch._dynamo.reset()
        super().tearDown()

    def _run_inductor_cg(self, model, x, *, annotate, num_warmup=3):
        patches = {
            "triton.cudagraphs": True,
            "triton.cudagraph_kernel_annotations": annotate,
        }
        with config.patch(patches), torch.no_grad():
            compiled = torch.compile(model, fullgraph=True)
            for _ in range(num_warmup):
                out = compiled(x)
                torch.cuda.synchronize()
        return compiled, out

    def test_annotations_populated_after_first_call(self):
        num_layers = 4
        model = OuterModel(dim=64, num_layers=num_layers).cuda()
        x = torch.randn(1, 64, device="cuda")

        self._run_inductor_cg(model, x, annotate=True)

        all_strs = _all_fqn_strings(dict(get_kernel_annotations()))
        missing = [
            i
            for i in range(num_layers)
            if not any(f"L.model.layers.{i}." in s for s in all_strs)
        ]
        self.assertEqual(
            missing,
            [],
            f"missing full-path FQNs for layers {missing}; saw {sorted(set(all_strs))}",
        )

    def test_each_layer_has_fused_triton_and_extern_annotation(self):
        """Every block must have both a fused-Triton annotation and an
        extern-mm annotation, each carrying the 'L.' prefix.

        OuterModel produces both kernel types per block: nn.Linear -> cuBLAS mm
        (extern) and silu(h)*h+x -> fused Triton pointwise. The fused Triton
        annotation is a multi-op '+'-joined string; the extern mm is a single op.
        This also guards the last block's fused kernel, which was once left
        unannotated when its extern mm was codegen'd first.
        """
        num_layers = 4
        model = OuterModel(dim=64, num_layers=num_layers).cuda()
        x = torch.randn(1, 64, device="cuda")

        self._run_inductor_cg(model, x, annotate=True)

        all_strs = _all_fqn_strings(dict(get_kernel_annotations()))
        _assert_per_layer_fused_and_extern(
            self, all_strs, num_layers, "L.model.layers."
        )

    def test_annotations_disabled_when_flag_off(self):
        model = OuterModel(dim=64, num_layers=4).cuda()
        x = torch.randn(1, 64, device="cuda")

        self._run_inductor_cg(model, x, annotate=False)

        self.assertEqual(dict(get_kernel_annotations()), {})

    def test_register_fqn_annotation_hooks_native_path(self):
        model = CNNEnsemble(n=4).cuda()
        static_input = torch.randn(4, 3, 32, 32, device="cuda")

        # Warmup on a side stream before capture.
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                model(static_input)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        handles = register_fqn_annotation_hooks(model)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, enable_annotations=True):
            static_output = model(static_input)
        for h in handles:
            h.remove()

        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        self.assertEqual(static_output.shape, (4, 10))

        all_strs = _all_fqn_strings(dict(get_kernel_annotations()))
        self.assertTrue(
            any("L.networks." in s for s in all_strs),
            f"expected networks.* FQNs; saw {sorted(set(all_strs))}",
        )

    def test_profiler_path_recovers_fqn(self):
        from torch.profiler import profile, ProfilerActivity

        model = OuterModel(dim=64, num_layers=4).cuda()
        x = torch.randn(1, 64, device="cuda")

        compiled, _ = self._run_inductor_cg(model, x, annotate=True)
        annotations = dict(get_kernel_annotations())
        self.assertTrue(annotations, "expected non-empty kernel annotations")

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                compiled(x)
                torch.cuda.synchronize()

        # The CUPTI field "graph node id" on kernel events carries the same id
        # that keys the annotations.
        with TemporaryFileName(suffix=".json") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

        kernel_events = [
            e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel"
        ]
        recovered: dict[int, str] = {}
        for ke in kernel_events:
            raw = ke.get("args", {}).get("graph node id")
            try:
                gid = int(raw)
            except (TypeError, ValueError):
                continue
            for ann in annotations.get(gid, []):
                if isinstance(ann, dict) and ann.get("name"):
                    recovered[gid] = ann["name"]

        seen_arg_keys = sorted({k for ke in kernel_events for k in ke.get("args", {})})
        self.assertTrue(
            recovered,
            f"No FQN recovered from Chrome trace kernel events. "
            f"Kernel event arg keys seen: {seen_arg_keys}",
        )
        self.assertTrue(
            any("L.model.layers." in v for v in recovered.values()),
            f"recovered FQNs lack layer hierarchy: {sorted(set(recovered.values()))}",
        )

    def test_save_kernel_annotations(self):
        import pickle

        model = OuterModel(dim=64, num_layers=4).cuda()
        x = torch.randn(1, 64, device="cuda")

        self._run_inductor_cg(model, x, annotate=True)
        self.assertTrue(dict(get_kernel_annotations()), "expected annotations")

        with TemporaryFileName(suffix=".pkl") as ann_path:
            save_kernel_annotations(ann_path)
            with open(ann_path, "rb") as f:
                saved = pickle.load(f)
        self.assertTrue(saved, "expected non-empty saved annotations")
        self.assertTrue(
            any(
                isinstance(v, list) and v and isinstance(v[0], dict)
                for v in saved.values()
            ),
            "saved annotations must be lists of dicts",
        )

    def test_kernel_annotations_match_expected(self):
        """The full set of kernel FQN annotations must match exactly.

        SharedMaskAttn compiles, per layer, to: an extern qkv addmm, a fused
        Triton causal-mask kernel (``where`` + ``full_default``), an extern SDPA,
        an extern proj mm, and a Triton proj bias-add. The key over-attribution
        guard is the mask kernel: its annotation must be exactly
        ``where + full_default``, not also the qkv view/permute/split ops it
        only reads.
        """
        model = SharedMaskAttn(n_layers=2, dim=32, n_heads=2).cuda()
        x = torch.randn(1, 16, 32, device="cuda")

        self._run_inductor_cg(model, x, annotate=True)
        all_strs = set(_all_fqn_strings(dict(get_kernel_annotations())))
        self.assertEqual(all_strs, _expected_shared_mask_attn_fqns(2))


class TestGraphViewHelpers(TestCase):
    """Unit tests for _clean_stack_name and _strip_instance_suffix."""

    def test_clean_stack_name_module_subscript(self):
        from torch._inductor.fx_passes.graph_view import _clean_stack_name

        inp = "L['self']._modules['layers']['0']._modules['attention']"
        self.assertEqual(_clean_stack_name(inp), "layers.0.attention")

    def test_clean_stack_name_dot_attr(self):
        from torch._inductor.fx_passes.graph_view import _clean_stack_name

        self.assertEqual(
            _clean_stack_name("L['self'].networks.1.conv"), "networks.1.conv"
        )

    def test_clean_stack_name_root(self):
        from torch._inductor.fx_passes.graph_view import _clean_stack_name

        self.assertEqual(_clean_stack_name("L['self']"), "")

    def test_strip_instance_suffix_removes_trailing_digit(self):
        from torch._inductor.fx_passes.graph_view import _strip_instance_suffix

        self.assertEqual(_strip_instance_suffix("convolution_1"), "convolution")
        self.assertEqual(_strip_instance_suffix("addmm_3"), "addmm")

    def test_strip_instance_suffix_no_suffix(self):
        from torch._inductor.fx_passes.graph_view import _strip_instance_suffix

        self.assertEqual(_strip_instance_suffix("linear"), "linear")

    def test_strip_instance_suffix_keeps_mid_digit(self):
        from torch._inductor.fx_passes.graph_view import _strip_instance_suffix

        # digits in the middle of a name are not stripped
        self.assertEqual(_strip_instance_suffix("conv2d"), "conv2d")


if __name__ == "__main__":
    run_tests()
