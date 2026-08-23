# Owner(s): ["module: inductor"]
"""End-to-end tests for CUDA graph kernel FQN annotation.

These exercise the two supported paths for attaching nn.Module fully-qualified
names to CUDA graph kernel nodes:

  1. Inductor cudagraph trees, driven by
     ``triton.cudagraph_kernel_annotations`` (tests 1, 2, 4).
  2. Standalone ``torch.cuda.CUDAGraph`` capture via
     ``register_fqn_annotation_hooks`` (test 3).

Annotations are recorded keyed by the graph node ``tools_id`` and each value is
a list of ``{"str": "<fqn>"}`` dicts (see ``mark_kernels``).  All tests require
CUDA with ``cudaGraphNodeGetToolsId`` (CUDA >= 13.1) and are skipped otherwise.
"""

import json
import unittest

import torch
import torch.nn as nn
from torch._inductor import config
from torch.cuda._graph_annotations import (
    _HAS_CUDA_BINDINGS,
    _is_tools_id_unavailable,
    clear_kernel_annotations,
    get_kernel_annotations,
    register_fqn_annotation_hooks,
    remap_to_exec_graph,
    save_kernel_annotations,
)
from torch.testing._internal.common_utils import run_tests, TemporaryFileName, TestCase


# --- Fixtures (ported from the external validation harness) ---
# Llama-shaped hierarchy exercising deep dotted FQNs:
#   L.model.layers.N.input_layernorm, L.model.layers.N.mlp, L.logits


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


# --- Profiler graph-node-id extraction (test 4) ---
# Candidate metadata keys under which CUPTI/kineto may surface the cuda graph
# node id on a kernel event.  The first GPU run confirms the real key (the test
# dumps observed keys on miss, see test_profiler_path_recovers_fqn).
_GRAPH_NODE_ID_KEYS = (
    "Cuda Graph Node Id",
    "cuda graph node id",
    "graph node id",
    "graphNodeId",
    "graph_node_id",
    "Cuda Graph Id",
    "cudaGraphId",
)


def _event_metadata(evt) -> dict:
    """Merge an in-memory kineto event's extra_meta() and metadata_json()."""
    md: dict = {}
    try:
        em = evt.extra_meta()
        if em:
            md.update(em)
    except Exception:
        pass
    try:
        mj = evt.metadata_json()
        if mj:
            s = mj.strip()
            if not s.startswith("{"):
                s = "{" + s + "}"
            md.update(json.loads(s))
    except Exception:
        pass
    return md


def _kernel_graph_node_id(evt):
    """Return the cuda graph node id for a kineto kernel event, or None.

    TODO(discovery): confirm on CUDA 13.1 hardware which key CUPTI/kineto uses.
    Both nsys and the torch profiler read this from the same CUPTI activity
    record, and the value must equal the ``tools_id`` that keys
    ``get_kernel_annotations()`` after ``remap_to_exec_graph``.
    """
    md = _event_metadata(evt)
    for k in _GRAPH_NODE_ID_KEYS:
        if k in md:
            try:
                return int(md[k])
            except (TypeError, ValueError):
                pass
    return None


def _all_fqn_strings(annotations) -> list:
    out = []
    for ann_list in annotations.values():
        for ann in ann_list:
            if isinstance(ann, dict):
                fqn = ann.get("str") or ann.get("module_name")
                if fqn:
                    out.append(fqn)
    return out


@unittest.skipUnless(
    torch.cuda.is_available() and _HAS_CUDA_BINDINGS and not _is_tools_id_unavailable(),
    "Requires CUDA with cudaGraphNodeGetToolsId (CUDA >= 13.1)",
)
class TestCudagraphFqnAnnotations(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        clear_kernel_annotations()

    def tearDown(self):
        clear_kernel_annotations()
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

        annotations = dict(get_kernel_annotations())
        self.assertTrue(annotations, "expected non-empty kernel annotations")

        all_strs = _all_fqn_strings(annotations)
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
        """Every block must have BOTH a fused-Triton annotation and an extern-mm
        annotation, each carrying the 'L.' prefix.

        This catches two bugs introduced together during code review:

        1. Extern kernels (cuBLAS mm from nn.Linear) were annotated without
           the 'L.' anchor prefix, making them inconsistent with Triton kernels.

        2. The last block's fused Triton kernel was silently unannotated.
           The inductor scheduler codegen'd the last block's extern mm before
           its Triton fused kernel; at that point the extern FQN was present in
           fx_extern_fqns, which caused get_fused_kernel_module_fqn to return
           None for the Triton kernel, suppressing its AnnotatedKernelCallLine.

        OuterModel produces both kernel types per block: nn.Linear -> cuBLAS mm
        (extern) and silu(h)*h+x -> fused Triton pointwise. The fused Triton
        annotation is a multi-op '+'-joined string; the extern mm is a single op.
        """
        num_layers = 4
        model = OuterModel(dim=64, num_layers=num_layers).cuda()
        x = torch.randn(1, 64, device="cuda")

        self._run_inductor_cg(model, x, annotate=True)

        annotations = dict(get_kernel_annotations())
        self.assertTrue(annotations, "expected non-empty kernel annotations")
        all_strs = _all_fqn_strings(annotations)

        missing_prefix = [s for s in all_strs if not s.startswith("L.")]
        self.assertEqual(
            missing_prefix,
            [],
            f"FQNs missing 'L.' prefix (likely extern kernels): {missing_prefix}",
        )

        for i in range(num_layers):
            layer_strs = [s for s in all_strs if f"L.model.layers.{i}." in s]
            fused = [s for s in layer_strs if " + " in s]
            extern = [s for s in layer_strs if " + " not in s]
            self.assertTrue(
                fused,
                f"layer {i} has no fused-Triton annotation; saw: {layer_strs}",
            )
            self.assertTrue(
                extern,
                f"layer {i} has no extern-mm annotation; saw: {layer_strs}",
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
        remap_to_exec_graph(g)

        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        self.assertEqual(static_output.shape, (4, 10))

        annotations = dict(get_kernel_annotations())
        self.assertTrue(annotations, "expected non-empty kernel annotations")
        all_strs = _all_fqn_strings(annotations)
        self.assertTrue(
            any("L.networks." in s for s in all_strs),
            f"expected networks.* FQNs; saw {sorted(set(all_strs))}",
        )

        # Hooks must be removed after capture so replay carries no overhead.
        remaining = sum(
            len(m._forward_pre_hooks) + len(m._forward_hooks) for m in model.modules()
        )
        self.assertEqual(remaining, 0, "annotation hooks were not cleaned up")

    def test_profiler_path_recovers_fqn(self):
        from torch.profiler import profile, ProfilerActivity

        num_layers = 4
        model = OuterModel(dim=64, num_layers=num_layers).cuda()
        x = torch.randn(1, 64, device="cuda")

        compiled, _ = self._run_inductor_cg(model, x, annotate=True)
        annotations = dict(get_kernel_annotations())
        self.assertTrue(annotations, "expected non-empty kernel annotations")

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                compiled(x)
                torch.cuda.synchronize()

        # Export to Chrome trace JSON and read kernel events.  The CUPTI field
        # "graph node id" carries the same graphNodeId that keyed our annotations.
        # Write to /artifacts/ when available (CI), otherwise use a temp file.
        import os

        artifacts_dir = "/artifacts"
        use_artifacts = os.path.isdir(artifacts_dir) and os.access(
            artifacts_dir, os.W_OK
        )

        if use_artifacts:
            trace_path = os.path.join(
                artifacts_dir, "cuda_graph_fqn_profiler_trace.json"
            )
            prof.export_chrome_trace(trace_path)
            with open(trace_path) as f:
                trace = json.load(f)
        else:
            with TemporaryFileName(suffix=".json") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    trace = json.load(f)

        # Save annotations (graph_node_id -> [{str: fqn}]) for post-processing.
        if use_artifacts:
            annotations_path = os.path.join(
                artifacts_dir, "cuda_graph_fqn_annotations.json"
            )
            with open(annotations_path, "w") as f:
                json.dump({str(k): v for k, v in annotations.items()}, f, indent=2)

        kernel_events = [
            e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel"
        ]
        recovered: dict[int, str] = {}
        for ke in kernel_events:
            args = ke.get("args", {})
            raw = args.get("graph node id")
            if raw is None:
                continue
            try:
                gid = int(raw)
            except (TypeError, ValueError):
                continue
            if gid in annotations:
                for ann in annotations[gid]:
                    if isinstance(ann, dict) and "str" in ann:
                        recovered[gid] = ann["str"]

        seen_arg_keys = sorted({k for ke in kernel_events for k in ke.get("args", {})})
        self.assertTrue(
            recovered,
            f"No FQN recovered from Chrome trace kernel events. "
            f"Kernel event arg keys seen: {seen_arg_keys}",
        )
        # The profiler path must reproduce the nsys result: per-layer FQNs.
        self.assertTrue(
            any("L.model.layers." in v for v in recovered.values()),
            f"recovered FQNs lack layer hierarchy: {sorted(set(recovered.values()))}",
        )

    def test_extern_and_triton_fqns_share_L_prefix(self):
        """Extern kernels (e.g. cuBLAS addmm from nn.Linear) and Triton fused
        kernels must both carry the 'L.' prefix in their FQN annotations.

        OuterModel produces both: nn.Linear -> cuBLAS addmm (extern kernel) and
        silu*h+x -> Triton fused kernel.  Without the fix, extern kernels emit
        e.g. "model.layers.0.fc1.linear.addmm" while Triton kernels emit
        "L.model.layers.0.fc1.linear", making the two namespaces inconsistent.
        """
        model = OuterModel(dim=64, num_layers=2).cuda()
        x = torch.randn(1, 64, device="cuda")
        self._run_inductor_cg(model, x, annotate=True)
        annotations = dict(get_kernel_annotations())
        self.assertTrue(annotations, "expected non-empty kernel annotations")
        all_strs = _all_fqn_strings(annotations)
        self.assertTrue(all_strs, "expected non-empty FQN strings in annotations")
        missing_prefix = [s for s in all_strs if not s.startswith("L.")]
        self.assertEqual(
            missing_prefix,
            [],
            f"FQNs missing 'L.' prefix (likely extern kernels): {missing_prefix}",
        )

    def test_save_annotations_and_join_trace(self):
        """End-to-end save -> annotate workflow: save_kernel_annotations writes the
        annotations pickle, the profiler writes the Chrome trace JSON, and
        _annotate_cuda_graph_trace.annotate_trace joins the two on the cuda graph
        node id -- producing (kernel name, graph node id, fqn) rows."""
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace
        from torch.profiler import profile, ProfilerActivity

        num_layers = 4
        model = OuterModel(dim=64, num_layers=num_layers).cuda()
        x = torch.randn(1, 64, device="cuda")

        compiled, _ = self._run_inductor_cg(model, x, annotate=True)
        self.assertTrue(
            dict(get_kernel_annotations()), "expected non-empty annotations"
        )

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                compiled(x)
                torch.cuda.synchronize()

        # Artifact 1: annotations pickle written by the public save API.
        import pickle

        with TemporaryFileName(suffix=".pkl") as ann_path:
            save_kernel_annotations(ann_path)
            with open(ann_path, "rb") as f:
                annotations = pickle.load(f)

        # Artifact 2: Chrome trace JSON (kernel events carry "graph node id").
        with TemporaryFileName(suffix=".json") as trace_path:
            prof.export_chrome_trace(trace_path)
            with open(trace_path) as f:
                trace = json.load(f)

        # Join the two on the cuda graph node id (annotate_trace merges the fqn
        # into each matching kernel event's args).
        annotated = annotate_trace(trace, annotations)
        self.assertGreater(annotated, 0, "no kernel events joined on graph node id")

        # The joined result is a (kernel name, graph node id, fqn) table.
        table = [
            (e.get("name"), e["args"]["graph node id"], e["args"]["str"])
            for e in trace["traceEvents"]
            if isinstance(e.get("args"), dict)
            and "str" in e["args"]
            and e["args"].get("graph node id")
        ]
        self.assertTrue(table, "join produced no (kernel, graph node id, fqn) rows")
        fqns = [row[2] for row in table]
        self.assertTrue(
            any("L.model.layers." in s for s in fqns),
            f"joined table lacks per-layer FQNs; saw {sorted(set(fqns))[:8]}",
        )

    def test_shared_mask_attn_annotation_discovery(self):
        """Discovery: print actual annotation strings for the AIB log.
        Used to determine expected values for test_kernel_annotations_match_expected.
        Run with cudagraph_fqn_compute_tracking=True to see the new path's output."""
        model = SharedMaskAttn(n_layers=2, dim=32, n_heads=2).cuda()
        x = torch.randn(1, 16, 32, device="cuda")
        patches = {
            "triton.cudagraphs": True,
            "triton.cudagraph_kernel_annotations": True,
            "triton.cudagraph_fqn_compute_tracking": True,
        }
        with config.patch(patches), torch.no_grad():
            compiled = torch.compile(model, fullgraph=True)
            for _ in range(3):
                compiled(x)
                torch.cuda.synchronize()
        all_strs = sorted(_all_fqn_strings(dict(get_kernel_annotations())))
        for s in all_strs:
            print(f"  annotation: {s!r}")
        self.assertTrue(all_strs, "expected non-empty annotations")

        # Verify no annotation spans FQNs from two different layer indices.
        # Cross-layer bleed is the specific regression this fix prevents.
        import re as _re

        for s in all_strs:
            layer_indices = {
                m.group(1)
                for part in s.split(" + ")
                for m in [_re.search(r"L\.layers\.(\d+)", part)]
                if m
            }
            self.assertLessEqual(
                len(layer_indices),
                1,
                f"annotation spans multiple layers (cross-layer bleed): {s!r}",
            )


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
