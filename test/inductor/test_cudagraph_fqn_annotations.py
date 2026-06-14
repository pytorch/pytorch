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
    disable_annotations,
    enable_annotations,
    get_kernel_annotations,
    register_fqn_annotation_hooks,
    remap_to_exec_graph,
    resolve_pending_annotations,
    save_kernel_annotations,
)
from torch.testing._internal.common_utils import run_tests, TemporaryFileName, TestCase


# ── Fixtures (ported from the external validation harness) ──────────────────
# Llama-shaped hierarchy exercising deep dotted FQNs:
#   L.model.layers.N.input_layernorm, L.model.layers.N.mlp, L.logits


class LeafModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.linear(x)                          # GEMM (cuBLAS addmm)
        h = torch.nn.functional.silu(h) * h + x    # pointwise → triton fused kernel
        return h


class LayerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = LeafModule(dim)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h = self.fc1(x)
        return h * self.scale + x   # mul + scale + add (residual)


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


# ── Profiler graph-node-id extraction (test 4) ──────────────────────────────

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
            if isinstance(ann, dict) and "str" in ann:
                out.append(ann["str"])
    return out


@unittest.skipUnless(
    torch.cuda.is_available()
    and _HAS_CUDA_BINDINGS
    and not _is_tools_id_unavailable(),
    "Requires CUDA with cudaGraphNodeGetToolsId (CUDA >= 13.1)",
)
class TestCudagraphFqnAnnotations(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        clear_kernel_annotations()

    def tearDown(self):
        clear_kernel_annotations()
        disable_annotations()
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

        enable_annotations()
        handles = register_fqn_annotation_hooks(model)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = model(static_input)
            resolve_pending_annotations()
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

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
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

    def test_save_annotations_and_join_trace(self):
        """End-to-end save -> annotate workflow: save_kernel_annotations writes the
        annotations JSON, the profiler writes the Chrome trace JSON, and
        _annotate_cuda_graph_trace.annotate_trace joins the two on the cuda graph
        node id -- producing (kernel name, graph node id, fqn) rows."""
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace
        from torch.profiler import profile, ProfilerActivity

        num_layers = 4
        model = OuterModel(dim=64, num_layers=num_layers).cuda()
        x = torch.randn(1, 64, device="cuda")

        compiled, _ = self._run_inductor_cg(model, x, annotate=True)
        self.assertTrue(dict(get_kernel_annotations()), "expected non-empty annotations")

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                compiled(x)
                torch.cuda.synchronize()

        # Artifact 1: annotations JSON written by the public save API.
        with TemporaryFileName(suffix=".json") as ann_path:
            save_kernel_annotations(ann_path)
            with open(ann_path) as f:
                annotations = {int(k): v for k, v in json.load(f).items()}

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


if __name__ == "__main__":
    run_tests()
