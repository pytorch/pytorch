# Owner(s): ["module: cuda graphs"]

"""Saving a captured CUDA graph to an archive.

Every test runs the producing side in a subprocess: saving requires cubin capture
armed before any CUDA work and the whole process on expandable segments, neither of
which can be arranged inside a shared test process that has already allocated.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

import torch
from torch.testing._internal.common_utils import (
    requires_cuda,
    requires_cuda_python_bindings,
    run_tests,
    skipIfRocm,
    TestCase,
)


def _kernel_capture_available():
    try:
        from torch.cuda import _graph_kernel_capture
    except ImportError:
        return False
    return _graph_kernel_capture.is_available()


# Captures a matmul into a graph and saves it, with the static input/output tensors.
_SAVE_SCRIPT = """
import torch
from torch.cuda import _graph_kernel_capture as cap
assert cap.start(), "could not arm cubin capture"

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
out = torch.empty(1024, 1024, device="cuda", dtype=torch.bfloat16)

def work():
    torch.mm(a, a, out=out)

stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):
        work()
torch.cuda.current_stream().wait_stream(stream)

graph = torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(graph):
    work()
{save_call}
print("SAVED")
"""


@skipIfRocm
@requires_cuda
@requires_cuda_python_bindings
@unittest.skipIf(
    not _kernel_capture_available(), "requires cupti-python and a usable CUPTI"
)
class TestCUDAGraphSave(TestCase):
    def _run(self, script, *, expandable=True, expect_success=True):
        env = os.environ.copy()
        if expandable:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        else:
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        proc = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True
        )
        output = proc.stdout.decode() + proc.stderr.decode()
        if expect_success and proc.returncode != 0:
            self.fail(f"subprocess failed:\n{output}")
        return output

    def _save_script(self, save_call):
        return _SAVE_SCRIPT.format(save_call=save_call)

    def _manifest(self, path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            manifest = next(n for n in names if n.endswith("manifest.json"))
            return json.loads(archive.read(manifest)), [
                n.split("/", 1)[1] for n in names
            ]

    def test_archive_carries_graph_kernels_memory_and_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.ptcg")
            self._run(self._save_script(f"graph.save({path!r}, tensors=[a, out])"))
            manifest, records = self._manifest(path)

            self.assertEqual(manifest["version"], 1)
            # a single matmul: one kernel node, no edges
            self.assertEqual(len(manifest["nodes"]), 1)
            node = manifest["nodes"][0]
            self.assertEqual(node["type"], "kernel")
            # cuBLASLt passes its arguments as one packed buffer, not per-parameter
            self.assertIsNone(node["args"])
            self.assertGreater(len(node["packed_args"]), 0)
            self.assertIn(
                "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", node["func_attrs"]
            )
            self.assertIn("CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION", node["node_attrs"])

            # exactly the modules the graph's kernels live in, not every module loaded
            self.assertEqual(set(manifest["kernels"]), {node["name"]})
            cubins = [r for r in records if r.startswith("cubins/")]
            self.assertEqual(len(cubins), len(set(manifest["kernels"].values())))

            # memory layout, at addresses a later process can reserve. Contents are
            # deliberately absent: parameters evolve after capture, so bytes written
            # here would be stale, and the caller fills the addresses at load.
            self.assertGreater(len(manifest["segments"]), 0)
            self.assertFalse(any(r.startswith("segments/") for r in records))
            for segment in manifest["segments"]:
                self.assertGreater(segment["expandable_segment_base"], 0)
                self.assertGreater(len(segment["blocks"]), 0)
                for block in segment["blocks"]:
                    self.assertIn(
                        block["state"],
                        ("active_allocated", "active_pending_free", "inactive"),
                    )

            self.assertEqual(len(manifest["tensors"]), 2)
            for tensor in manifest["tensors"]:
                self.assertEqual(tensor["dtype"], "torch.bfloat16")
                self.assertEqual(tensor["shape"], [1024, 1024])
                # each tensor must live in a segment the archive carries
                self.assertTrue(
                    any(
                        s["address"]
                        <= tensor["address"]
                        < s["address"] + s["total_size"]
                        for s in manifest["segments"]
                    )
                )
            # Tensor records are metadata only: the bytes are already inside the
            # segment blobs, so writing them again would double the archive.
            self.assertFalse(any(r.startswith("tensors/") for r in records))

    def test_warns_when_carrying_memory_the_caller_did_not_list(self):
        # Static inputs are normally passed in `tensors`, which is what records the
        # metadata to place them and their contents. Memory the graph reads that was
        # not listed is still carried, so the archive is complete, but it is called
        # out: it means scratch space outside the graph's pool.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.ptcg")
            out = self._run(
                self._save_script(
                    "import warnings\n"
                    "with warnings.catch_warnings(record=True) as caught:\n"
                    "    warnings.simplefilter('always')\n"
                    f"    graph.save({path!r})\n"
                    "print('WARNINGS', [str(w.message) for w in caught])"
                )
            )
            self.assertIn("because the graph reads it", out)
            # and listing the tensors silences it
            quiet = self._run(
                self._save_script(
                    "import warnings\n"
                    "with warnings.catch_warnings(record=True) as caught:\n"
                    "    warnings.simplefilter('always')\n"
                    f"    graph.save({path!r}, tensors=[a, out])\n"
                    "print('WARNINGS', [str(w.message) for w in caught])"
                )
            )
            self.assertNotIn("because the graph reads it", quiet)

    def test_saves_before_instantiate_and_after_later_modification(self):
        # keep_graph=True: the caller may modify the template and instantiate much
        # later, so save must work at any point while the template is live.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.ptcg")
            out = self._run(
                self._save_script(
                    "assert not graph._has_graph_exec\n"
                    f"graph.save({path!r})\n"
                    "graph.instantiate()\n"
                    "graph.replay()\n"
                    "torch.cuda.synchronize()"
                )
            )
            self.assertIn("SAVED", out)
            manifest, _ = self._manifest(path)
            self.assertEqual(len(manifest["nodes"]), 1)
            self.assertEqual(manifest["tensors"], [])

    def test_post_instantiate_hook_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.ptcg")
            script = (
                _SAVE_SCRIPT.format(save_call="").replace(
                    "graph = torch.cuda.CUDAGraph(keep_graph=True)",
                    "graph = torch.cuda.CUDAGraph(keep_graph=True)\n"
                    "graph.register_post_instantiate_hook(\n"
                    f"    torch.cuda.graphs.save_graph_hook({path!r}, tensors=lambda: [a, out])\n"
                    ")",
                )
                + "\ngraph.instantiate()\n"
            )
            self._run(script)
            manifest, records = self._manifest(path)
            self.assertEqual(len(manifest["nodes"]), 1)
            self.assertEqual(len(manifest["tensors"]), 2)
            self.assertFalse(any(r.startswith("tensors/") for r in records))

    def test_save_fn_receives_the_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.ptcg")
            sidecar = os.path.join(tmp, "sidecar.pt")
            self._run(
                self._save_script(
                    "def writer(tensors):\n"
                    f"    torch.save([t.cpu() for t in tensors], {sidecar!r})\n"
                    f"graph.save({path!r}, tensors=[a, out], save_fn=writer)"
                )
            )
            manifest, records = self._manifest(path)
            self.assertEqual(len(manifest["tensors"]), 2)
            self.assertFalse(any(r.startswith("tensors/") for r in records))
            self.assertTrue(os.path.exists(sidecar))
            self.assertEqual(len(torch.load(sidecar)), 2)

    def test_saves_despite_unreferenced_non_expandable_memory(self):
        # The refusal is scoped to memory the graph reaches. A cudaMalloc segment
        # elsewhere in the process -- what a MemPool on a custom allocator produces --
        # does not make an unrelated graph unserializable.
        stray = (
            "torch.cuda.init()\n"
            "torch._C._accelerator_setAllocatorSettings('expandable_segments:False')\n"
            "stray = torch.empty(64 << 20, dtype=torch.uint8, device='cuda')\n"
            "torch._C._accelerator_setAllocatorSettings('expandable_segments:True')\n"
            "print('STRAY', stray.data_ptr())\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.ptcg")
            script = self._save_script(f"graph.save({path!r}, tensors=[a, out])")
            script = script.replace("a = torch.randn", stray + "a = torch.randn", 1)
            out = self._run(script)
            self.assertIn("SAVED", out)
            stray_addr = int(
                next(l for l in out.splitlines() if l.startswith("STRAY")).split()[1]
            )
            manifest, _ = self._manifest(path)
            self.assertGreater(len(manifest["segments"]), 0)
            for seg in manifest["segments"]:
                self.assertFalse(
                    seg["address"] <= stray_addr < seg["address"] + seg["total_size"],
                    "the stray cudaMalloc segment must not have been saved",
                )

    def test_refuses_without_expandable_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.ptcg")
            out = self._run(
                self._save_script(f"graph.save({path!r})"),
                expandable=False,
                expect_success=False,
            )
            self.assertIn("not an expandable segment", out)
            self.assertIn("expandable_segments:True", out)
            self.assertFalse(os.path.exists(path))

    def _raw_graph_env(self):
        from cuda.bindings import driver

        from torch.cuda._utils import _check_cuda_bindings_driver as chk

        # an allocation makes the primary context current for the driver API
        torch.zeros(1, device="cuda")
        return driver, chk

    def test_refuses_host_nodes(self):
        # A host node's payload is a host function pointer plus an opaque userData,
        # which cannot be rebound from outside the library that created it. This is
        # what rejects an NCCL collective on the network transport, where NCCL adds
        # one to push proxy args. Built by hand, so no NCCL is needed.
        import ctypes

        from torch.cuda._graph_serialization import _collect_nodes, UnserializableGraph

        driver, chk = self._raw_graph_env()
        graph = chk(driver.cuGraphCreate(0))
        callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _user_data: None)
        params = driver.CUDA_HOST_NODE_PARAMS()
        params.fn = ctypes.cast(callback, ctypes.c_void_p).value
        params.userData = 0
        chk(driver.cuGraphAddHostNode(graph, [], 0, params))
        with self.assertRaisesRegex(UnserializableGraph, "HOST"):
            _collect_nodes(driver, int(graph))
        chk(driver.cuGraphDestroy(graph))

    def test_event_nodes_keep_record_and_wait_paired(self):
        # A CUevent handle is process-local, so only the identity is recorded: nodes
        # sharing an event must share an index, and distinct events must not
        # collide. Load recreates one event per index, which reproduces ordering
        # inside the graph.
        from torch.cuda._graph_serialization import _collect_nodes

        driver, chk = self._raw_graph_env()
        graph = chk(driver.cuGraphCreate(0))
        first = chk(driver.cuEventCreate(0))
        second = chk(driver.cuEventCreate(0))
        record_first = chk(driver.cuGraphAddEventRecordNode(graph, [], 0, first))
        # two waits on the same event, plus a record of a different one
        chk(driver.cuGraphAddEventWaitNode(graph, [record_first], 1, first))
        chk(driver.cuGraphAddEventWaitNode(graph, [record_first], 1, first))
        chk(driver.cuGraphAddEventRecordNode(graph, [record_first], 1, second))

        nodes, _edges, _edge_data, num_events = _collect_nodes(driver, int(graph))
        self.assertEqual(num_events, 2)
        by_type: dict[str, list[int]] = {}
        for node in nodes:
            by_type.setdefault(node["type"], []).append(node["event"])
        self.assertEqual(len(by_type["event_record"]), 2)
        self.assertEqual(len(by_type["event_wait"]), 2)
        # both waits refer to the same event as the first record
        self.assertEqual(len(set(by_type["event_wait"])), 1)
        self.assertIn(by_type["event_wait"][0], by_type["event_record"])
        # and the two records are distinct events
        self.assertEqual(len(set(by_type["event_record"])), 2)

        chk(driver.cuGraphDestroy(graph))
        chk(driver.cuEventDestroy(first))
        chk(driver.cuEventDestroy(second))

    def test_refuses_when_capture_was_not_armed(self):
        # Arming after the kernels loaded is the mistake this has to catch: the
        # archive would otherwise be missing the code it needs.
        script = self._save_script("").replace(
            'assert cap.start(), "could not arm cubin capture"', ""
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.ptcg")
            script += (
                "\nfrom torch.cuda import _graph_kernel_capture as late\n"
                "assert late.start()\n"
                f"graph.save({path!r})\n"
            )
            out = self._run(script, expect_success=False)
            self.assertIn("no cubins were captured", out)


if __name__ == "__main__":
    run_tests()
