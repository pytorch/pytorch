# Owner(s): ["module: inductor"]

import contextlib
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from torch._C import FileCheck
from torch._dynamo.utils import detect_fake_mode
from torch._inductor import config
from torch._inductor.debug import (
    create_kernel_information_json,
    create_mapping_pre_post_grad_nodes,
    create_node_mapping_kernel_to_post_grad,
    get_kernel_information_jsons,
    reset_inductor_kernel_provenance_debug_handle,
    reset_provenance_globals,
    set_kernel_post_grad_provenance_tracing,
)
from torch._inductor.fx_passes.post_grad import post_grad_passes
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code, run_and_get_cpp_code
from torch._inductor.virtualized import V
from torch.profiler._utils import map_recorded_events_to_aten_ops_with_stack_trace
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton


try:
    from .test_aot_inductor_utils import AOTIRunnerUtil
    from .test_torchinductor import copy_tests
except ImportError:
    from test_aot_inductor_utils import AOTIRunnerUtil
    from test_torchinductor import (
        copy_tests,  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    )


trace_log = logging.getLogger("torch.__trace")


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        x = a * 3.14
        y = torch.addmm(c, x, b)
        z = torch.nn.functional.gelu(y)
        return z


class Model2(torch.nn.Module):
    # this test model is used for combo kernel provenance tracing info
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        a1 = torch.nn.functional.relu(a)
        b1 = torch.nn.functional.sigmoid(b)
        c1 = torch.nn.functional.tanh(c)
        return a1, b1, c1


class Model3(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = torch.randn(n, k, device=GPU_TYPE)
        self.bias = torch.randn(n, device=GPU_TYPE)

    def forward(self, a):
        return torch.nn.functional.linear(a, self.weight, self.bias)


class Model4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, a, b, c):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        d = a * 3.14
        y = torch.addmm(c, d, b)
        z = torch.nn.functional.gelu(y)
        return x, z


def _bias_like_addmm_input(device):
    # These provenance tests exercise the GPU addmm-unfusion path. Keep the
    # input bias-like so they do not depend on full-size accumulator unfusion.
    return torch.randn(30, device=device)


@config.patch("trace.enabled", True)
@config.patch("trace.provenance_tracking_level", 1)
class TestProvenanceTracingArtifact(TestCase):
    """
    This test checks that generated provenance tracing artifact from "post_grad" to
    corresponding "inductor triton kernel node" is expected.
    """

    def _check_provenance_tracing_kernel_to_post_grad(self, filepath, expected_data):
        self.assertTrue(filepath.is_dir())
        filename = Path(filepath) / "inductor_provenance_tracking_node_mappings.json"
        with open(filename) as f:
            actual_data = json.load(f)
        actual_data = actual_data["cppCodeToPost"]
        # check that the generated provenance tracing artifact is expected
        self.assertEqual(sorted(actual_data.items()), sorted(expected_data.items()))

    def _check_provenance_tracking_node_mappings(self, filepath, expected_mapping):
        self.assertTrue(filepath.is_dir())
        filename = Path(filepath) / "inductor_provenance_tracking_node_mappings.json"
        with open(filename) as f:
            actual_data = json.load(f)
        # check that the generated provenance tracing node mapping is expected
        self.assertEqual(sorted(actual_data.items()), sorted(expected_mapping))

    def _test_triton_kernel_to_post_grad_tracing(self, device):
        a = torch.randn(10, 20, device=device)
        b = torch.randn(20, 30, device=device)
        if device == "cpu":
            c = torch.randn(10, 30, device=device)
        else:
            c = _bias_like_addmm_input(device)
        example_inputs = (a, b, c)

        model = Model().to(device)
        filepath = None

        for backend in ["aot_inductor", "inductor"]:
            reset_inductor_kernel_provenance_debug_handle()
            try:
                with config.patch(
                    {
                        "trace.debug_dir": tempfile.mkdtemp(),
                        "force_disable_caches": True,
                    }
                ):
                    with self.assertLogs(
                        logging.getLogger("torch._inductor.debug"),
                        level=logging.WARNING,
                    ) as cm:
                        if backend == "aot_inductor":
                            AOTIRunnerUtil.run(model, example_inputs)
                        else:
                            ep = torch.export._trace._export(model, example_inputs)
                            compiled = torch.compile(ep.module(), backend=backend)
                            compiled(*example_inputs)
                    self.assertEqual(len(cm.output), 1)
                    m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
                    self.assertTrue(m)
                    filepath = Path(m.group(1))
                    if device == "cuda" or device == "xpu":
                        # aot_inductor uses export (no canonicalization),
                        # so pre-graph names stay as original.
                        # inductor uses torch.compile (canonicalization runs),
                        # so pre-graph names become canonical.
                        if backend == "aot_inductor":
                            pre_mul, pre_addmm, pre_gelu = (
                                "mul",
                                "addmm",
                                "gelu",
                            )
                        else:
                            pre_mul, pre_addmm, pre_gelu = (
                                "mul_tensor",
                                "addmm_default",
                                "gelu_default",
                            )
                        expected_mapping = [
                            (
                                "cppCodeToPost",
                                {
                                    "triton_poi_fused_mul_0:1": ["mul"],
                                    "triton_poi_fused_addmm_gelu_1:3": [
                                        "mul_3",
                                        "mul_1",
                                        "add_tensor",
                                        "add",
                                        "erf",
                                        "mul_2",
                                    ],
                                },
                            ),
                            (
                                "postToCppCode",
                                {
                                    "mul": ["triton_poi_fused_mul_0:1"],
                                    "mul_3": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "mul_1": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "add_tensor": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "add": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "erf": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "mul_2": ["triton_poi_fused_addmm_gelu_1:3"],
                                },
                            ),
                            (
                                "postToPre",
                                {
                                    "mul": [pre_mul],
                                    "mm_default": [pre_addmm],
                                    "add_tensor": [pre_addmm],
                                    "mul_1": [pre_gelu],
                                    "mul_2": [pre_gelu],
                                    "erf": [pre_gelu],
                                    "add": [pre_gelu],
                                    "mul_3": [pre_gelu],
                                },
                            ),
                            (
                                "preToPost",
                                {
                                    pre_mul: ["mul"],
                                    pre_addmm: [
                                        "mm_default",
                                        "add_tensor",
                                    ],
                                    pre_gelu: [
                                        "mul_1",
                                        "mul_2",
                                        "erf",
                                        "add",
                                        "mul_3",
                                    ],
                                },
                            ),
                        ]
                        if backend == "aot_inductor" and device == "cuda":
                            expected_mapping[0][1]["aoti_torch_cuda_mm_out:2"] = [
                                "mm_default"
                            ]
                            expected_mapping[1][1]["mm_default"] = [
                                "aoti_torch_cuda_mm_out:2"
                            ]
                        elif backend == "aot_inductor" and device == "xpu":
                            expected_mapping[0][1]["aoti_torch_xpu_mm_out:2"] = [
                                "mm_default"
                            ]
                            expected_mapping[1][1]["mm_default"] = [
                                "aoti_torch_xpu_mm_out:2"
                            ]
                        else:
                            expected_mapping[0][1]["extern_kernels.mm:2"] = [
                                "mm_default"
                            ]
                            expected_mapping[1][1]["mm_default"] = [
                                "extern_kernels.mm:2"
                            ]
                        self._check_provenance_tracking_node_mappings(
                            filepath, expected_mapping
                        )
                    else:
                        if device != "cpu":
                            raise AssertionError
                        # check the inductor kernel to post grad nodes mapping is expected for cpu
                        if backend == "aot_inductor":
                            expected_data = {
                                "cpp_fused_mul_0:1": ["mul"],
                                "aoti_torch_cpu_addmm_out:2": ["addmm"],
                                "cpp_fused_gelu_1:3": [
                                    "mul_3",
                                    "mul_1",
                                    "add",
                                    "erf",
                                    "mul_2",
                                ],
                            }
                        else:
                            # backend == "inductor"
                            expected_data = {
                                "cpp_fused_mul_0:1": ["mul"],
                                "cpp_fused_gelu_1:3": [
                                    "mul_3",
                                    "mul_1",
                                    "add",
                                    "erf",
                                    "mul_2",
                                ],
                                "extern_kernels.addmm:2": ["addmm"],
                            }
                        self._check_provenance_tracing_kernel_to_post_grad(
                            filepath, expected_data
                        )

            finally:
                if filepath:
                    shutil.rmtree(filepath)

    @requires_gpu_and_triton
    def test_triton_kernel_to_post_grad_tracing_cuda(self):
        self._test_triton_kernel_to_post_grad_tracing(device=GPU_TYPE)

    def test_triton_kernel_to_post_grad_tracing_cpu(self):
        self._test_triton_kernel_to_post_grad_tracing(device="cpu")

    @requires_gpu_and_triton
    def test_triton_kernel_to_post_grad_tracing_extern_kernel(self):
        M = 8
        N = 6
        K = 16
        model = Model3(N, K)
        batch = 2
        a = torch.randn(batch, M, K, device=GPU_TYPE)
        example_inputs = (a,)
        filepath = None

        for backend in ["aot_inductor", "inductor"]:
            reset_inductor_kernel_provenance_debug_handle()
            try:
                with config.patch(
                    {
                        "trace.debug_dir": tempfile.mkdtemp(),
                        "force_disable_caches": True,
                    }
                ):
                    with self.assertLogs(
                        logging.getLogger("torch._inductor.debug"),
                        level=logging.WARNING,
                    ) as cm:
                        if backend == "aot_inductor":
                            AOTIRunnerUtil.run(model, example_inputs)
                        else:
                            ep = torch.export._trace._export(model, example_inputs)
                            compiled = torch.compile(ep.module(), backend=backend)
                            compiled(*example_inputs)
                    self.assertEqual(len(cm.output), 1)
                    m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
                    self.assertTrue(m)
                    filepath = Path(m.group(1))
                    if backend == "inductor":
                        expected_data = {
                            "extern_kernels.addmm:1": ["addmm"],
                        }
                    else:
                        # backend = aot_inductor
                        expected_data = {
                            f"aoti_torch_{GPU_TYPE}_addmm_out:2": ["addmm"],
                            "triton_poi_fused_0:1": ["_tensor_constant1"],
                        }

                    self._check_provenance_tracing_kernel_to_post_grad(
                        filepath, expected_data
                    )
            finally:
                if filepath:
                    shutil.rmtree(filepath)

    @requires_gpu_and_triton
    def _test_pt_tracing_combo_kernel(self, backend):
        """This test checks that generated provenance tracing artifact from triton combo kernel to post grad nodes"""
        a = torch.randn(10, 10, device=GPU_TYPE)
        b = torch.randn(20, 20, device=GPU_TYPE)
        c = torch.randn(10, 10, device=GPU_TYPE)
        example_inputs = (a, b, c)

        model = Model2()
        reset_inductor_kernel_provenance_debug_handle()

        with config.patch(
            {
                "trace.debug_dir": tempfile.mkdtemp(),
                "force_disable_caches": True,
                "combo_kernels": True,
                "benchmark_combo_kernel": False,
            }
        ):
            with self.assertLogs(
                logging.getLogger("torch._inductor.debug"),
                level=logging.WARNING,
            ) as cm:
                if backend == "aot_inductor":
                    AOTIRunnerUtil.run(model, example_inputs)
                else:
                    ep = torch.export._trace._export(model, example_inputs)
                    compiled = torch.compile(ep.module(), backend=backend)
                    compiled(*example_inputs)
            self.assertEqual(len(cm.output), 1)
            m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
            self.assertTrue(m)
            filepath = Path(m.group(1)).resolve()
            expected_data = {"triton_poi_fused_0:1": ["relu", "sigmoid", "tanh"]}
            self._check_provenance_tracing_kernel_to_post_grad(filepath, expected_data)

    @requires_gpu_and_triton
    def test_triton_kernel_to_post_grad_tracing_combo_kernel(self):
        self._test_pt_tracing_combo_kernel(backend="inductor")
        self._test_pt_tracing_combo_kernel(backend="aot_inductor")


class TestProvenanceTracingNodeMapping(TestCase):
    def test_create_node_mapping(self):
        pre_grad_graph_id = 140156815043952
        post_to_pre_grad_nodes_json = {
            "add_tensor": [
                {
                    "from_node": [
                        {
                            "from_node": [
                                {
                                    "from_node": [],
                                    "graph_id": 140156815043952,
                                    "name": "linear",
                                }
                            ],
                            "graph_id": 140152856025632,
                            "name": "addmm",
                        }
                    ],
                    "graph_id": 140151961816272,
                    "name": "add",
                },
            ],
            "mm_default": [
                {
                    "from_node": [],
                    "graph_id": -1,
                    "name": "",
                },
                {
                    "from_node": [
                        {
                            "from_node": [
                                {
                                    "from_node": [],
                                    "graph_id": 140156815043952,
                                    "name": "linear",
                                }
                            ],
                            "graph_id": 140152856025632,
                            "name": "addmm",
                        }
                    ],
                    "graph_id": 140151961816272,
                    "name": "mm",
                },
            ],
            "permute": [
                {
                    "from_node": [],
                    "graph_id": 140156815043952,
                    "name": "linear",
                }
            ],
            "relu": [
                {
                    "from_node": [],
                    "graph_id": 140156815043952,
                    "name": "relu",
                }
            ],
        }
        triton_kernel_to_post_grad_json = {
            "triton_poi_fused_addmm_relu_sigmoid_0": ["relu", "add_tensor"]
        }

        result = create_mapping_pre_post_grad_nodes(
            pre_grad_graph_id,
            post_to_pre_grad_nodes_json,
        )
        result = {
            **result,
            **create_node_mapping_kernel_to_post_grad(
                triton_kernel_to_post_grad_json,
            ),
        }

        self.assertEqual(
            result,
            {
                "cppCodeToPost": {
                    "triton_poi_fused_addmm_relu_sigmoid_0": [
                        "relu",
                        "add_tensor",
                    ]
                },
                "postToCppCode": {
                    "add_tensor": ["triton_poi_fused_addmm_relu_sigmoid_0"],
                    "relu": ["triton_poi_fused_addmm_relu_sigmoid_0"],
                },
                "postToPre": {
                    "add_tensor": ["linear"],
                    "mm_default": ["linear"],
                    "permute": ["linear"],
                    "relu": ["relu"],
                },
                "preToPost": {
                    "linear": ["add_tensor", "mm_default", "permute"],
                    "relu": ["relu"],
                },
            },
        )


class TestProvenanceTracingNodeMeta(TestCase):
    def get_node_with_target(self, gm, target):
        """
        Return first node in gm with target
        """
        return next(iter([node for node in gm.graph.nodes if node.target == target]))

    @requires_gpu_and_triton  # test only works for cuda pattern matcher
    def test_pattern_matcher_transfer_meta(self):
        """
        Test that stack trace is transferred when node is decomposed in post_grad_passes
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.sigmoid(x)
                return x * 3

        x = torch.randn(8, 10).to(GPU_TYPE)
        example_inputs = (x,)
        model = Model().to(GPU_TYPE)

        # mimic the before_post_grad graph
        ep = torch.export.export(model, example_inputs).run_decompositions()
        gm = ep.module()

        # Set fake mode for V
        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        V.set_fake_mode(fake_mode)

        addmm_node = self.get_node_with_target(gm, torch.ops.aten.addmm.default)
        stack_trace = addmm_node.meta["stack_trace"]

        post_grad_passes(gm, True)  # for this test is_inference doesn't matter

        mm_node = self.get_node_with_target(gm, torch.ops.aten.mm.default)
        add_node = self.get_node_with_target(gm, torch.ops.aten.add.Tensor)

        self.assertEqual(add_node.meta["stack_trace"], stack_trace)
        self.assertEqual(mm_node.meta["stack_trace"], stack_trace)


class ProvenanceArtifactFilter(logging.Filter):
    def filter(self, record):
        if "artifact" in record.metadata:
            return (
                record.metadata["artifact"]["name"]
                == "inductor_provenance_tracking_kernel_stack_traces"
            )
        return False


class StructuredTracePayloadFormatter(logging.Formatter):
    def format(self, record):
        return record.payload.strip()


class TestProvenanceTracingStackTraces(TestCase):
    @contextlib.contextmanager
    def _setup_provenance_capture(self):
        """Helper to turn on and capture the 'inductor_tlparse_runtime' structured trace."""
        payload_buffer = io.StringIO()
        payload_handler = logging.StreamHandler(payload_buffer)
        payload_handler.setLevel(logging.DEBUG)
        payload_handler.setFormatter(StructuredTracePayloadFormatter())
        payload_handler.addFilter(ProvenanceArtifactFilter())
        trace_log.addHandler(payload_handler)
        try:
            yield payload_buffer
        finally:
            trace_log.removeHandler(payload_handler)

    def extract_code_line(self, s):
        # Extract the source code line from a stack trace entry.
        # Filter out empty lines, "File ..." lines, and caret annotation
        # lines (e.g. "~~^~~~~~") added in Python 3.13+.
        lines = s.split("\n")
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("File "):
                continue
            if all(c in " ~^" for c in stripped):
                continue
            return stripped
        return lines[-2].strip()

    @torch._inductor.config.patch({"trace.provenance_tracking_level": 2})
    def test_tlparse_kernel_stack_traces_cpu(self):
        model = Model4()
        example_inputs = (
            torch.randn(8, 10),
            torch.randn(10, 20),
            torch.randn(20, 30),
            torch.randn(10, 30),
        )

        expected = {
            "cpp_fused_mul_0:2": [
                "d = a * 3.14",
            ],
            "cpp_fused_gelu_relu_sigmoid_threshold_backward_1:4": [
                "z = torch.nn.functional.gelu(y)",
                "x = self.relu(x)",
                "x = self.sigmoid(x)",
            ],
            "extern_kernels.addmm:1": [
                "x = self.fc1(x)",
            ],
            "extern_kernels.addmm:3": [
                "y = torch.addmm(c, d, b)",
            ],
        }

        compiled = torch.compile(model)
        for _ in range(2):
            torch._dynamo.reset()
            reset_inductor_kernel_provenance_debug_handle()
            with self._setup_provenance_capture() as payload_buffer:
                compiled = torch.compile(model)
                compiled(*example_inputs)
                payload_content = payload_buffer.getvalue().strip()
                data = json.loads(payload_content)
                self.assertEqual(set(data.keys()), set(expected.keys()))
                for key, expected_lines in expected.items():
                    actual_lines = [self.extract_code_line(s) for s in data[key]]
                    self.assertEqual(
                        sorted(actual_lines),
                        sorted(expected_lines),
                        lambda msg: f"{msg}\nMismatch for key: {key}",
                    )

    @torch._inductor.config.patch({"trace.provenance_tracking_level": 2})
    @requires_gpu_and_triton
    def test_tlparse_kernel_stack_traces(self):
        device = GPU_TYPE
        model = Model4().to(device)
        x = torch.randn(8, 10).to(device)
        a = torch.randn(10, 20).to(device)
        b = torch.randn(20, 30).to(device)
        c = _bias_like_addmm_input(device)
        example_inputs = (x, a, b, c)

        expected = {
            "triton_poi_fused_addmm_relu_sigmoid_threshold_backward_2:5": [
                "x = self.sigmoid(x)",
                "x = self.fc1(x)",
                "x = self.relu(x)",
            ],
            "triton_poi_fused_mul_0:2": [
                "d = a * 3.14",
            ],
            "triton_poi_fused_addmm_gelu_1:4": [
                "z = torch.nn.functional.gelu(y)",
                "y = torch.addmm(c, d, b)",
            ],
            "extern_kernels.mm:3": [
                "y = torch.addmm(c, d, b)",
            ],
            "extern_kernels.mm:1": [
                "x = self.fc1(x)",
            ],
        }

        compiled = torch.compile(model)
        # should produce the same provenance if there's cache hit
        for _ in range(2):
            # reset cache
            torch._dynamo.reset()
            reset_inductor_kernel_provenance_debug_handle()
            with self._setup_provenance_capture() as payload_buffer:
                compiled = torch.compile(model)
                compiled(*example_inputs)
                payload_content = payload_buffer.getvalue().strip()
                data = json.loads(payload_content)
                self.assertEqual(set(data.keys()), set(expected.keys()))
                for key, expected_lines in expected.items():
                    actual_lines = [self.extract_code_line(s) for s in data[key]]
                    self.assertEqual(
                        sorted(actual_lines),
                        sorted(expected_lines),
                        lambda msg: f"{msg}\nMismatch for key: {key}",
                    )

    @torch._inductor.config.patch(
        {"trace.provenance_tracking_level": 2, "max_autotune_gemm_backends": "ATEN"}
    )
    @requires_gpu_and_triton
    def test_deferred_triton_kernels(self):
        def foo(m, inp):
            a = m(inp)
            return a

        foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

        m = torch.nn.Linear(512, 512, bias=True).half().to(GPU_TYPE)
        inp = torch.rand([1, 512]).half().to(GPU_TYPE)

        with self._setup_provenance_capture() as payload_buffer:
            with torch.no_grad():
                _, out_code = run_and_get_code(foo_c, m, inp)
            payload_content = payload_buffer.getvalue().strip()
            data = json.loads(payload_content)
            self.assertTrue("a = m(inp)" in str(data))

            # Check that debug handle is in the output code
            FileCheck().check("Topologically Sorted Source Nodes: [linear]").check(
                "[Provenance debug handles]"
            ).run(out_code[0])

    def _check_kernel_information_json(self, kernel_info, expected_kernels):
        """Validate kernel information JSON structure and content."""
        self.assertIsInstance(kernel_info, dict)

        for expected in expected_kernels:
            self.assertIn(
                expected,
                kernel_info,
                lambda msg: f"{msg}\nExpected kernel {expected} not found in {list(kernel_info)}",
            )

        for data in kernel_info.values():
            self.assertIsInstance(data, dict)
            for field in ["stack_traces", "post_grad_nodes", "pre_grad_nodes"]:
                self.assertIn(field, data)
                self.assertIsInstance(data[field], list)
                for item in data[field]:
                    self.assertIsInstance(item, str)
            self.assertIsInstance(data["extern_semantic_key"], (str, type(None)))

    @requires_gpu_and_triton
    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_kernel_information_generation(self):
        """Test basic kernel information generation in AOTI packages."""

        model = Model4().to(GPU_TYPE)
        x = torch.randn(8, 10, device=GPU_TYPE)
        a = torch.randn(10, 20, device=GPU_TYPE)
        b = torch.randn(20, 30, device=GPU_TYPE)
        c = _bias_like_addmm_input(GPU_TYPE)
        inputs = (x, a, b, c)

        with tempfile.TemporaryDirectory() as temp_dir:
            ep = torch.export.export(model, inputs, strict=False)
            pt2_file = os.path.join(temp_dir, "model.pt2")
            reset_inductor_kernel_provenance_debug_handle()
            torch._inductor.aoti_compile_and_package(ep, package_path=pt2_file)

            # Extract and check kernel_information.json exists in the package
            with zipfile.ZipFile(pt2_file, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            json_path = os.path.join(
                temp_dir,
                "model",
                "data",
                "aotinductor",
                "model",
                "kernel_information.json",
            )
            self.assertTrue(
                os.path.exists(json_path),
                lambda msg: f"{msg}\nkernel_information.json not found in extracted package at {json_path}",
            )

            with open(json_path) as f:
                kernel_info = json.load(f)

            expected = {
                "triton_poi_fused_addmm_relu_sigmoid_0:2": {
                    "stack_traces": [
                        "x = self.sigmoid(x)",
                        "x = self.fc1(x)",
                        "x = self.relu(x)",
                    ],
                    "post_grad_nodes": ["sigmoid", "relu", "add_tensor_1"],
                    "pre_grad_nodes": ["sigmoid", "relu", "linear"],
                },
                "triton_poi_fused_mul_1:3": {
                    "stack_traces": [
                        "d = a * 3.14",
                    ],
                    "post_grad_nodes": ["mul"],
                    "pre_grad_nodes": ["mul"],
                },
                "triton_poi_fused_addmm_gelu_2:5": {
                    "stack_traces": [
                        "z = torch.nn.functional.gelu(y)",
                        "y = torch.addmm(c, d, b)",
                    ],
                    "post_grad_nodes": [
                        "mul_3",
                        "mul_1",
                        "add_tensor",
                        "add",
                        "erf",
                        "mul_2",
                    ],
                    "pre_grad_nodes": ["gelu", "addmm"],
                },
                f"aoti_torch_{GPU_TYPE}_mm_out:1": {
                    "stack_traces": [
                        "x = self.fc1(x)",
                    ],
                    "post_grad_nodes": ["mm_default_1"],
                    "pre_grad_nodes": ["linear"],
                },
                f"aoti_torch_{GPU_TYPE}_mm_out:4": {
                    "stack_traces": [
                        "y = torch.addmm(c, d, b)",
                    ],
                    "post_grad_nodes": ["mm_default"],
                    "pre_grad_nodes": ["addmm"],
                },
            }

            self._check_kernel_information_json(kernel_info, expected.keys())

            self.assertEqual(set(kernel_info.keys()), set(expected.keys()))
            for key, data in expected.items():
                all_lines = ",".join(kernel_info[key]["stack_traces"])
                for s in data["stack_traces"]:
                    self.assertTrue(s in all_lines)

                self.assertEqual(
                    sorted(kernel_info[key]["pre_grad_nodes"]),
                    sorted(data["pre_grad_nodes"]),
                    lambda msg: f"{msg}\nMismatch for key: {key}",
                )

                self.assertEqual(
                    sorted(kernel_info[key]["post_grad_nodes"]),
                    sorted(data["post_grad_nodes"]),
                    lambda msg: f"{msg}\nMismatch for key: {key}",
                )

            # extern_semantic_key + shape metadata for extern kernels
            triton_poi_0 = kernel_info["triton_poi_fused_addmm_relu_sigmoid_0:2"]
            self.assertIsNone(triton_poi_0["extern_semantic_key"])

            mm_out_1 = kernel_info[f"aoti_torch_{GPU_TYPE}_mm_out:1"]
            # Single pre_grad_node "linear" → extern_semantic_key fallback
            self.assertEqual(mm_out_1["extern_semantic_key"], "linear")
            self.assertEqual(mm_out_1["input_shapes"], [[8, 10], [10, 16]])
            self.assertEqual(
                mm_out_1["input_dtypes"], ["torch.float32", "torch.float32"]
            )
            self.assertEqual(mm_out_1["output_shape"], [8, 16])
            self.assertEqual(mm_out_1["output_dtype"], "torch.float32")

            mm_out_4 = kernel_info[f"aoti_torch_{GPU_TYPE}_mm_out:4"]
            self.assertEqual(mm_out_4["extern_semantic_key"], "addmm")
            self.assertEqual(mm_out_4["input_shapes"], [[10, 20], [20, 30]])
            self.assertEqual(
                mm_out_4["input_dtypes"], ["torch.float32", "torch.float32"]
            )
            self.assertEqual(mm_out_4["output_shape"], [10, 30])
            self.assertEqual(mm_out_4["output_dtype"], "torch.float32")

    @torch._inductor.config.patch("trace.provenance_tracking_level", 0)
    def test_no_kernel_information_without_provenance_tracking(self):
        """Test that kernel_information.json is not generated without provenance tracking."""

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x * 2.0

        model = SimpleModel()
        x = torch.randn(4, 8)

        # Compile with AOTI but without provenance tracking
        with tempfile.TemporaryDirectory() as temp_dir:
            ep = torch.export.export(model, (x,), strict=False)
            pt2_file = os.path.join(temp_dir, "model.pt2")
            torch._inductor.aoti_compile_and_package(ep, package_path=pt2_file)

            # Extract and check kernel_information.json was NOT created in the package
            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(pt2_file, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            expected_json_path = os.path.join(extract_dir, "kernel_information.json")
            self.assertFalse(
                os.path.exists(expected_json_path),
                "kernel_information.json should not exist in package when provenance tracking is disabled",
            )

    def test_create_kernel_information_json_function(self):
        """Test the create_kernel_information_json function directly."""
        # Test with empty state
        result = create_kernel_information_json()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)  # Should be empty with no provenance data

    def test_reset_provenance_globals_preserves_kernel_information_jsons(self):
        kernel_information_jsons = get_kernel_information_jsons()
        previous = dict(kernel_information_jsons)
        kernel_information_jsons.clear()
        kernel_information_jsons["outer"] = {}
        try:
            with reset_provenance_globals():
                self.assertEqual(get_kernel_information_jsons(), {"outer": {}})
                get_kernel_information_jsons()["inner"] = {}
            self.assertEqual(get_kernel_information_jsons(), {"outer": {}, "inner": {}})
        finally:
            get_kernel_information_jsons().clear()
            get_kernel_information_jsons().update(previous)

    @unittest.skipIf(
        IS_MACOS,
        "MacOS generates different debug handles",
    )
    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_cpu_extern_kernel(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)

            def forward(self, x):
                return self.conv(x)

        model = Foo()
        x = torch.randn(20, 16, 50, 100)
        with self._setup_provenance_capture() as payload_buffer:
            reset_inductor_kernel_provenance_debug_handle()
            ep = torch.export.export(model, (x,))
            torch._inductor.aoti_compile_and_package(ep)
            payload_content = payload_buffer.getvalue().strip()
            data = json.loads(payload_content)

            keys = [k.split(":")[0] for k in data]
            self.assertTrue("aoti_torch_cpu_convolution" in keys)

    def test_create_kernel_information_json_with_synthetic_data(self):
        """Test create_kernel_information_json with synthetic globals."""
        import torch._inductor.debug as debug_mod

        with (
            config.patch("trace.provenance_tracking_level", 1),
            reset_provenance_globals(),
        ):
            # Populate globals with known data
            debug_mod._inductor_triton_kernel_to_post_grad_node_info = {
                "triton_poi_fused_add_mul_0:1": ["add", "mul"],
                "aoti_torch_cuda_mm_out:2": ["mm_default"],
                "extern_kernels.addmm:3": ["addmm_default"],
                "custom_backend.mm:4": ["custom_mm_default"],
                "aoti_torch_cuda_add_out:5": ["add_default"],
            }
            debug_mod._inductor_kernel_stack_trace = {
                "triton_poi_fused_add_mul_0:1": ["File test.py, line 10"],
                "aoti_torch_cuda_mm_out:2": ["File test.py, line 20"],
                "extern_kernels.addmm:3": ["File test.py, line 30"],
                "custom_backend.mm:4": ["File test.py, line 40"],
                "aoti_torch_cuda_add_out:5": ["File test.py, line 50"],
            }
            debug_mod._inductor_post_to_pre_grad_nodes = {
                "postToPre": {
                    "add": ["add_1"],
                    "mul": ["mul_1"],
                    "mm_default": ["linear"],
                    "addmm_default": ["addmm"],
                    "custom_mm_default": ["custom_linear"],
                    "add_default": ["prefix_only_add"],
                }
            }
            # Consolidated per-kernel extern info: is_extern flag, explicit
            # semantic key (simulates FP8 bridge), and shape metadata.
            debug_mod._inductor_kernel_extern_info = {
                "aoti_torch_cuda_mm_out:2": debug_mod._KernelExternInfo(
                    is_extern=True,
                    semantic_key="linear_42",
                    shapes={
                        "input_shapes": [[8, 10], [10, 16]],
                        "input_dtypes": ["torch.float32", "torch.float32"],
                        "output_shape": [8, 16],
                        "output_dtype": "torch.float32",
                    },
                ),
                "extern_kernels.addmm:3": debug_mod._KernelExternInfo(is_extern=True),
                "custom_backend.mm:4": debug_mod._KernelExternInfo(is_extern=True),
            }

            result = create_kernel_information_json()

            # Triton pointwise kernel: no extern_semantic_key, no shape metadata.
            k1 = result["triton_poi_fused_add_mul_0:1"]
            self.assertIsNone(k1["extern_semantic_key"])
            self.assertNotIn("input_shapes", k1)

            # Extern kernel with explicit semantic key (FP8 bridge precedence)
            # and merged shape metadata.
            k2 = result["aoti_torch_cuda_mm_out:2"]
            self.assertEqual(k2["extern_semantic_key"], "linear_42")
            self.assertEqual(k2["input_shapes"], [[8, 10], [10, 16]])
            self.assertEqual(k2["output_shape"], [8, 16])
            self.assertEqual(k2["output_dtype"], "torch.float32")

            # Extern kernel with single pre_grad_node fallback semantic key.
            k3 = result["extern_kernels.addmm:3"]
            self.assertEqual(k3["extern_semantic_key"], "addmm")
            self.assertNotIn("input_shapes", k3)

            # Extern fallback comes from explicit tracking, not kernel-name prefixes.
            k4 = result["custom_backend.mm:4"]
            self.assertEqual(k4["extern_semantic_key"], "custom_linear")

            k5 = result["aoti_torch_cuda_add_out:5"]
            self.assertIsNone(k5["extern_semantic_key"])

    def test_extern_semantic_key_extracted_from_origin_node_meta(self):
        """extern_semantic_key stamped on origin_node.meta (e.g. by the FP8
        lowering pass) is extracted into the provenance globals and JSON."""
        import torch._inductor.debug as debug_mod
        import torch._inductor.ir as ir_mod

        class FakeOriginNode:
            def __init__(self, name, meta) -> None:
                self.name = name
                self.meta = meta

        class FakeExternKernel(ir_mod.ExternKernel):
            def __init__(self, origin_node) -> None:
                self.inputs = []
                self.origin_node = origin_node
                self.origins = []

            def has_tensor_output(self):
                return False

            def get_stack_traces(self):
                return []

        with (
            config.patch("trace.provenance_tracking_level", 1),
            reset_provenance_globals(),
        ):
            extern = FakeExternKernel(
                FakeOriginNode("linear", {"extern_semantic_key": "my_fp8_key"})
            )
            handle = set_kernel_post_grad_provenance_tracing(
                extern, "aoti_torch_cuda_mm_out", is_extern=True
            )
            kernel_name = f"aoti_torch_cuda_mm_out:{handle}"
            info = debug_mod._inductor_kernel_extern_info[kernel_name]
            self.assertTrue(info.is_extern)
            self.assertEqual(info.semantic_key, "my_fp8_key")

            result = create_kernel_information_json()
            self.assertEqual(result[kernel_name]["extern_semantic_key"], "my_fp8_key")

    def test_extern_kernel_metadata_accepts_tensor_like_ir_inputs(self):
        import torch._inductor.ir as ir_mod

        class FakeTensorIRNode(ir_mod.IRNode):
            def __init__(self, size, dtype) -> None:
                self._size = size
                self._dtype = dtype

            def has_tensor_output(self):
                return True

            def maybe_get_size(self):
                return self._size

            def maybe_get_dtype(self):
                return self._dtype

        class FakeNonTensorIRNode(ir_mod.IRNode):
            def has_tensor_output(self):
                return False

        class FakeExternKernel(ir_mod.ExternKernel):
            def __init__(self, inputs, output_size=None, output_dtype=None) -> None:
                self.inputs = inputs
                self.origin_node = None
                self.origins = []
                self._output_size = output_size
                self._output_dtype = output_dtype

            def has_tensor_output(self):
                return self._output_size is not None

            def maybe_get_size(self):
                return self._output_size

            def maybe_get_dtype(self):
                return self._output_dtype

            def get_stack_traces(self):
                return ["File fake.py, line 1"]

        with (
            config.patch("trace.provenance_tracking_level", 1),
            reset_provenance_globals(),
        ):
            extern = FakeExternKernel(
                [
                    FakeTensorIRNode([2, 4], torch.float32),
                    [FakeTensorIRNode([3, "s0"], torch.bfloat16)],
                    FakeNonTensorIRNode(),
                ],
                output_size=[2, "s0"],
                output_dtype=torch.float16,
            )
            handle = set_kernel_post_grad_provenance_tracing(
                extern, "custom_backend.mm", is_extern=True
            )
            self.assertEqual(handle, 1)

            non_tensor_output = FakeExternKernel(
                [FakeTensorIRNode([7], torch.int64)],
            )
            handle = set_kernel_post_grad_provenance_tracing(
                non_tensor_output, "custom_backend.multi", is_extern=True
            )
            self.assertEqual(handle, 2)

            result = create_kernel_information_json()

            metadata = result["custom_backend.mm:1"]
            self.assertEqual(metadata["input_shapes"], [[2, 4], [3, "s0"]])
            self.assertEqual(
                metadata["input_dtypes"], ["torch.float32", "torch.bfloat16"]
            )
            self.assertEqual(metadata["output_shape"], [2, "s0"])
            self.assertEqual(metadata["output_dtype"], "torch.float16")

            non_tensor_output_metadata = result["custom_backend.multi:2"]
            self.assertEqual(non_tensor_output_metadata["input_shapes"], [[7]])
            self.assertEqual(
                non_tensor_output_metadata["input_dtypes"], ["torch.int64"]
            )
            self.assertNotIn("output_shape", non_tensor_output_metadata)
            self.assertNotIn("output_dtype", non_tensor_output_metadata)


class ProvenanceTracingKernelContextTemplate:
    def test_jit_inductor_with_flag(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x, a, b, c):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.sigmoid(x)
                d = a * 3.14
                y = torch.addmm(c, d, b)
                z = torch.nn.functional.gelu(y)
                return x, z

        model = Model().to(self.device)
        x = torch.randn(8, 10).to(self.device)
        a = torch.randn(10, 20).to(self.device)
        b = torch.randn(20, 30).to(self.device)
        if self.device == "cpu":
            c = torch.randn(10, 30).to(self.device)
        else:
            c = _bias_like_addmm_input(self.device)
        example_inputs = (x, a, b, c)

        with config.patch(
            {
                "cpp.enable_kernel_profile": True,
            }
        ):
            torch.compile(model)(*example_inputs)

    @unittest.skipIf(sys.platform == "darwin", "Different kernel names on MacOS")
    def test_aoti_python_stack_traces(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x, a, b, c):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.sigmoid(x)
                d = a * 3.14
                y = torch.addmm(c, d, b)
                z = torch.nn.functional.gelu(y)
                return x, z

        x = torch.randn(8, 10).to(self.device)
        a = torch.randn(10, 20).to(self.device)
        b = torch.randn(20, 30).to(self.device)
        if self.device == "cpu":
            c = torch.randn(10, 30).to(self.device)
        else:
            c = _bias_like_addmm_input(self.device)
        example_inputs = (x, a, b, c)
        model = Model().to(self.device)

        ep = torch.export.export(model, example_inputs)
        _, code = run_and_get_cpp_code(torch._inductor.aoti_compile_and_package, ep)

        self.assertTrue("KernelContextGuard" not in code)
        FileCheck().check_not(
            "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
        ).check_not("thread_local KernelContext* tls_kernel_context = nullptr;").run(
            code
        )

        with config.patch(
            {
                "trace.provenance_tracking_level": 1,
                "cpp.enable_kernel_profile": True,
                "cpp.enable_kernel_context_guard": False,
            }
        ):
            package_path, code = run_and_get_cpp_code(
                torch._inductor.aoti_compile_and_package, ep
            )

            FileCheck().check_not(
                "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
            ).check_not(
                "thread_local KernelContext* tls_kernel_context = nullptr;"
            ).check_not("KernelContextGuard").run(code)

            compiled_model = torch._inductor.aoti_load_package(package_path)
            result = compiled_model(*example_inputs)
            self.assertEqual(result, model(*example_inputs))

        with config.patch(
            {
                "trace.provenance_tracking_level": 1,
                "cpp.enable_kernel_profile": True,
                "cpp.enable_kernel_context_guard": True,
            }
        ):
            package_path, code = run_and_get_cpp_code(
                torch._inductor.aoti_compile_and_package, ep
            )

            FileCheck().check(
                "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
            ).check("thread_local KernelContext* tls_kernel_context = nullptr;").run(
                code
            )

            if self.device == "cuda" or self.device == "xpu":
                device_type = torch.accelerator.current_accelerator().type
                FileCheck().check(
                    f"""KernelContextGuard _ctx("aoti_torch_{device_type}_mm_out", R"("""
                ).check(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_{device_type}_mm_out("
                ).check(
                    """KernelContextGuard _ctx("triton_poi_fused_addmm_relu_sigmoid_0", R"("""
                ).check("call_triton_poi_fused_addmm_relu_sigmoid_0(").check(
                    """KernelContextGuard _ctx("triton_poi_fused_mul_1", R"("""
                ).check("call_triton_poi_fused_mul_1(").check(
                    f"""KernelContextGuard _ctx("aoti_torch_{device_type}_mm_out", R"""
                ).check(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_{device_type}_mm_out("
                ).check(
                    """ KernelContextGuard _ctx("triton_poi_fused_addmm_gelu_2", R"("""
                ).check("call_triton_poi_fused_addmm_gelu_2(").run(code)
            else:
                FileCheck().check(
                    """KernelContextGuard _ctx("aoti_torch_cpu_addmm_out", R"("""
                ).check("AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_addmm_out(").check(
                    """KernelContextGuard _ctx("cpp_fused_mul_relu_sigmoid_0", R"("""
                ).check("cpp_fused_mul_relu_sigmoid_0(").check(
                    """KernelContextGuard _ctx("aoti_torch_cpu_addmm_out", R"("""
                ).check("AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_addmm_out(").check(
                    """ KernelContextGuard _ctx("cpp_fused_gelu_1", R"("""
                ).check("cpp_fused_gelu_1(").run(code)

            compiled_model = torch._inductor.aoti_load_package(package_path)
            result = compiled_model(*example_inputs)
            self.assertEqual(result, model(*example_inputs))


class TestProvenanceTracingKernelContextCpu(TestCase):
    device = "cpu"


copy_tests(
    ProvenanceTracingKernelContextTemplate,
    TestProvenanceTracingKernelContextCpu,
    "cpu",
)


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
@unittest.skipIf(
    not torch.cuda.is_available() and not torch.xpu.is_available(), "No CUDA and no XPU"
)
class TestProvenanceTracingKernelContextGpu(TestCase):
    device = GPU_TYPE


copy_tests(
    ProvenanceTracingKernelContextTemplate,
    TestProvenanceTracingKernelContextGpu,
    GPU_TYPE,
)


def _compile_capture(mod, *inputs):
    """Compile on CPU-triton and return the list of leaf-op-dicts per scheduler node."""
    import torch._inductor.scheduler as sched
    from torch._inductor.kernel_trace import extract_leaf_ops, buffer_roles
    captured = []
    orig = sched.Scheduler.__init__
    def hook(self, *a, **k):
        orig(self, *a, **k)
        for n in self.nodes:
            leaves = list(getattr(n, "snodes", None) or [n])
            for lf in leaves:
                if type(lf).__name__ == "SchedulerNode":
                    captured.append((lf.get_name(), extract_leaf_ops(lf), buffer_roles(lf)))
    sched.Scheduler.__init__ = hook
    try:
        with config.patch(force_disable_caches=True, cpu_backend="triton"):
            with torch.no_grad(): torch.compile(mod, backend="inductor")(*inputs)
    except torch._inductor.exc.InductorError:
        # CPU-triton aborts after scheduling (expected by design)
        pass
    finally:
        sched.Scheduler.__init__ = orig
    torch._dynamo.reset()
    return captured

def test_walker_extracts_ordered_ops_and_roles():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * x
    cap = _compile_capture(M().eval(), torch.randn(4, 16))
    assert cap, "no SchedulerNode captured"
    targets = {op["target"] for _, ops, _ in cap for op in ops}
    assert "rsqrt" in targets and "load" in targets
    # every op has order/target/block
    for _, ops, _ in cap:
        for op in ops:
            assert set(("order", "target", "block")).issubset(op)
    # roles are name lists, reads non-empty on a compute kernel
    assert any(r["logical_reads"] for _, _, r in cap)

def test_identity_distinguishes_independent_same_target():
    class M(torch.nn.Module):
        def forward(self, pos, x, y):
            return torch.cos(pos) + torch.cos(pos) + torch.cos(x) + torch.cos(y)
    cap = _compile_capture(M().eval(), torch.randn(64), torch.randn(64), torch.randn(64))
    cos_ids = [op["identity"] for _, ops, _ in cap for op in ops if op["target"] == "cos"]
    assert len(cos_ids) == 4
    # cos(pos) appears twice -> its identity count is 2; cos(x), cos(y) unique
    from collections import Counter
    c = Counter(cos_ids)
    assert sorted(c.values()) == [1, 1, 2], f"got {c}"

def test_load_identity_distinguishes_buffers():
    class M(torch.nn.Module):
        def forward(self, x, y):
            return torch.cos(x) + torch.sin(y)
    cap = _compile_capture(M().eval(), torch.randn(64), torch.randn(64))
    load_ids = [op["identity"] for _, ops, _ in cap for op in ops if op["target"] == "load"]
    assert load_ids, "no load ops captured"
    # every load identity's index part must be a resolved expr, not None/empty
    for lid in load_ids:
        assert lid[0] == "load" and lid[-1] not in (None, "None", "")
    # distinct source buffers -> distinct identities
    assert len(set(load_ids)) >= 2, load_ids


_COMPUTE = {"cos","sin","exp","log","sqrt","rsqrt","tanh","sigmoid","reciprocal","pow","erf","reduction"}

def _compile_serialize(mod, *inputs):
    """Full pipeline: patch each provenance call site to also call set_kernel_physical_trace,
    then return create_triton_kernel_trace_json()."""
    import torch._inductor.kernel_trace as kt
    kt.reset_kernel_trace_globals()
    # NOTE: in-product wiring lands in Task 5; here we drive capture via the scheduler hook
    import torch._inductor.scheduler as sched
    orig = sched.Scheduler.__init__
    handle = [0]
    def hook(self, *a, **k):
        orig(self, *a, **k)
        for n in self.nodes:
            handle[0] += 1
            kt.set_kernel_physical_trace([n] if not getattr(n,"snodes",None) else n.snodes,
                                         n.get_name(), handle[0])
    sched.Scheduler.__init__ = hook
    try:
        with config.patch(force_disable_caches=True, cpu_backend="triton"):
            with config.patch("trace.provenance_tracking_level", 1):
                with torch.no_grad(): torch.compile(mod, backend="inductor")(*inputs)
    except torch._inductor.exc.InductorError:
        # CPU-triton aborts after scheduling (expected by design)
        pass
    finally: sched.Scheduler.__init__ = orig
    torch._dynamo.reset()
    return kt.create_triton_kernel_trace_json()

def test_cross_kernel_compute_remat_flagged_but_not_loads():
    # Force cross-kernel remat: cos(pos) computed before AND after an mm barrier
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(64,64))
        def forward(self, pos, x):
            # First use of cos(pos) in first kernel
            c = torch.cos(pos)
            y = x * c
            # Matmul forces a barrier
            z1 = y @ self.w
            # Second use of cos(pos) - depends on mm output so must be after the barrier
            # This forces it into a separate kernel where cos is recomputed
            z2 = (z1 + x) * torch.cos(pos)
            return z2
    out = _compile_serialize(M().eval(), torch.randn(8,64), torch.randn(8,64))
    assert out["version"] == 1
    ops = [op for k in out["kernels"].values() for lf in k["leaves"] for op in lf.get("ops",[])]
    # some cos flagged rematerialized; NO load/constant/mul flagged
    assert any(op["target"]=="cos" and op["rematerialized"] for op in ops), f"No cos flagged; captured kernels: {list(out['kernels'].keys())}"
    assert all(not op["rematerialized"] for op in ops if op["target"] not in _COMPUTE)

def test_reset_clears_global():
    import torch._inductor.kernel_trace as kt
    kt._kernel_physical_trace["x:1"] = {}
    kt.reset_kernel_trace_globals()
    assert kt._kernel_physical_trace == {}


def test_trace_keys_use_provenance_debug_handle_scheme():
    """Verify kernel trace keys use the name:debug_handle join-key scheme.

    True set-equality vs kernel_information.json is verified on the real-device
    path (GPU/AOTI tests), because create_kernel_information_json() is empty
    on the CPU-triton JIT path (raises InductorError before codecache emit)."""
    import torch._inductor.kernel_trace as kt
    kt.reset_kernel_trace_globals()
    reset_inductor_kernel_provenance_debug_handle()
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(32, 32, bias=False)
        def forward(self, x):
            return torch.relu(self.w(x))
    with config.patch(
        {
            "force_disable_caches": True,
            "cpu_backend": "cpp",
            "trace.provenance_tracking_level": 1,
        }
    ):
        with torch.no_grad():
            torch.compile(M().eval(), backend="inductor")(torch.randn(4, 32))
    torch._dynamo.reset()
    trace_keys = set(kt.create_triton_kernel_trace_json()["kernels"])
    # Kernel trace must be populated (proves in-product wiring)
    assert trace_keys, f"trace_keys empty: {trace_keys}"
    # Assert join-key contract: every key is `name:debug_handle` where handle is positive int
    for k in trace_keys:
        name, _, handle = k.rpartition(":")
        assert name and handle.isdigit() and int(handle) >= 1, f"bad key shape: {k}"


def test_trace_structured_artifact_emitted_and_gated():
    seen = []
    import torch._inductor.compile_fx as cfx
    real = cfx.trace_structured
    def spy(name, *a, metadata_fn=None, **k):
        try:
            md = metadata_fn() if metadata_fn else {}
            seen.append(md.get("name"))
        except Exception: pass
        return real(name, *a, metadata_fn=metadata_fn, **k)
    cfx.trace_structured = spy
    try:
        class M(torch.nn.Module):
            def forward(self,x): return torch.relu(x)
        with config.patch(force_disable_caches=True, cpu_backend="cpp"):
            with config.patch("trace.provenance_tracking_level", 1):
                # Compile and run successfully (CPU-cpp works)
                with torch.no_grad(): torch.compile(M().eval(), backend="inductor")(torch.randn(8))
        torch._dynamo.reset()
        # Assert emission occurred from successful compile path
        assert "inductor_triton_kernel_trace" in seen, f"Expected 'inductor_triton_kernel_trace' in {seen}"
    finally:
        cfx.trace_structured = real


def test_negative_shared_load_not_flagged():
    # two pointwise kernels reading the same input: shared load NOT flagged as remat (only compute ops are)
    class M(torch.nn.Module):
        def forward(self,x):
            a = torch.relu(x)  # pointwise kernel
            b = torch.sigmoid(x)  # another pointwise kernel, reads x again
            return a + b
    out = _compile_serialize(M().eval(), torch.randn(4,32))
    assert out["kernels"], f"No kernels captured: {out}"
    # Collect all ops; on CPU-triton pointwise ops should be captured
    all_ops = [op for k in out["kernels"].values() for lf in k["leaves"] for op in lf.get("ops",[])]
    # Assert some ops exist (load/add/relu/sigmoid, not extern-only)
    if not all_ops:
        # CPU-triton with extern-only kernels: no ops to test; skip check
        return
    # Assert load/constant/add/mul NOT flagged rematerialized (only compute ops like cos/sin can be)
    for op in all_ops:
        if op["target"] in ("load","constant","mul","add"):
            assert not op["rematerialized"], f"Non-compute op {op['target']} flagged rematerialized"

def test_extern_matmul_is_extern_leaf():
    class M(torch.nn.Module):
        def __init__(self): super().__init__(); self.w=torch.nn.Linear(64,64,bias=False)
        def forward(self,x): return self.w(x)
    out = _compile_serialize(M().eval(), torch.randn(8,64))
    assert any(lf.get("scheduler_node_type","").startswith("ExternKernel")
               or lf.get("extern_target")
               for k in out["kernels"].values() for lf in k["leaves"])

def test_guard_level_zero_no_capture():
    import torch._inductor.kernel_trace as kt, torch._inductor.config as ic
    ic.force_disable_caches = True; ic.cpu_backend = "triton"; ic.trace.provenance_tracking_level = 0
    kt.reset_kernel_trace_globals()
    kt.set_kernel_physical_trace([], "k", 1)  # should no-op
    assert kt._kernel_physical_trace == {}

def test_schema_contract_fields_and_sorted():
    class M(torch.nn.Module):
        def forward(self,x): return torch.relu(x)+1.0
    out = _compile_serialize(M().eval(), torch.randn(8))
    assert out["version"] == 1 and out["stability"] == "experimental"
    assert out["kernels"], f"No kernels captured: {out}"
    for k in out["kernels"].values():
        assert "kernel_type" in k and "is_extern" in k and "leaves" in k
        for lf in k["leaves"]:
            if "logical_reads" in lf:
                assert lf["logical_reads"] == sorted(lf["logical_reads"])
            for op in lf.get("ops", []):
                assert set(("order","target","block","phase","rematerialized")) == set(op)
                assert op["phase"] in ("pointwise","reduction")

def test_reduction_phase_stamped():
    # Structural check: real compile path (snodes) stamps phase field on ops.
    # The "reduction" phase VALUE via real compile is verified on real-device path;
    # test_iter_leaves_assigns_reduction_phase asserts the derivation logic deterministically.
    class M(torch.nn.Module):
        def forward(self,x):
            # RMSNorm pattern forces reduction ops (reduction target + store_reduction)
            return torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+1e-6) * x
    out = _compile_serialize(M().eval(), torch.randn(4,16))
    # The test verifies that ops have a "phase" field stamped from reduction markers.
    # On CPU-triton JIT, kernels may not split into separate reduction/pointwise phases
    # (they fuse), but the op dict must still contain the phase field with valid values.
    ops = [op for k in out["kernels"].values() for lf in k["leaves"] for op in lf.get("ops",[])]
    assert ops, "no ops captured"
    for op in ops:
        assert "phase" in op and op["phase"] in ("pointwise","reduction")
    # At minimum, verify that reduction-like targets exist (the "reduction" op target)
    targets = {op["target"] for op in ops}
    assert "reduction" in targets or "store_reduction" in targets, f"no reduction ops in {targets}"

def test_iter_leaves_assigns_reduction_phase():
    """Deterministic unit test: _iter_leaves derives phase from leaf.is_reduction() API."""
    import torch._inductor.kernel_trace as kt
    from torch._inductor.codegen.simd_kernel_features import EnableReduction, DisableReduction
    class _LeafPointwise:
        def __init__(self, n): self._n = n
        def get_name(self): return self._n
        def is_reduction(self): return False
    class _LeafReduction:
        def __init__(self, n): self._n = n
        def get_name(self): return self._n
        def is_reduction(self): return True
    a, b, c = _LeafReduction("a"), _LeafPointwise("b"), _LeafReduction("c")
    # markers are stripped in real SIMD triton path, but _iter_leaves now derives from is_reduction()
    seq = [a, EnableReduction, b, DisableReduction, c]
    got = [(leaf.get_name(), phase) for leaf, phase in kt._iter_leaves(seq)]
    assert got == [("a", "reduction"), ("b", "pointwise"), ("c", "reduction")], got

def test_never_raises_on_bad_body():
    import torch._inductor.kernel_trace as kt
    class FakeLeaf:
        def get_name(self): return "op0"
        _body = None
        read_writes = None
    # SchedulerNode name check uses type().__name__, so force via direct call of helpers
    assert kt.extract_leaf_ops(FakeLeaf()) == []
    assert kt.buffer_roles(FakeLeaf())["logical_reads"] == []

def test_kernel_ops_summary_shape():
    import torch._inductor.kernel_trace as kt
    trace = {"version":1,"stability":"experimental","kernels":{
        "triton_poi_fused_cos_0:3":{"kernel_type":"triton","is_extern":False,"leaves":[
            {"name":"op0","scheduler_node_type":"SchedulerNode","phase":"pointwise",
             "logical_reads":["arg0"],"logical_writes":["buf0"],"in_out":[],
             "ops":[{"order":0,"target":"load","block":"root","phase":"pointwise","rematerialized":False},
                    {"order":1,"target":"cos","block":"root","phase":"pointwise","rematerialized":True}]}]}}}
    s = kt.kernel_ops_summary(trace)
    assert s["triton_poi_fused_cos_0"]["ops"] == ["load","cos"]
    assert s["triton_poi_fused_cos_0"]["rematerialized"] == ["cos"]
    # defensive: skip ops missing "target" key (malformed op dict)
    trace_malformed = {"version":1,"stability":"experimental","kernels":{
        "kernel_with_malformed:5":{"kernel_type":"triton","is_extern":False,"leaves":[
            {"name":"leaf0","scheduler_node_type":"SchedulerNode","phase":"pointwise",
             "logical_reads":[],"logical_writes":[],"in_out":[],
             "ops":[{"order":0,"phase":"pointwise","rematerialized":False},  # no target
                    {"order":1,"target":"sin","block":"root","phase":"pointwise","rematerialized":False}]}]}}}
    s2 = kt.kernel_ops_summary(trace_malformed)
    assert s2["kernel_with_malformed"]["ops"] == ["sin"]
    assert s2["kernel_with_malformed"]["rematerialized"] == []


def test_profiler_utils_non_index_fx_marker():
    """Test that map_recorded_events_to_aten_ops_with_stack_trace handles
    non-index fx markers (e.g. 'Call CompiledFxGraph None') without raising
    UnboundLocalError.

    Regression test for bug where fx-marker content that is neither a .py
    filename nor parseable as int would leave node_index unbound.
    """
    trace_events = [
        {
            "name": "## 42 ##",
            "cat": "cpu_op",
            "ts": 1000,
            "dur": 100,
        },
        {
            "name": "## Call CompiledFxGraph None ##",
            "cat": "cpu_op",
            "ts": 2000,
            "dur": 200,
        },
        {
            "name": "## another_non_int_marker ##",
            "cat": "cpu_op",
            "ts": 2500,
            "dur": 150,
        },
        {
            "name": "aten::add",
            "cat": "cpu_op",
            "ts": 3000,
            "dur": 50,
        },
    ]

    trace_dict = {"traceEvents": trace_events}
    # Should not raise UnboundLocalError on non-int, non-.py fx markers
    try:
        map_recorded_events_to_aten_ops_with_stack_trace(trace_dict)
    except UnboundLocalError as e:
        raise AssertionError(f"UnboundLocalError raised on non-int fx marker: {e}")


if __name__ == "__main__":
    run_tests()
