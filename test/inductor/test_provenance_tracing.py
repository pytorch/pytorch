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
    reset_inductor_kernel_provenance_debug_handle,
)
from torch._inductor.fx_passes.post_grad import post_grad_passes
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code, run_and_get_cpp_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import (
    requires_cuda_and_triton,
    requires_gpu_and_triton,
)


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
        c = torch.randn(10, 30, device=device)
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
                                    "mul": ["mul"],
                                    "mm_default": ["addmm"],
                                    "add_tensor": ["addmm"],
                                    "mul_1": ["gelu"],
                                    "mul_2": ["gelu"],
                                    "erf": ["gelu"],
                                    "add": ["gelu"],
                                    "mul_3": ["gelu"],
                                },
                            ),
                            (
                                "preToPost",
                                {
                                    "mul": ["mul"],
                                    "addmm": ["mm_default", "add_tensor"],
                                    "gelu": ["mul_1", "mul_2", "erf", "add", "mul_3"],
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

    def extract_code_line(self, s, i=-2):
        # Extract ith line
        return s.split("\n")[i].strip()

    @torch._inductor.config.patch({"trace.provenance_tracking_level": 2})
    @requires_gpu_and_triton
    def test_tlparse_kernel_stack_traces(self):
        device = GPU_TYPE
        model = Model4().to(device)
        x = torch.randn(8, 10).to(device)
        a = torch.randn(10, 20).to(device)
        b = torch.randn(20, 30).to(device)
        c = torch.randn(10, 30).to(device)
        example_inputs = (x, a, b, c)

        expected = {
            "triton_poi_fused_addmm_relu_sigmoid_threshold_backward_0:2": [
                "x = self.sigmoid(x)",
                "x = self.fc1(x)",
                "x = self.relu(x)",
            ],
            "triton_poi_fused_mul_1:3": [
                "d = a * 3.14",
            ],
            "triton_poi_fused_addmm_gelu_2:5": [
                "z = torch.nn.functional.gelu(y)",
                "y = torch.addmm(c, d, b)",
            ],
            "extern_kernels.mm:1": [
                "x = self.fc1(x)",
            ],
            "extern_kernels.mm:4": [
                "y = torch.addmm(c, d, b)",
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
                        f"Mismatch for key: {key}",
                    )

    @torch._inductor.config.patch(
        {"trace.provenance_tracking_level": 2, "max_autotune_gemm_backends": "ATEN"}
    )
    @requires_cuda_and_triton
    def test_deferred_triton_kernels(self):
        def foo(m, inp):
            a = m(inp)
            return a

        foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

        m = torch.nn.Linear(512, 512, bias=True).half().cuda()
        inp = torch.rand([1, 512]).half().cuda()

        with self._setup_provenance_capture() as payload_buffer:
            with torch.no_grad():
                _, out_code = run_and_get_code(foo_c, m, inp)
            payload_content = payload_buffer.getvalue().strip()
            data = json.loads(payload_content)
            self.assertTrue("a = m(inp)" in str(data))

            # Check that debug handle is in the output code
            FileCheck().check("Topologically Sorted Source Nodes: [a]").check(
                "[Provenance debug handles]"
            ).run(out_code[0])

    def _check_kernel_information_json(self, kernel_info, expected_kernels):
        """Validate kernel information JSON structure and content."""
        self.assertIsInstance(kernel_info, dict)

        for expected in expected_kernels:
            self.assertIn(
                expected,
                kernel_info,
                f"Expected kernel {expected} not found in {list(kernel_info)}",
            )

        for data in kernel_info.values():
            self.assertIsInstance(data, dict)
            for field in ["stack_traces", "post_grad_nodes", "pre_grad_nodes"]:
                self.assertIn(field, data)
                self.assertIsInstance(data[field], list)
                for item in data[field]:
                    self.assertIsInstance(item, str)

    @requires_gpu_and_triton
    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_kernel_information_generation(self):
        """Test basic kernel information generation in AOTI packages."""

        model = Model4().to(GPU_TYPE)
        x = torch.randn(8, 10, device=GPU_TYPE)
        a = torch.randn(10, 20, device=GPU_TYPE)
        b = torch.randn(20, 30, device=GPU_TYPE)
        c = torch.randn(10, 30, device=GPU_TYPE)
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
                f"kernel_information.json not found in extracted package at {json_path}",
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
                    f"Mismatch for key: {key}",
                )

                self.assertEqual(
                    sorted(kernel_info[key]["post_grad_nodes"]),
                    sorted(data["post_grad_nodes"]),
                    f"Mismatch for key: {key}",
                )

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
        c = torch.randn(10, 30).to(self.device)
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
        c = torch.randn(10, 30).to(self.device)
        example_inputs = (x, a, b, c)
        model = Model().to(self.device)

        ep = torch.export.export(model, example_inputs)
        _, code = run_and_get_cpp_code(torch._inductor.aoti_compile_and_package, ep)

        self.assertTrue("KernelContextGuard" not in code)

        with config.patch(
            {
                "trace.provenance_tracking_level": 1,
                "cpp.enable_kernel_profile": True,
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

            if self.device == "cuda":
                FileCheck().check(
                    """KernelContextGuard _ctx("aoti_torch_cuda_mm_out", R"("""
                ).check("AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(").check(
                    """KernelContextGuard _ctx("triton_poi_fused_addmm_relu_sigmoid_0", R"("""
                ).check("call_triton_poi_fused_addmm_relu_sigmoid_0(").check(
                    """KernelContextGuard _ctx("triton_poi_fused_mul_1", R"("""
                ).check("call_triton_poi_fused_mul_1(").check(
                    """KernelContextGuard _ctx("aoti_torch_cuda_mm_out", R"("""
                ).check("AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(").check(
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
@unittest.skipIf(not torch.cuda.is_available(), "No CUDA")
class TestProvenanceTracingKernelContextGpu(TestCase):
    device = "cuda"


copy_tests(
    ProvenanceTracingKernelContextTemplate,
    TestProvenanceTracingKernelContextGpu,
    "cuda",
)


if __name__ == "__main__":
    run_tests()
