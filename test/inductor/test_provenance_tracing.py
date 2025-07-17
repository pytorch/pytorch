# Owner(s): ["module: inductor"]

import json
import logging
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from torch._inductor import config
from torch._inductor.debug import create_node_mapping
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.testing._internal.triton_utils import requires_cuda


try:
    from .test_aot_inductor_utils import AOTIRunnerUtil
except ImportError:
    from test_aot_inductor_utils import AOTIRunnerUtil


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
        self.weight = torch.randn(n, k, device="cuda")
        self.bias = torch.randn(n, device="cuda")

    def forward(self, a):
        return torch.nn.functional.linear(a, self.weight, self.bias)


@config.patch("trace.enabled", True)
@config.patch("trace.provenance_tracking", True)
class TestProvenanceTracingArtifact(TestCase):
    """
    This test checks that generated provenance tracing artifact from "post_grad" to
    corresponding "inductor triton kernel node" is expected.
    """

    def _check_provenance_tracing_artifact(self, filepath, expected_data):
        self.assertTrue(filepath.is_dir())
        filename = Path(filepath) / "inductor_generated_kernel_to_post_grad_nodes.json"
        with open(filename) as f:
            actual_data = json.load(f)
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

        model = Model()
        filepath = None

        for backend in ["aot_inductor", "inductor"]:
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
                    if device == "cuda":
                        expected_data = {
                            "triton_poi_fused_mul_0": ["mul"],
                            "triton_poi_fused_addmm_gelu_1": [
                                "mul_3",
                                "mul_1",
                                "add_tensor",
                                "add",
                                "erf",
                                "mul_2",
                            ],
                        }
                        if backend == "aot_inductor":
                            expected_data["aoti_torch_cuda_mm_out"] = ["mm_default"]
                        else:
                            expected_data["extern_kernels.mm"] = ["mm_default"]
                        self._check_provenance_tracing_artifact(filepath, expected_data)
                        expected_mapping = [
                            (
                                "cppCodeToPost",
                                {
                                    "triton_poi_fused_mul_0": ["mul"],
                                    "triton_poi_fused_addmm_gelu_1": [
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
                                    "mul": ["triton_poi_fused_mul_0"],
                                    "mul_3": ["triton_poi_fused_addmm_gelu_1"],
                                    "mul_1": ["triton_poi_fused_addmm_gelu_1"],
                                    "add_tensor": ["triton_poi_fused_addmm_gelu_1"],
                                    "add": ["triton_poi_fused_addmm_gelu_1"],
                                    "erf": ["triton_poi_fused_addmm_gelu_1"],
                                    "mul_2": ["triton_poi_fused_addmm_gelu_1"],
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
                        if backend == "aot_inductor":
                            expected_mapping[0][1]["aoti_torch_cuda_mm_out"] = [
                                "mm_default"
                            ]
                            expected_mapping[1][1]["mm_default"] = [
                                "aoti_torch_cuda_mm_out"
                            ]
                        else:
                            expected_mapping[0][1]["extern_kernels.mm"] = ["mm_default"]
                            expected_mapping[1][1]["mm_default"] = ["extern_kernels.mm"]
                        self._check_provenance_tracking_node_mappings(
                            filepath, expected_mapping
                        )
                    else:
                        assert device == "cpu"
                        # check the inductor kernel to post grad nodes mapping is expected for cpu
                        if backend == "aot_inductor":
                            expected_data = {
                                "cpp_fused_mul_0": ["mul"],
                                "aoti_torch_cpu_addmm_out": ["addmm"],
                                "cpp_fused_gelu_1": [
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
                                "cpp_fused_mul_0": ["mul"],
                                "cpp_fused_gelu_1": [
                                    "mul_3",
                                    "mul_1",
                                    "add",
                                    "erf",
                                    "mul_2",
                                ],
                                "extern_kernels.addmm": ["addmm"],
                            }
                        self._check_provenance_tracing_artifact(filepath, expected_data)

            finally:
                if filepath:
                    shutil.rmtree(filepath)

    @requires_cuda
    def test_triton_kernel_to_post_grad_tracing_cuda(self):
        self._test_triton_kernel_to_post_grad_tracing(device="cuda")

    @unittest.skipIf(HAS_GPU, "the test is only for cpu")
    def test_triton_kernel_to_post_grad_tracing_cpu(self):
        self._test_triton_kernel_to_post_grad_tracing(device="cpu")

    @requires_cuda
    def test_triton_kernel_to_post_grad_tracing_extern_kernel(self):
        M = 8
        N = 6
        K = 16
        model = Model3(N, K)
        batch = 2
        a = torch.randn(batch, M, K, device="cuda")
        example_inputs = (a,)
        filepath = None

        for backend in ["aot_inductor", "inductor"]:
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
                            "extern_kernels.addmm": ["addmm"],
                        }
                    else:
                        # backend = aot_inductor
                        expected_data = {
                            "aoti_torch_cuda_addmm_out": ["addmm"],
                            "triton_poi_fused_0": ["_tensor_constant1"],
                        }
                    self._check_provenance_tracing_artifact(filepath, expected_data)
            finally:
                if filepath:
                    shutil.rmtree(filepath)

    @requires_cuda
    def _test_pt_tracing_combo_kernel(self, backend):
        """This test checks that generated provenance tracing artifact from triton combo kernel to post grad nodes"""
        a = torch.randn(10, 10, device="cuda")
        b = torch.randn(20, 20, device="cuda")
        c = torch.randn(10, 10, device="cuda")
        example_inputs = (a, b, c)

        model = Model2()

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
            expected_data = {"triton_poi_fused_0": ["relu", "sigmoid", "tanh"]}
            self._check_provenance_tracing_artifact(filepath, expected_data)

    @requires_cuda
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

        result = create_node_mapping(
            pre_grad_graph_id,
            post_to_pre_grad_nodes_json,
            triton_kernel_to_post_grad_json,
        )
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


if __name__ == "__main__":
    run_tests()
