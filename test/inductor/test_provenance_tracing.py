# Owner(s): ["module: inductor"]

import json
import logging
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from torch._dynamo.utils import detect_fake_mode
from torch._inductor import config
from torch._inductor.debug import (
    create_mapping_pre_post_grad_nodes,
    create_node_mapping_kernel_to_post_grad,
)
from torch._inductor.fx_passes.post_grad import post_grad_passes
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
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

    @requires_cuda  # test only works for cuda pattern matcher
    def test_pattern_matcher_transfer_meta(self):
        """
        Test that stack trace is transfered when node is decomposed in post_grad_passes
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

        x = torch.randn(8, 10).to("cuda")
        example_inputs = (x,)
        model = Model().to("cuda")

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


if __name__ == "__main__":
    run_tests()
