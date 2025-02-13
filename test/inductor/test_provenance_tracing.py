# Owner(s): ["module: inductor"]

import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import torch
from torch._inductor import config
from torch._inductor.debug import create_node_mapping
from torch._inductor.test_case import run_tests, TestCase
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


@requires_cuda
@config.patch("trace.enabled", True)
class TestProvenanceTracingArtifact(TestCase):
    """
    This test checks that generated provenance tracing artifact from "post_grad" to
    corresponding "inductor triton kernel node" is expected.
    """

    def _check_provenance_tracing_artifact(self, filepath):
        self.assertTrue(filepath.is_dir())
        filename = Path(filepath) / "inductor_triton_kernel_to_post_grad_nodes.json"
        with open(filename) as f:
            actual_data = json.load(f)
        # check that the generated provenance tracing artifact is expected
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
        self.assertEqual(sorted(actual_data.items()), sorted(expected_data.items()))

        filename = Path(filepath) / "inductor_provenance_tracking_node_mappings.json"
        with open(filename) as f:
            actual_data = json.load(f)
        # check that the generated provenance tracing artifact is expected
        expected_data = [
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
        self.assertEqual(sorted(actual_data.items()), sorted(expected_data))

    def test_triton_kernel_to_post_grad_tracing(self):
        a = torch.randn(10, 20, device="cuda")
        b = torch.randn(20, 30, device="cuda")
        c = torch.randn(10, 30, device="cuda")
        example_inputs = (a, b, c)

        model = Model()
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
                    self._check_provenance_tracing_artifact(filepath)
            finally:
                shutil.rmtree(filepath)


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
