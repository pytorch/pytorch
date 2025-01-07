# Owner(s): ["module: inductor"]

import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import torch
from torch._inductor import config
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
                "erf",
                "add_tensor",
                "mul_1",
                "add",
                "mul_2",
            ],
        }
        self.assertEqual(sorted(actual_data), sorted(expected_data))

    def test_triton_kernel_to_post_grad_tracing(self):
        a = torch.randn(10, 20, device="cuda")
        b = torch.randn(20, 30, device="cuda")
        c = torch.randn(10, 30, device="cuda")
        example_inputs = (a, b, c)

        model = Model()
        ep = torch.export._trace._export(model, example_inputs)
        gm = ep.module()

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
                            so_path = torch._inductor.aot_compile(gm, example_inputs)
                            optimized = AOTIRunnerUtil.load("cuda", so_path)
                            optimized(*example_inputs)
                        else:
                            compiled = torch.compile(gm, backend=backend)
                            compiled(*example_inputs)
                    self.assertEqual(len(cm.output), 1)
                    m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
                    self.assertTrue(m)
                    filepath = Path(m.group(1))
                    self._check_provenance_tracing_artifact(filepath)
            finally:
                shutil.rmtree(filepath)


if __name__ == "__main__":
    run_tests()
