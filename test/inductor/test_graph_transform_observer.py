# Owner(s): ["module: inductor"]
import glob
import math
import os
import shutil
import tempfile
import unittest

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_ATTENTION
from torch.testing._internal.common_device_type import (
    Capability,
    instantiate_device_type_tests,
    requires_capabilities,
)
from torch.testing._internal.common_utils import HardwareClassification, IS_LINUX


class TestGraphTransformObserver(TestCase):
    hw_classification = HardwareClassification.CUDA

    @requires_capabilities(Capability.lib.triton)
    @unittest.skipUnless(PLATFORM_SUPPORTS_FUSED_ATTENTION, "Requires fused attention")
    def test_sdpa_rewriter(self, device):
        if shutil.which("dot") is None:
            self.skipTest("Requires dot")
        try:
            import pydot  # noqa: F401
        except ImportError:
            self.skipTest("Requires pydot")

        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        log_url = tempfile.mkdtemp()
        inductor_config.trace.log_url_for_graph_xform = log_url
        inductor_config.force_disable_caches = True
        compiled_fn = torch.compile(dot_prod_attention, fullgraph=True)

        tensor_shape = (4, 2, 16, 32)
        q = torch.randn(tensor_shape, device=device)
        k = torch.randn(tensor_shape, device=device)
        v = torch.randn(tensor_shape, device=device)
        compiled_fn(q, k, v)

        found_input_svg = False
        found_output_svg = False
        for filepath_object in glob.glob(log_url + "/*"):
            if os.path.isfile(filepath_object):
                if filepath_object.endswith("input_graph.dot"):
                    found_input_svg = True
                elif filepath_object.endswith("output_graph.dot"):
                    found_output_svg = True

        self.assertTrue(found_input_svg)
        self.assertTrue(found_output_svg)


instantiate_device_type_tests(TestGraphTransformObserver, globals(), only_for="cuda")


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
