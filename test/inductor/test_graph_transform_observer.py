# Owner(s): ["module: inductor"]
import glob
import math
import os
import shutil
import tempfile
from unittest import skipIf

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._utils import _is_privateuse1_backend_available
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_ATTENTION
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, IS_LINUX
from torch.testing._internal.inductor_utils import HAS_TRITON


def _fused_attention_available() -> bool:
    # Out-of-tree accelerator backends provide fused SDPA not covered by
    # PLATFORM_SUPPORTS_FUSED_ATTENTION.
    return PLATFORM_SUPPORTS_FUSED_ATTENTION or _is_privateuse1_backend_available()


class TestGraphTransformObserver(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @skipIf(not HAS_TRITON, "requires triton")
    @skipIf(not _fused_attention_available(), "requires fused attention support")
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


instantiate_device_type_tests(TestGraphTransformObserver, globals(), except_for="cpu")


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
