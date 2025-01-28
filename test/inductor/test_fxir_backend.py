"""
Test the FX IR backend.
"""

import torch

from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    requires_gpu,
)

from torch._inductor.codegen.common import register_backend_for_device

@requires_gpu()
class FxirTestCase(InductorTestCase):
    device = GPU_TYPE

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Register the FX backend.
        register_backend_for_device(cls.device, TritonScheduling, WrapperFxCodegen)

    def test_basic(self):
        func = torch.add
        x = torch.rand(8, device=self.device)
        args = (x, x)
        opt = torch.compile(func)
        result = opt(*args)

        # TODO take the FX IR from the backend
        # May want to mock create() function
