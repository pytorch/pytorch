"""
Test the FX IR backend.
"""

from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper_fxir import WrapperFXCodegen
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    requires_gpu,
)

@requires_gpu
class FXIRTestCase(InductorTestCase):
    device = GPU_TYPE

    def setUpClass(cls):
        super().setUpClass()

        # Register the FX backend.
        register_backend_for_device(device.type, TritonScheduling, WrapperFXCodegen)

    def test_basic(self):
        func = torch.add
        x = torch.rand(8, device=self.device)
        args = (x, x)
        opt = torch.compile(func, *args)

        # TODO take the FX IR from the backend
        # May want to mock create() function
