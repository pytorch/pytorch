"""
Test the FX IR backend.
"""

import torch

from torch._inductor.virtualized import V
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    requires_gpu,
)

from torch._inductor.codegen.common import (
    register_backend_for_device,
    get_wrapper_codegen_for_device,
)

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

        try:
            result = opt(*args)
        except torch._inductor.exc.InductorError:
            # Expect this exception, since we don't support Python codegen.
            pass

        # Get the FX graph from the backend.
        # TODO call this while mocking codegen:
        #    gm = V.graph.wrapper_code.gm
