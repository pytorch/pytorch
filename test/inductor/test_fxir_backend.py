"""
Test the FX IR backend.
"""

import torch
import unittest

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

    def _run_and_capture_graphs(self, opt, args) -> torch.fx.GraphModule:

        gms = []
        def generate(self, *args, **kwargs):
            nonlocal gms
            gms.append(self.gm)
            self._generate(*args, **kwargs)

        with unittest.mock.patch.object(torch._inductor.codegen.wrapper_fxir.WrapperFxCodegen, "generate", generate):
            try:
                result = opt(*args)
            except torch._inductor.exc.InductorError:
                # Expect this exception, since the FX backend doesn't support Python codegen.
                pass

        return gms

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


        # Get the FX graph from the backend.
        gms = self._run_and_capture_graphs(opt, args)
        result = gms[0](*args)
