"""
Test the FX IR backend.
"""

import torch
import unittest

from torch._inductor import config
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
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

    @config.patch(compile_threads=1) # Disable async compile
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

    def _compile_and_check(self, func, args, expected_num_kernels: int = 1):
        opt = torch.compile(func)

        # Get the FX graph from the backend.
        (gm,) = self._run_and_capture_graphs(opt, args)

        # Check code
        num_kernels = len(list(gm.graph.find_nodes(op="call_function", target=triton_kernel_wrapper_mutation)))
        self.assertEqual(num_kernels, expected_num_kernels)

        # Check accuracy
        result = gm(*args)[0]
        ref = func(*args)
        self.assertTrue(torch.allclose(result, ref))

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Register the FX backend.
        register_backend_for_device(cls.device, TritonScheduling, WrapperFxCodegen)

    def test_basic(self):
        func = torch.add
        args = [torch.rand(8, device=self.device) for _ in range(2)]

        self._compile_and_check(func, args)

    def test_multiple_kernels(self):
        def foo(x, y):
            return x.sum() + y.sum()

        args = [torch.rand(length, device=self.device) for length in [517, 1029]]
        self._compile_and_check(foo, args, expected_num_kernels=2)




