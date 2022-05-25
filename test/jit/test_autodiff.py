# Owner(s): ["oncall: jit"]

import torch

from torch.testing._internal.jit_utils import JitTestCase
from typing import List

class TestAutodiffJit(JitTestCase):
    def test_undefined_tensor_lists(self):
        def fn(tensor_list: List[torch.Tensor], add_tensor):
            cat = torch.cat(tensor_list, dim=1)
            r = torch.sin(cat + add_tensor)
            return r

        fn_s = torch.jit.script(fn)

        a = torch.rand((3, 6), requires_grad=True)
        b = torch.rand((3, 10), requires_grad=True)
        x = [a, b]
        y = torch.rand((3, 16), requires_grad=True)

        ret = fn_s(x, y)
        ret.sum().backward()
        ret = fn_s(x, y)
        ret.sum().backward()

        ret = fn_s(x, y)
        s = ret.sum()

        # backward_fn expects 2 inputs: (grad_output, current_grad_r)
        # current_grad_r is provided because we need to add this contribution
        # to grad_r when we return it.
        backward_fn = s.grad_fn.next_functions[0][0]

        # check behavior with defined tensor
        grad_out = torch.rand((3, 16))
        grad_inputs = backward_fn(grad_out, None)

        # expect 3 tensors: grad_y, grad_a, grad_b
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            self.assertTrue(isinstance(x, torch.Tensor))

        # now test with undefined grad_out
        grad_inputs = backward_fn(None, None)

        # expect all of them to be None
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            if x is not None:
                self.assertEqual(0, torch.max(torch.abs(x)).item())
