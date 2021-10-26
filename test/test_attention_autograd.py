import torch
from itertools import product
from torch.testing._internal.common_utils import (TestCase, gradcheck, gradgradcheck, run_tests)

class TestAttentionAutograd(TestCase):

    def test_attn_gradcheck(self):
        for q_grad, k_grad, v_grad in product([False, True], [False, True], [False, True]):
            if not q_grad and not k_grad and not v_grad:
                continue
            q = torch.rand(2, 3, dtype=torch.double, requires_grad=q_grad)
            k = torch.rand(2, 3, dtype=torch.double, requires_grad=k_grad)
            v = torch.rand(2, 4, dtype=torch.double, requires_grad=v_grad)
            self.assertTrue(gradcheck(torch.attn, (q, k, v)))
            self.assertTrue(gradgradcheck(torch.attn, (q, k, v)))

if __name__ == '__main__':
    run_tests()
