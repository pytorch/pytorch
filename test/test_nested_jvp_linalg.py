import torch
import unittest

class TestNestedJVPLinalg(unittest.TestCase):
    def test_linalg_det_nested_jvp(self):
        # Fixes #192540
        A = torch.tensor([[1.3, 0.4], [-0.7, 0.9]], dtype=torch.float64, requires_grad=True)
        f = torch.linalg.det

        H_true = torch.zeros(2, 2, 2, 2, dtype=torch.float64)
        H_true[0, 0, 1, 1] = H_true[1, 1, 0, 0] = 1.0
        H_true[0, 1, 1, 0] = H_true[1, 0, 0, 1] = -1.0

        fwd, rev = torch.func.jacfwd, torch.func.jacrev
        H_fwd_fwd = fwd(fwd(f))(A)
        H_rev_rev = rev(rev(f))(A)

        self.assertTrue(torch.allclose(H_rev_rev, H_true, atol=1e-12))
        self.assertTrue(torch.allclose(H_fwd_fwd, H_true, atol=1e-12))

if __name__ == '__main__':
    unittest.main()
