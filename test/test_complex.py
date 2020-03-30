import torch
from torch.testing._internal.common_utils import TestCase, run_tests

devices = (torch.device('cpu'), torch.device('cuda:0'))

class TestComplexTensor(TestCase):
    def test_to_list_with_complex_64(self):
        # test that the complex float tensor has expected values and
        # there's no garbage value in the resultant list
        self.assertEqual(torch.zeros((2, 2), dtype=torch.complex64).tolist(), [[0j, 0j], [0j, 0j]])

    def test_exp_with_complex_64(self):
        import numpy as np

        w = torch.tensor([0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j], dtype=torch.complex64)
        w = torch.exp(w)

        x = np.array([0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
        x = np.exp(x)

        self.assertEqual(w * 2j, x * 2j)

    def test_exp_with_complex_128(self):
        import numpy as np

        w = torch.tensor([0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j], dtype=torch.complex128)
        w = torch.exp(w)

        x = np.array([0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
        x = np.exp(x)

        self.assertEqual(w * 2j, x * 2j)

    def test_addition_and_mult_with_complex_64(self):
        import numpy as np

        w = torch.tensor([3 + 3j], dtype=torch.complex64)
        w = 1.323j * w + 1.0

        x = np.array([3 + 3j], dtype=np.complex64)
        x = 1.323j * x + 1.0

        self.assertEqual(w, x)

    def test_exp_and_addition_with_complex_64(self):
        import numpy as np

        w = torch.tensor([3 + 3j], dtype=torch.complex64)
        w = torch.exp(w + 1.0)

        x = np.array([3 + 3j], dtype=np.complex64)
        x = np.exp(x + 1.0)

        self.assertEqual(w, x)

    def test_exp_and_addition_with_complex_128(self):
        import numpy as np

        w = torch.tensor([3 + 3j], dtype=torch.complex128)
        w = torch.exp(w + 1.0)

        x = np.array([3 + 3j], dtype=np.complex128)
        x = np.exp(x + 1.0)

        self.assertEqual(w, x)



if __name__ == '__main__':
    run_tests()
