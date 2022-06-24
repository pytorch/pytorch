# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest
import torch

class TestNonUniformObserver(unittest.TestCase):
    """
        Test case 1: calculate_qparams
        Test that error is thrown when k == 0
    """
    def test_calculate_qparams_invalid(self):
        obs = APoTObserver(b=0, k=0)

        with self.assertRaises(AssertionError):
            qparams = obs.calculate_qparams(signed=False, min_val=torch.tensor([0]), max_val=torch.tensor([0]))

    """
        Test case 2: calculate_qparams
        APoT paper example: https://arxiv.org/pdf/1909.13144.pdf
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * note: b = k * n
    """
    def test_calculate_qparams_2terms(self):
        obs = APoTObserver(b=4, k=2)
        alpha, gamma, quantization_levels, level_indices = obs.calculate_qparams(signed=False, min_val=torch.tensor([0]), max_val=torch.tensor([1]))

        # calculate expected gamma value
        gamma_test = 0
        for i in range(2):
            gamma_test += 2**(-i)

        gamma_test = 1 / gamma_test

        # check gamma value
        self.assertEqual(gamma, gamma_test)

        # check quantization levels size
        quantlevels_size_test = int(len(quantization_levels))
        quantlevels_size = 2**4
        self.assertEqual(quantlevels_size_test, quantlevels_size)

        # check level indices size
        levelindices_size_test = int(len(level_indices))
        self.assertEqual(levelindices_size_test, 16)

        # check level indices unique values
        level_indices_test_list = level_indices.tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

    """
        Test case 3: calculate_qparams
        Assume hardcoded parameters:
        * b = 6 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 3 (number of additive terms)
    """
    def test_calculate_qparams_3terms(self):
        obs = APoTObserver(b=6, k=2)

        alpha, gamma, quantization_levels, level_indices = obs.calculate_qparams(signed=False, min_val=torch.tensor([0]), max_val=torch.tensor([1]))

        # calculate expected gamma value
        gamma_test = 0
        for i in range(3):
            gamma_test += 2**(-i)

        gamma_test = 1 / gamma_test

        # check gamma value
        self.assertEqual(gamma, gamma_test)

        # check quantization levels size
        quantlevels_size_test = int(len(quantization_levels))
        quantlevels_size = 2**6
        self.assertEqual(quantlevels_size_test, quantlevels_size)

        # check level indices size
        levelindices_size_test = int(len(level_indices))
        self.assertEqual(levelindices_size_test, 64)

        # check level indices unique values
        level_indices_test_list = level_indices.tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

    """
        Test case 4: calculate_qparams
        Same as test case 2 but with signed = True
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * signed = True
    """
    def test_calculate_qparams_signed(self):
        obs = APoTObserver(b=4, k=2)
        alpha, gamma, quantization_levels, level_indices = obs.calculate_qparams(signed=True, min_val=torch.tensor([0]), max_val=torch.tensor([1]))

        # calculate expected gamma value
        gamma_test = 0
        for i in range(2):
            gamma_test += 2**(-i)

        gamma_test = 1 / gamma_test

        # check gamma value
        self.assertEqual(gamma, gamma_test)

        # check quantization levels size
        quantlevels_size_test = int(len(quantization_levels))
        self.assertEqual(quantlevels_size_test, 49)

        # check negatives of each element contained
        # in quantization levels
        quantlevels_test_list = quantization_levels.tolist()
        negatives_contained = True
        for ele in quantlevels_test_list:
            if not (-ele) in quantlevels_test_list:
                negatives_contained = False
        self.assertTrue(negatives_contained)

        # check level indices size
        levelindices_size_test = int(len(level_indices))
        self.assertEqual(levelindices_size_test, 49)

        # check level indices unique elements
        level_indices_test_list = level_indices.tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

    """
        Test forward method on hard-coded tensor with arbitrary values.
        Checks that alpha is max of abs value of max and min values in tensor.
    """
    def test_forward(self):
        obs = APoTObserver(b=4, k=2)

        X = torch.tensor([0.0, -100.23, -37.18, 3.42, 8.93, 9.21, 87.92])

        X = obs.forward(X)

        qparams = obs.calculate_qparams(True)

        alpha = qparams[0]

        min_val = torch.min(X)
        max_val = torch.max(X)

        expected_alpha = torch.max(-min_val, max_val)

        self.assertEqual(alpha, expected_alpha)


if __name__ == '__main__':
    unittest.main()
