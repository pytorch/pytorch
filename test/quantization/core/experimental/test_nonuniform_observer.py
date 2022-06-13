# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest
import torch

class TestNonUniformObserver(unittest.TestCase):
    """
        Test case 1
        Test that error is thrown when k == 0
    """
    def test_calculate_qparams1(self):
        obs1 = APoTObserver(max_val=0.0, b=0, k=0)

        with self.assertRaises(NotImplementedError):
            obs.calculate_qparams()

    """
        Test case 2
        APoT paper example: https://arxiv.org/pdf/1909.13144.pdf
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * note: b = k * n
    """
    def test_calculate_qparams2(self):
        obs2 = APoTObserver(max_val=1.0, b=4, k=2)
        obs2_result = obs2.calculate_qparams(signed=False)

        # calculate expected gamma value
        gamma_test2 = 0
        for i in range(2):
            gamma_test2 += 2**(-i)

        gamma_test2 = 1 / gamma_test2

        # check gamma value
        self.assertEqual(obs2_result[0], gamma_test2)

        # check quantization levels size
        quantlevels_size_test2 = int(len(obs2_result[1]))
        quantlevels_size2 = 2**4
        self.assertEqual(quantlevels_size_test2, quantlevels_size2)

        # check level indices size
        levelindices_size_test2 = int(len(obs2_result[2]))
        self.assertEqual(levelindices_size_test2, 16)

        # check level indices unique values
        level_indices2_test_list = obs2_result[2].tolist()
        self.assertEqual(len(level_indices2_test_list), len(set(level_indices2_test_list)))

        obs2.quant_levels_visualization(obs2_result, "plt1")

    """
        Test case 3
        Assume hardcoded parameters:
        * b = 6 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 3 (number of additive terms)
    """
    def test_calculate_qparams3(self):
        obs3 = APoTObserver(max_val=1.0, b=6, k=2)

        obs3_result = obs3.calculate_qparams(signed=False)

        # calculate expected gamma value
        gamma_test3 = 0
        for i in range(3):
            gamma_test3 += 2**(-i)

        gamma_test3 = 1 / gamma_test3

        # check gamma value
        self.assertEqual(obs3_result[0], gamma_test3)

        # check quantization levels size
        quantlevels_size_test3 = int(len(obs3_result[1]))
        quantlevels_size3 = 2**6
        self.assertEqual(quantlevels_size_test3, quantlevels_size3)

        # check level indices size
        levelindices_size_test3 = int(len(obs3_result[2]))
        self.assertEqual(levelindices_size_test3, 64)

        # check level indices unique values
        level_indices3_test_list = obs3_result[2].tolist()
        self.assertEqual(len(level_indices3_test_list), len(set(level_indices3_test_list)))

        obs3.quant_levels_visualization(obs3_result, "plt2")

    """
        Test case 4
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * signed = True
    """
    def test_calculate_qparams4(self):
        obs4 = APoTObserver(max_val=1.0, b=4, k=2)
        obs4_result = obs4.calculate_qparams(signed=True)

        # calculate expected gamma value
        gamma_test4 = 0
        for i in range(2):
            gamma_test4 += 2**(-i)

        gamma_test4 = 1 / gamma_test4

        # check gamma value
        self.assertEqual(obs4_result[0], gamma_test4)

        # check quantization levels size
        quantlevels_size_test4 = int(len(obs4_result[1]))
        self.assertEqual(quantlevels_size_test4, 49)

        # check negatives of each element contained
        # in quantization levels
        quantlevels_test_list = obs4_result[1].tolist()
        negatives_contained = True
        for ele in quantlevels_test_list:
            if not (-ele) in quantlevels_test_list:
                negatives_contained = False
        self.assertTrue(negatives_contained)

        # check level indices size
        levelindices_size_test4 = int(len(obs4_result[2]))
        self.assertEqual(levelindices_size_test4, 49)

        # check level indices unique elements
        level_indices4_test_list = obs4_result[2].tolist()
        self.assertEqual(len(level_indices4_test_list), len(set(level_indices4_test_list)))

        obs4.quant_levels_visualization(obs4_result, "plt3")

if __name__ == '__main__':
    unittest.main()
