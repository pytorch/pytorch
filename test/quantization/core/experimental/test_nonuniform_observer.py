# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest

class TestNonUniformObserver(unittest.TestCase):
    """
        Test case 1
        Test that error is thrown when k == 0
    """
    def test_calculate_qparams_invalid(self):
        obs = APoTObserver(max_val=0.0, b=0, k=0)

        with self.assertRaises(AssertionError):
            obs_result = obs.calculate_qparams(signed=False)

    """
        Test case 2
        APoT paper example: https://arxiv.org/pdf/1909.13144.pdf
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * note: b = k * n
    """
    def test_calculate_qparams_2terms(self):
        obs = APoTObserver(max_val=1.0, b=4, k=2)
        obs_result = obs.calculate_qparams(signed=False)

        # calculate expected gamma value
        gamma_test = 0
        for i in range(2):
            gamma_test += 2**(-i)

        gamma_test = 1 / gamma_test

        # check gamma value
        self.assertEqual(obs_result[0], gamma_test)

        # check quantization levels size
        quantlevels_size_test = int(len(obs_result[1]))
        quantlevels_size = 2**4
        self.assertEqual(quantlevels_size_test, quantlevels_size)

        # check level indices size
        levelindices_size_test = int(len(obs_result[2]))
        self.assertEqual(levelindices_size_test, 16)

        # check level indices unique values
        level_indices_test_list = obs_result[2].tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

    """
        Test case 3
        Assume hardcoded parameters:
        * b = 6 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 3 (number of additive terms)
    """
    def test_calculate_qparams_3terms(self):
        obs = APoTObserver(max_val=1.0, b=6, k=2)

        obs_result = obs.calculate_qparams(signed=False)

        # calculate expected gamma value
        gamma_test = 0
        for i in range(3):
            gamma_test += 2**(-i)

        gamma_test = 1 / gamma_test

        # check gamma value
        self.assertEqual(obs_result[0], gamma_test)

        # check quantization levels size
        quantlevels_size_test = int(len(obs_result[1]))
        quantlevels_size = 2**6
        self.assertEqual(quantlevels_size_test, quantlevels_size)

        # check level indices size
        levelindices_size_test = int(len(obs_result[2]))
        self.assertEqual(levelindices_size_test, 64)

        # check level indices unique values
        level_indices_test_list = obs_result[2].tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

    """
        Test case 4
        Same as test case 2 but with signed = True
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * signed = True
    """
    def test_calculate_qparams_signed(self):
        obs = APoTObserver(max_val=1.0, b=4, k=2)
        obs_result = obs.calculate_qparams(signed=True)

        # calculate expected gamma value
        gamma_test = 0
        for i in range(2):
            gamma_test += 2**(-i)

        gamma_test = 1 / gamma_test

        # check gamma value
        self.assertEqual(obs_result[0], gamma_test)

        # check quantization levels size
        quantlevels_size_test = int(len(obs_result[1]))
        self.assertEqual(quantlevels_size_test, 49)

        # check negatives of each element contained
        # in quantization levels
        quantlevels_test_list = obs_result[1].tolist()
        negatives_contained = True
        for ele in quantlevels_test_list:
            if not (-ele) in quantlevels_test_list:
                negatives_contained = False
        self.assertTrue(negatives_contained)

        # check level indices size
        levelindices_size_test = int(len(obs_result[2]))
        self.assertEqual(levelindices_size_test, 49)

        # check level indices unique elements
        level_indices_test_list = obs_result[2].tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

    """
        Test case 5
        Assume hardcoded parameters:
        * b = 6 (total number of bits across all terms)
        * k = 1 (base bitwidth, i.e. bitwidth of every term)
        * n = 6 (number of additive terms)
    """
    def test_calculate_qparams_k1(self):
        obs = APoTObserver(max_val=1.0, b=6, k=1)

        obs_result = obs.calculate_qparams(signed=False)

        # calculate expected gamma value
        gamma_test = 0
        for i in range(6):
            gamma_test += 2**(-i)

        gamma_test = 1 / gamma_test

        # check gamma value
        self.assertEqual(obs_result[0], gamma_test)

        # check quantization levels size
        quantlevels_size_test = int(len(obs_result[1]))
        quantlevels_size = 2**6
        self.assertEqual(quantlevels_size_test, quantlevels_size)

        # check level indices size
        levelindices_size_test = int(len(obs_result[2]))
        level_indices_size = 2**6
        self.assertEqual(levelindices_size_test, level_indices_size)

        # check level indices unique values
        level_indices_test_list = obs_result[2].tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

if __name__ == '__main__':
    unittest.main()
