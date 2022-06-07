# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest
import torch

class TestNonUniformObserver(unittest.TestCase):
    def test_calculate_qparams(self):
        """
        Test case 1
        Assume default parameters for b, k, n (empty observer)
        """
        obs1 = APoTObserver()

        obs1_result = obs1.calculate_qparams(signed=False)

        self.assertEqual(obs1_result[0], 0.0)
        self.assertTrue(torch.equal(obs1_result[1], torch.tensor([])))
        self.assertTrue(torch.equal(obs1_result[2], torch.tensor([])))

        """
        Test case 2
        APoT paper example: https://arxiv.org/pdf/1909.13144.pdf
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * note: b = k * n
        """
        obs2 = APoTObserver(max_val = 1.0, b = 4, k = 2)
        obs2_result = obs2.calculate_qparams(signed=False)

        quantization_levels2_test = torch.tensor([0.0000, 0.0208, 0.0417, 0.0625, 0.0833, 0.1250, 0.1667, 0.1875, 0.2500, 0.3333, 0.3750, 0.5000, 0.6667, 0.6875, 0.7500, 1.0000])
        level_indices2_test = torch.tensor([ 0, 3, 12, 15, 2, 14, 8, 11, 10, 1, 13, 9, 4, 7, 6, 5])

        self.assertEqual(obs2_result[0], (2/3))
        self.assertTrue(torch.allclose(obs2_result[1], quantization_levels2_test, 1e-04, 1e-04))
        self.assertTrue(torch.equal(obs2_result[2], level_indices2_test))

        """
        Test case 3
        Assume hardcoded parameters:
        * b = 6 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 3 (number of additive terms)
        """
        obs3 = APoTObserver(max_val = 1.0, b = 6, k = 2)

        obs3_result = obs3.calculate_qparams(signed=False)

        quantization_levels3 = obs3_result[1]

        quantization_levels3_test = torch.tensor([0.0000, 0.0022, 0.0045, 0.0067, 0.0089, 0.0112, 0.0134, 0.0156, 0.0179,\
            0.0223, 0.0268, 0.0312, 0.0357, 0.0379, 0.0446, 0.0469, 0.0536, 0.0625,\
            0.0714, 0.0737, 0.0759, 0.0781, 0.0893, 0.0938, 0.1071, 0.1094, 0.1250,\
            0.1429, 0.1473, 0.1518, 0.1562, 0.1786, 0.1875, 0.2143, 0.2188, 0.2500,\
            0.2857, 0.2879, 0.2946, 0.2969, 0.3036, 0.3125, 0.3571, 0.3594, 0.3750,\
            0.4286, 0.4375, 0.5000, 0.5714, 0.5737, 0.5759, 0.5781, 0.5893, 0.5938,\
            0.6071, 0.6094, 0.6250, 0.7143, 0.7188, 0.7500, 0.8571, 0.8594, 0.8750,\
            1.0000])

        level_indices3_test = torch.tensor([0, 3, 12, 15, 48, 51, 60, 63, 2, 14, 50, 62, 8, 11, 56, 59, 10, 58,\
        32, 35, 44, 47, 34, 46, 40, 43, 42, 1, 13, 49, 61, 9, 57, 33, 45, 41,\
        4, 7, 52, 55, 6, 54, 36, 39, 38, 5, 53, 37, 16, 19, 28, 31, 18, 30,\
        24, 27, 26, 17, 29, 25, 20, 23, 22, 21])

        self.assertEqual(obs3_result[0], (1/1.75))
        self.assertTrue(torch.allclose(obs3_result[1], quantization_levels3_test, 1e-04, 1e-04))
        self.assertTrue(torch.equal(obs3_result[2], level_indices3_test))

        """
        Test case 4
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * signed = True
        """
        obs4 = APoTObserver(max_val = 1.0, b = 4, k = 2)
        obs4_result = obs4.calculate_qparams(signed=True)

        quantization_levels4_test = torch.tensor([-4.0000, -3.0000, -2.7500, -2.6667, -2.5833, -2.3333, -2.0000, -1.5000,\
        -1.3333, -1.3333, -1.1667, -1.0000, -0.7500, -0.6667, -0.6667, -0.5833,\
        -0.5000, -0.3333, -0.3333, -0.2500, -0.1667, -0.1667, -0.0833, -0.0833,\
         0.0000,  0.0833,  0.0833,  0.1667,  0.1667,  0.2500,  0.3333,  0.3333,\
         0.5000,  0.5833,  0.6667,  0.6667,  0.7500,  1.0000,  1.1667,  1.3333,\
         1.3333,  1.5000,  2.0000,  2.3333,  2.5833,  2.6667,  2.7500,  3.0000,\
         4.0000])
        level_indices4_test = torch.tensor([48, 47, 46, 45, 44, 43, 41, 34, 42, 27, 20, 40, 39, 38, 13, 37, 33, 36,\
        26, 32, 31, 19, 30, 25, 24, 23, 18, 17, 29, 16, 12, 22, 15, 11, 35, 10,\
        9,  8, 28, 21,  6, 14,  7,  5,  4,  3,  2,  1,  0])

        self.assertEqual(obs4_result[0], (8/3))
        self.assertTrue(torch.allclose(obs4_result[1], quantization_levels4_test, 1e-04, 1e-04))
        self.assertTrue(torch.equal(obs4_result[2], level_indices4_test))

    # def test_override_calculate_qparams(self):
    #     t = torch.Tensor()
    #     obs = APoTObserver(t, t, t, 0, 0)

    #     raised = False
    #     try:
    #         obs._calculate_qparams(t, t)
    #     except Exception:
    #         raised = True
    #     self.assertFalse(raised, 'Exception raised')

if __name__ == '__main__':
    unittest.main()
