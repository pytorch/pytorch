# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest

class TestNonUniformObserver(unittest.TestCase):
    def test_calculate_qparams(self):
        """
        Test case 1
        Test that error is thrown when k == 0
        """
        obs1 = APoTObserver()

        with self.assertRaises(AssertionError):
            obs1_result = obs1.calculate_qparams(signed=False)

        """
        Test case 2
        APoT paper example: https://arxiv.org/pdf/1909.13144.pdf
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * note: b = k * n
        """
        obs2 = APoTObserver(max_val=1.0, b=4, k=2)
        obs2_result = obs2.calculate_qparams(signed=False)

        self.assertEqual(obs2_result[0], (2 / 3))

        quantlevels_size_test2 = int(len(obs2_result[1]))
        self.assertEqual(quantlevels_size_test2, 16)

        levelindices_size_test2 = int(len(obs2_result[2]))
        self.assertEqual(levelindices_size_test2, 16)

        unique_elts = True
        level_indices2_test_list = obs2_result[2].tolist()
        for i in range(16):
            if level_indices2_test_list.count(i) != 1:
                unique_elts = False

        self.assertTrue(unique_elts)

        """
        Test case 3
        Assume hardcoded parameters:
        * b = 6 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 3 (number of additive terms)
        """
        obs3 = APoTObserver(max_val=1.0, b=6, k=2)

        obs3_result = obs3.calculate_qparams(signed=False)

        self.assertEqual(obs3_result[0], (1 / 1.75))

        quantlevels_size_test3 = int(len(obs3_result[1]))
        self.assertEqual(quantlevels_size_test3, 64)

        levelindices_size_test3 = int(len(obs3_result[2]))
        self.assertEqual(levelindices_size_test3, 64)

        unique_elts = True
        level_indices3_test_list = obs3_result[2].tolist()
        for i in range(64):
            if level_indices3_test_list.count(i) != 1:
                unique_elts = False

        self.assertTrue(unique_elts)

        """
        Test case 4
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * signed = True
        """
        obs4 = APoTObserver(max_val=1.0, b=4, k=2)
        obs4_result = obs4.calculate_qparams(signed=True)

        self.assertEqual(obs4_result[0], (8 / 3))
        quantlevels_size_test4 = int(len(obs4_result[1]))
        self.assertEqual(quantlevels_size_test4, 49)

        quantlevels_test_list = obs4_result[1].tolist()
        negatives_contained = True
        for ele in quantlevels_test_list:
            if not (-ele) in quantlevels_test_list:
                negatives_contained = False
        self.assertTrue(negatives_contained)

        levelindices_size_test4 = int(len(obs4_result[2]))
        self.assertEqual(levelindices_size_test4, 49)

        unique_elts = True
        level_indices4_test_list = obs4_result[2].tolist()
        for i in range(49):
            if level_indices4_test_list.count(i) != 1:
                unique_elts = False

        self.assertTrue(unique_elts)

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
