# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest
import torch

class TestNonUniformObserver(unittest.TestCase):
    def test_calculate_qparams(self):
        """
        cases to test:
        1. works for empty observer
        2. works for normal observer (vasiliy playground example)
        3. alpha is gotten correctly -> test after checking w charlie about computation implementation
        4. works for normal observer of larger size
        """

        obs1 = APoTObserver()

        obs1_result = obs1.calculate_qparams(signed=False)

        print(obs1_result[1])

        self.assertEqual(obs1_result[0], 0.0)
        self.assertTrue(torch.equal(obs1_result[1], torch.tensor([])))
        self.assertTrue(torch.equal(obs1_result[2], torch.tensor([])))

        """
        APoT paper example: https://arxiv.org/pdf/1909.13144.pdf
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * note: b = k * n
        """
        obs2 = APoTObserver(max_val = 1.0, b = 4, k = 2)

        obs2_result = obs2.calculate_qparams(signed=False)

        quantization_levels2 = obs2_result[1]

        print("!!!!!!!")
        print(quantization_levels2)
        correctVal = torch.tensor([0.0000, 0.0208, 0.0417, 0.0625, 0.0833, 0.1250, 0.1667, 0.1875, 0.2500, 0.3333, 0.3750, 0.5000, 0.6667, 0.6875, 0.7500, 1.0000])
        print(correctVal)

        print("size 1: ", quantization_levels2.size())
        print("size 2: ", correctVal.size())

        print("$$$$$$$$$$$$$$$")
        for i in range(16):
            print(torch.eq(quantization_levels2[i], correctVal[i]))

        # print("-------------")
        # print(quantization_levels2.shape())
        # test = torch.tensor([0.0000, 0.0208, 0.0417, 0.0625, 0.0833, 0.1250, 0.1667, 0.1875, 0.2500, 0.3333, 0.3750, 0.5000, 0.6667, 0.6875, 0.7500, 1.0000])
        # print(test.shape())

        self.assertEqual(obs2_result[0], (2/3))
        self.assertTrue(torch.equal(torch.tensor([0.0000, 0.0208, 0.0417, 0.0625, 0.0833, 0.1250, 0.1667, 0.1875, 0.2500, 0.3333, 0.3750, 0.5000, 0.6667, 0.6875, 0.7500, 1.0000]), torch.tensor([0.0000, 0.0208, 0.0417, 0.0625, 0.0833, 0.1250, 0.1667, 0.1875, 0.2500, 0.3333, 0.3750, 0.5000, 0.6667, 0.6875, 0.7500, 1.0000])))
        self.assertTrue(torch.equal(quantization_levels2, obs2_result[1]))
        self.assertTrue(torch.allclose(quantization_levels2, correctVal, 1e-04, 1e-04))
        self.assertTrue(torch.equal(obs2_result[2], torch.tensor([])))

        # t = torch.Tensor()
        # obs = APoTObserver(t, t, t, 0, 0)

        # with self.assertRaises(NotImplementedError):
        #     obs.calculate_qparams()

        # t = torch.Tensor()
        # obs = APoTObserver(t, t, t, 0, 0)

        # raised = False
        # try:
        #     obs.calculate_qparams()
        # except Exception as e:
        #     raised = True
        #     print(e)
        # self.assertFalse(raised, 'Exception raised')

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
