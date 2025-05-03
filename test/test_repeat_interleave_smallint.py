import torch
import unittest

class TestRepeatInterleaveSmallInt(unittest.TestCase):
    def test_smallint_repeats(self):
        base = torch.arange(5, dtype=torch.int8)
        out = torch.repeat_interleave(base, torch.tensor(2, dtype=torch.int8))
        self.assertTrue(torch.equal(
            out, torch.tensor([0,0,1,1,2,2,3,3,4,4], dtype=torch.int8)
        ))

        base16 = torch.arange(3, dtype=torch.int16)
        out16 = torch.repeat_interleave(base16, torch.tensor([1,2,3], dtype=torch.int16))
        self.assertTrue(torch.equal(
            out16, torch.tensor([0,1,1,2,2,2], dtype=torch.int16)
        ))

if __name__ == "__main__":
    unittest.main()
