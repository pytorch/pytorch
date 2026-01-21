import torch
import unittest


class TestTopkRecompilation(unittest.TestCase):
    def test_topk_out_recompilations(self):
        """Test that topk with out= does not cause excessive recompilations"""

        def get_num_torch_recompiles():
            guard_failures = torch._dynamo.utils.guard_failures
            num_recompiles = [len(guard_failures[code]) for code in guard_failures]
            return 0 if len(num_recompiles) == 0 else max(num_recompiles)

        def topk_func(input, k, out):
            torch.topk(input, k, out=out)

        torch._dynamo.reset()
        opt_model = torch.compile(topk_func)

        values = torch.empty(3)
        indices = torch.empty(3, dtype=torch.long)

        x = torch.arange(1., 6.)
        opt_model(x, 3, out=(values, indices))
        recompiles_1 = get_num_torch_recompiles()

        x = torch.arange(1., 8.)
        opt_model(x, 3, out=(values, indices))
        recompiles_2 = get_num_torch_recompiles()

        x = torch.arange(1., 10.)
        opt_model(x, 3, out=(values, indices))
        recompiles_3 = get_num_torch_recompiles()

        self.assertLessEqual(recompiles_3, 2)


if __name__ == "__main__":
    unittest.main()
