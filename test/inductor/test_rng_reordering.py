# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config as inductor_config
from torch._inductor.test_case import TestCase, run_tests


class TestRNGReordering(TestCase):
    def test_reorder_for_locality_preserves_randint_order(self):
        """
        Regression test: reorder_for_locality must not reorder RNG ops
        such as aten.randint, since they consume global RNG state.
        """

        with inductor_config.patch(fallback_random=True):

            def fn():
                torch.manual_seed(0)
                out = torch.randint(0, 100, (4, 1), dtype=torch.int64)
                _other = torch.randint(0, 100, (2, 1), dtype=torch.int64)
                return out

            compiled = torch.compile(fn, backend="inductor")

            torch.manual_seed(0)
            eager = fn()

            torch.manual_seed(0)
            compiled_out = compiled()

            torch.testing.assert_close(eager, compiled_out)


if __name__ == "__main__":
    run_tests()