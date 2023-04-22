# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestSplitCatFxPasses(TestCase):
    @torch._inductor.config.patch(split_cat_fx_passes=True)
    def test_split_normalization(self):
        def arg_only(x):
            return [torch.relu(s) for s in torch.split(x, 2, 1)]

        def kwarg1(x):
            return [torch.relu(s) for s in torch.split(x, 2, dim=1)]

        def kwarg2(x):
            return [
                torch.relu(s) for s in torch.split(x, split_size_or_sections=2, dim=1)
            ]

        def kwarg3(x):
            return [
                torch.relu(s)
                for s in torch.split(tensor=x, split_size_or_sections=2, dim=-1)
            ]

        def no_replace(x):
            return [torch.relu(s) for s in torch.split(x, [16, 16], dim=1)]

        def multi_split(x):
            return [torch.split(s, 2, 1) for s in torch.split(x, 2, 1)]

        def unequal_split(x):
            return [torch.relu(s) for s in torch.split(x, 3, 1)]

        def arg_only_cm(x):
            return [torch.relu(s) for s in x.split(2, 1)]

        def kwarg1_cm(x):
            return [torch.relu(s) for s in x.split(2, dim=1)]

        def kwarg2_cm(x):
            return [torch.relu(s) for s in x.split(split_size=2, dim=1)]

        def multi_split_cm(x):
            return [s.split(2, 1) for s in x.split(2, 1)]

        def unequal_split_cm(x):
            return [torch.relu(s) for s in x.split(3, 1)]

        args = [
            torch.randn(2, 32),
        ]
        for fn, expected_split_norm_count in [
            (arg_only, 1),
            (kwarg1, 1),
            (kwarg2, 1),
            (kwarg3, 1),
            (no_replace, 0),
            (multi_split, 17),
            (unequal_split, 1),
            (arg_only_cm, 1),
            (kwarg1_cm, 1),
            (kwarg2_cm, 1),
            (multi_split_cm, 17),
            (unequal_split_cm, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn, dynamic=True)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["split_cat_norm"],
                expected_split_norm_count,
            )
            counters.clear()


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
