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

        def arg_only_dim0(x):
            return [torch.relu(s) for s in torch.split(x, 2, 0)]

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

        def list_replace(x):
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

        def cm_with_list(x):
            return [torch.relu(s) for s in x.split([16, 16], dim=-1)]

        args = [
            torch.randn(2, 32),
        ]
        for fn, expected_split_norm_count in [
            (arg_only, 1),
            (arg_only_dim0, 1),
            (kwarg1, 1),
            (kwarg2, 1),
            (kwarg3, 1),
            (list_replace, 1),
            (multi_split, 17),
            (unequal_split, 1),
            (arg_only_cm, 1),
            (kwarg1_cm, 1),
            (kwarg2_cm, 1),
            (multi_split_cm, 17),
            (unequal_split_cm, 1),
            (cm_with_list, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn, dynamic=True)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["split_cat_norm"],
                expected_split_norm_count,
            )
            counters.clear()

    @torch._inductor.config.patch(split_cat_fx_passes=True)
    def test_consecutive_split_merge(self):
        def multi_split(x):
            return [torch.split(s, 2, 1) for s in torch.split(x, 2, 1)]

        def multi_split_2(x):
            return [torch.split(s, 1, 1) for s in torch.split(x, 2, 1)]

        def multi_split_2_neg_dim(x):
            return [torch.split(s, 1, 1) for s in torch.split(x, 2, -1)]

        def multi_split_with_sizes(x):
            return [torch.split(s, 2, 1) for s in torch.split(x, [16, 16], 1)]

        def multi_split_kwarg1(x):
            return [torch.split(s, 2, dim=1) for s in torch.split(x, 2, dim=1)]

        def multi_split_kwarg2(x):
            return [
                torch.split(s, split_size_or_sections=2, dim=1)
                for s in torch.split(x, split_size_or_sections=2, dim=1)
            ]

        def unequal_multi_split(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1.split([6, 4], 1))
            final_items.extend(item2.split([4, 4, 4], 1))

            return [torch.relu(s) for s in final_items]

        def unequal_multi_split_neg_index(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[-3]
            item1 = fs[-2]
            item2 = fs[-1]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1.split([6, 4], 1))
            final_items.extend(item2.split([4, 4, 4], 1))

            return [torch.relu(s) for s in final_items]

        # Shouldn't merge
        def diff_dims(x):
            return [torch.split(s, 2, dim=0) for s in torch.split(x, 2, dim=1)]

        def some_users_not_splits(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1.split([6, 4], 1))
            final_items.append(torch.sin(item2))

            return [torch.relu(s) for s in final_items]

        def split_with_cat(x):
            fs = torch.split(x, [4, 4, 24], dim=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = [item0, item1]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        def duplicate_getitems(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1_1 = fs[1]
            item1_2 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1_1.split([6, 4], 1))
            final_items.extend(item1_2)
            final_items.append(torch.sin(item2))

            return [torch.relu(s) for s in final_items]

        def duplicate_getitems_neg_index(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1_1 = fs[1]
            item1_2 = fs[-2]  # negative index
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1_1.split([6, 4], 1))
            final_items.extend(item1_2)
            final_items.append(torch.sin(item2))

            return [torch.relu(s) for s in final_items]

        def split_getitem_gap(x):
            fs = torch.split(x, [4, 4, 24], dim=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [
                item0,
            ]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        def split_getitem_out_of_order(x):
            fs = torch.split(x, [4, 4, 4, 20], dim=1)
            item0 = fs[0]
            item2 = fs[2]
            item1 = fs[1]
            item3 = fs[3]

            final_items = [item0, item2, item1]
            final_items.extend(item3.split((4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        args = [
            torch.randn(2, 32),
        ]
        for fn, expected_split_merged in [
            (multi_split, 16),
            (multi_split_2, 16),
            (multi_split_2_neg_dim, 16),
            (multi_split_with_sizes, 2),
            (multi_split_kwarg1, 16),
            (multi_split_kwarg2, 16),
            (unequal_multi_split, 3),
            (unequal_multi_split_neg_index, 3),
            (diff_dims, 0),
            (some_users_not_splits, 2),
            (split_with_cat, 1),
            (duplicate_getitems, 1),
            (duplicate_getitems_neg_index, 1),
            (split_getitem_gap, 1),
            (split_getitem_out_of_order, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn, dynamic=True)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["consecutive_split_merged"],
                expected_split_merged,
            )
            counters.clear()

    @torch._inductor.config.patch(split_cat_fx_passes=False)
    def test_config_flag_is_respected(self):
        def split_with_cat(x):
            fs = torch.split(x, [4, 4, 24], dim=-1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = [item0, item1]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        args = [
            torch.randn(2, 32),
        ]

        expected = split_with_cat(*args)
        actual = torch.compile(split_with_cat, dynamic=True)(*args)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            counters["inductor"]["consecutive_split_merged"],
            0,
        )
        self.assertEqual(
            counters["inductor"]["split_cat_norm"],
            0,
        )


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
