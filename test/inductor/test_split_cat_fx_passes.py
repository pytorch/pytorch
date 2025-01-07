# Owner(s): ["module: inductor"]


import torch
from torch._dynamo.utils import counters, optimus_scuba_log
from torch._inductor.fx_passes.misc_patterns import numpy_compat_normalization
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_gpu


def patch(f):
    f = torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "remove_split_with_size_one_pass": {},
            "merge_getitem_cat_pass": {},
            "merge_splits_pass": {},
            "mutate_cat_pass": {},
            "split_cat_pass": {},
            "unbind_stack_pass": {},
        },
        post_grad_fusion_options={},
    )(f)
    return f


class TestSplitCatFxPasses(TestCase):
    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
        },
        post_grad_fusion_options={},
    )
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

        def normalize_reshape_with_dynamic_shape(x):
            return x.reshape(4, 16)

        args = [
            torch.randn(2, 32),
        ]
        for fn, dynamic, expected_split_norm_count in [
            (arg_only, False, 1),
            (arg_only_dim0, False, 1),
            (kwarg1, False, 1),
            (kwarg2, False, 1),
            (kwarg3, False, 1),
            (list_replace, False, 0),
            (multi_split, False, 17),
            (unequal_split, False, 1),
            (arg_only_cm, False, 1),
            (kwarg1_cm, False, 1),
            (kwarg2_cm, False, 1),
            (multi_split_cm, False, 17),
            (unequal_split_cm, False, 1),
            (cm_with_list, False, 1),
            (normalize_reshape_with_dynamic_shape, True, 0),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn, dynamic=dynamic)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["normalization_pass"],
                expected_split_norm_count,
                msg=f"for {fn}",
            )
            counters.clear()

    @patch
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

        def split_partial_getitem_cat(x):
            fs = torch.split(x, [4, 4, 24], dim=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [
                item0,
            ]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        def next_split_getitem_partial_used(x):
            fs = torch.split(x, [4, 4, 24], dim=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [item0]
            ns = item2.split((4, 4, 4, 4, 4, 4), 1)
            final_items.extend(ns[0:1])
            final_items.extend(ns[3:4])

            return torch.cat(final_items, dim=1)

        args = [
            torch.randn(2, 32),
        ]
        for fn, expected_split_merged in [
            (multi_split, 0),
            (multi_split_2, 16),
            (multi_split_2_neg_dim, 16),
            (multi_split_with_sizes, 2),
            (multi_split_kwarg1, 0),
            (multi_split_kwarg2, 0),
            (unequal_multi_split, 3),
            (unequal_multi_split_neg_index, 3),
            (diff_dims, 0),
            (some_users_not_splits, 2),
            (split_with_cat, 1),
            (duplicate_getitems, 1),
            (duplicate_getitems_neg_index, 1),
            (split_getitem_gap, 1),
            (split_getitem_out_of_order, 1),
            (next_split_getitem_partial_used, 1),
            (split_partial_getitem_cat, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["merge_splits_pass"],
                expected_split_merged,
            )
            if expected_split_merged > 0:
                self.assertIn("merge_splits_pass_pre_grad", optimus_scuba_log)
            counters.clear()

    @patch
    def test_split_cat_merge(self):
        def simple_split_cat(x):
            return torch.cat(torch.split(x, 4, dim=1), dim=1)

        def simple_split_cat_argspec1(x):
            return torch.cat(torch.split(x, 4, dim=1), 1)

        def simple_split_cat_argspec2(x):
            return torch.cat(tensors=torch.split(x, 4, dim=1), dim=1)

        def simple_split_cat_argspec3(x):
            return torch.cat(torch.split(x, 4, dim=1), -2)

        def simple_split_cat_argspec4(x):
            return torch.cat(tensors=torch.split(x, 4, dim=1), dim=-2)

        def simple_split_stack(x):
            return torch.stack(torch.split(x, 4, dim=1), dim=1)

        def simple_split_stack_argspec1(x):
            return torch.stack(torch.split(x, 4, dim=1), 1)

        def simple_split_stack_argspec2(x):
            return torch.stack(tensors=torch.split(x, 4, dim=1), dim=1)

        def split_cat_addn_args(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 5, 32, 16)] + split_output + [torch.ones(2, 6, 32, 16)],
                dim=1,
            )

        def split_stack_addn_args(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)]
                + split_output
                + [torch.ones(2, 4, 32, 16), torch.ones(2, 4, 32, 16)],
                dim=1,
            )

        def split_cat_addn_args_dim2(x):
            split_output = list(torch.split(x, 4, dim=2))
            return torch.cat(
                [torch.ones(2, 32, 5, 16)] + split_output + [torch.ones(2, 32, 6, 16)],
                dim=2,
            )

        # split_dim=1, cat_dim=2
        def split_cat_dim_mismatch(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        def split_stack_dim_mismatch(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        # split_dim=1, cat_dim=3
        def split_cat_dim_mismatch2(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=3,
            )

        def split_stack_dim_mismatch2(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=3,
            )

        # split_dim=2, cat_dim=0
        def split_cat_dim_mismatch3(x):
            split_output = list(torch.split(x, 4, dim=2))
            return torch.cat(
                [torch.ones(2, 32, 4, 16)] + split_output + [torch.ones(2, 32, 4, 16)],
                dim=0,
            )

        def split_stack_dim_mismatch3(x):
            split_output = list(torch.split(x, 4, dim=2))
            return torch.stack(
                [torch.ones(2, 32, 4, 16)] + split_output + [torch.ones(2, 32, 4, 16)],
                dim=0,
            )

        def input_shuffling(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=1,
            )

        def input_shuffling_stack(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=1,
            )

        def input_shuffling_dim_mismatch(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        def input_shuffling_dim_mismatch_stack(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        def input_shuffling_multiple_output(x):
            split_output = list(torch.split(x, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output[6])

            return cat1, stack1, relu1

        def input_shuffling_direct_output(x):
            split_output = list(torch.split(x, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            return cat1, stack1, split_output[6]

        def input_shuffling_multiple_output_same_ranges(x):
            split_output = list(torch.split(x, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

            cat2 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output[6])

            return cat1, cat2, stack1, relu1

        def unequal_split_multiple_output(x):
            split_output = list(torch.split(x, [2, 4, 4, 4, 4, 4, 8, 2], dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output[6])

            return cat1, stack1, relu1

        def multi_split_cat(x1, x2):
            split_output_1 = list(torch.split(x1, 4, dim=1))
            split_output_2 = list(torch.split(x2, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output_1[1], split_output_1[2], split_output_1[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output_2[1], split_output_2[2], split_output_2[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output_1[4],
                    split_output_1[5],
                    torch.ones(2, 4, 32, 16),
                    split_output_2[4],
                    split_output_2[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output_1[6])
            relu2 = torch.relu(split_output_2[6])

            return cat1, stack1, relu1, relu2

        # TODO: Add more tests:
        # * Cases where replacement shouldn't happen
        default_args = [
            torch.randn(2, 32, 32, 16),
        ]
        multi_args = [
            torch.randn(2, 32, 32, 16),
            torch.randn(2, 32, 32, 16),
        ]
        for (
            fn,
            expected_split_added,
            expected_split_removed,
            expected_cat_added,
            expected_cat_removed,
            expected_sections_removed,
            args,
        ) in [
            (simple_split_cat, 0, 0, 0, 0, 0, default_args),
            (simple_split_cat_argspec1, 0, 0, 0, 0, 0, default_args),
            (simple_split_cat_argspec2, 0, 0, 0, 0, 0, default_args),
            (simple_split_cat_argspec3, 0, 1, 0, 1, 7, default_args),
            (simple_split_cat_argspec4, 0, 1, 0, 1, 7, default_args),
            (simple_split_stack, 0, 1, 0, 1, 7, default_args),
            (simple_split_stack_argspec1, 0, 1, 0, 1, 7, default_args),
            (simple_split_stack_argspec2, 0, 1, 0, 1, 7, default_args),
            (split_cat_addn_args, 0, 1, 1, 1, 7, default_args),
            (split_stack_addn_args, 0, 1, 1, 1, 7, default_args),
            (split_cat_addn_args_dim2, 0, 1, 1, 1, 7, default_args),
            (split_cat_dim_mismatch, 0, 1, 1, 1, 7, default_args),
            (split_stack_dim_mismatch, 0, 1, 1, 1, 7, default_args),
            (split_cat_dim_mismatch2, 0, 1, 1, 1, 7, default_args),
            (split_stack_dim_mismatch2, 0, 1, 1, 1, 7, default_args),
            (split_cat_dim_mismatch3, 0, 1, 1, 1, 7, default_args),
            (split_stack_dim_mismatch3, 0, 1, 1, 1, 7, default_args),
            (input_shuffling, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_stack, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_dim_mismatch, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_dim_mismatch_stack, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_multiple_output, 1, 1, 2, 2, 3, default_args),
            (input_shuffling_direct_output, 1, 1, 2, 2, 3, default_args),
            (unequal_split_multiple_output, 1, 1, 2, 2, 3, default_args),
            (multi_split_cat, 1, 1, 2, 2, 3, multi_args),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["scmerge_split_added"],
                expected_split_added,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_removed"],
                expected_split_removed,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_added"],
                expected_cat_added,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_removed"],
                expected_cat_removed,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_sections_removed"],
                expected_sections_removed,
            )
            counters.clear()

    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={},
    )
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
        actual = torch.compile(split_with_cat)(*args)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            counters["inductor"]["merge_splits_pass"],
            0,
        )
        self.assertEqual(
            counters["inductor"]["normalization_pass"],
            0,
        )

    @patch
    def test_split_cat_merge_mutation(self):
        args = [
            torch.randn(2, 32, 32, 16),
        ]

        def split_cat_mutation(x):
            splits = torch.split(x, 4, dim=1)
            splits[1].copy_(splits[0])
            return torch.cat(splits, dim=1)

        expected = split_cat_mutation(*args)
        actual = torch.compile(split_cat_mutation)(*args)

        torch.testing.assert_close(actual, expected)

        self.assertEqual(counters["inductor"]["scmerge_split_removed"], 0)
        self.assertEqual(counters["inductor"]["scmerge_cat_removed"], 0)

    @patch
    def test_split_squeeze(self):
        def split_squeeze_stack(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_callmethod(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [s.squeeze(1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_callmethod_none_dim(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [s.squeeze() for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_kwarg1(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, dim=1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_kwarg1_callmethod(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [s.squeeze(dim=1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_multi_squeeze_users(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return (
                torch.stack(split_items),
                torch.relu(split_items[0]),
                torch.tanh(split_items[1]),
            )

        def split_size_not_1(x):
            items = list(torch.split(x, 2, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items)

        def dim_mismatch(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 0) for s in items]
            return torch.stack(split_items)

        def other_users(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items), torch.relu(items[0])

        def other_users_2(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items[1:]]
            return torch.stack(split_items), torch.relu(items[0])

        def graph_should_be_topological_sorted(x):
            output = []
            for t in x.split(1):
                output.append(torch.sin(t.squeeze(dim=0)))
            output = torch.stack(output)
            return output

        args = [
            torch.randn(2, 32),
        ]
        for fn, split_squeeze_replaced in [
            (split_squeeze_stack, 1),
            (split_squeeze_stack_callmethod, 1),
            # TODO handle none dim
            (split_squeeze_stack_callmethod_none_dim, 0),
            (split_squeeze_stack_kwarg1, 1),
            (split_squeeze_stack_kwarg1_callmethod, 1),
            (split_squeeze_multi_squeeze_users, 1),
            (split_size_not_1, 0),
            (dim_mismatch, 0),
            (other_users, 0),
            (other_users_2, 0),
            (graph_should_be_topological_sorted, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["split_cat_pass"],
                split_squeeze_replaced,
            )
            counters.clear()

    @patch
    def test_unbind_stack(self):
        def unbind_stack(x):
            return torch.stack(torch.unbind(x, 1), 1)

        def unbind_cat(x):  # noqa: F841
            return torch.cat(torch.unbind(x, dim=-3), 1)

        def unbind_stack_argspec1(x):
            return torch.stack(torch.unbind(input=x, dim=1), dim=1)

        def unbind_stack_argspec2(x):
            return torch.stack(tensors=torch.unbind(x, dim=1), dim=1)

        def dim_mismatch(x):
            return torch.stack(torch.unbind(x, dim=1), 0)

        def split_squeeze_stack(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items, 1)

        def split_squeeze_stack_callmethod(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items, 1)

        def other_users(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items, 1), torch.relu(items[0])

        def other_users_2(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items[1:]]
            return torch.stack(split_items, 1), torch.relu(items[0])

        def unbind_cat_addn_args(x):
            split_output = list(torch.unbind(x, dim=1))

            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=1,
            )

        def unbind_stack_addn_args(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.stack(
                [torch.ones(2, 32, 16)]
                + split_output
                + [torch.ones(2, 32, 16), torch.ones(2, 32, 16)],
                dim=1,
            )

        def unbind_cat_addn_args_dim2(x):
            split_output = list(torch.unbind(x, dim=2))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        # split_dim=1, cat_dim=2
        def unbind_cat_dim_mismatch(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        def unbind_stack_dim_mismatch(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.stack(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        def unbind_cat_multi_users(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=1,
            ), torch.stack(
                [torch.ones(2, 32, 16)]
                + split_output
                + [torch.ones(2, 32, 16), torch.ones(2, 32, 16)],
                dim=1,
            )

        def unbind_cat_multi_users_diff_dims(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=1,
            ), torch.stack(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        args = [
            torch.randn(2, 32, 32, 16),
        ]
        for (
            fn,
            expected_unbind_added,
            expected_unbind_removed,
            expected_cat_added,
            expected_cat_removed,
            expected_sections_removed,
            expected_unbind_normalized,
        ) in [
            (unbind_stack, 0, 1, 0, 1, 31, 2),
            (unbind_stack_argspec1, 0, 1, 0, 1, 31, 2),
            (unbind_stack_argspec2, 0, 1, 0, 1, 31, 2),
            (dim_mismatch, 0, 1, 0, 1, 31, 2),
            (split_squeeze_stack, 0, 1, 0, 1, 31, 2),
            (split_squeeze_stack_callmethod, 0, 1, 0, 1, 31, 2),
            (other_users, 0, 0, 0, 0, 0, 2),
            (other_users_2, 0, 0, 0, 0, 0, 2),
            (unbind_cat_addn_args, 0, 1, 1, 1, 31, 1),
            (unbind_stack_addn_args, 0, 1, 1, 1, 31, 2),
            (unbind_cat_addn_args_dim2, 0, 1, 1, 1, 31, 1),
            (unbind_cat_dim_mismatch, 0, 1, 1, 1, 31, 1),
            (unbind_stack_dim_mismatch, 0, 1, 1, 1, 31, 2),
            (unbind_cat_multi_users, 0, 1, 2, 2, 31, 2),
            (unbind_cat_multi_users_diff_dims, 0, 1, 2, 2, 31, 2),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["scmerge_split_added"],
                expected_unbind_added,
                msg=f"for {fn}",
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_removed"],
                expected_unbind_removed,
                msg=f"for {fn}",
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_added"],
                expected_cat_added,
                msg=f"for {fn}",
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_removed"],
                expected_cat_removed,
                msg=f"for {fn}",
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_sections_removed"],
                expected_sections_removed,
                msg=f"for {fn}",
            )
            self.assertEqual(
                counters["inductor"]["normalization_pass"],
                expected_unbind_normalized,
                msg=f"for {fn}",
            )
            counters.clear()

    @patch
    def test_split_cat_new_patterns(self):
        def split_cat_split(x):
            l1_out = torch.split(x, [200, 50, 50, 20, 20, 20, 20, 20, 20, 50, 30], 1)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item10 = l1_out[10]
            cat_1 = torch.cat((item0, item1), 1)
            cat_2 = torch.cat((item9, item10), 1)
            l2_out = torch.split(cat_1, [50, 120, 80], 1)
            l3_out = torch.split(cat_2, [10, 20, 50], 1)
            item11 = l2_out[0]
            item12 = l2_out[1]
            item13 = l2_out[2]
            item14 = l3_out[0]
            item15 = l3_out[1]
            item16 = l3_out[2]

            output = torch.cat(
                [
                    item11,
                    item12,
                    item13,
                    item14,
                    item15,
                    item16,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                ],
                1,
            )
            return output

        def split_cat_split_kwarg(x):
            l1_out = torch.split(
                x, [200, 50, 50, 20, 20, 20, 20, 20, 20, 50, 30], dim=1
            )
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item10 = l1_out[10]
            cat_1 = torch.cat((item0, item1), dim=1)
            cat_2 = torch.cat((item9, item10), dim=1)
            l2_out = torch.split(cat_1, [50, 120, 80], dim=1)
            l3_out = torch.split(cat_2, [10, 20, 50], dim=1)
            item11 = l2_out[0]
            item12 = l2_out[1]
            item13 = l2_out[2]
            item14 = l3_out[0]
            item15 = l3_out[1]
            item16 = l3_out[2]

            output = torch.cat(
                [
                    item11,
                    item12,
                    item13,
                    item14,
                    item15,
                    item16,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                ],
                dim=1,
            )
            return output

        def remove_cat_node_with_all_getitmes(x):
            l1_out = torch.split(
                x, [50, 50, 200, 20, 20, 20, 20, 20, 40, 10, 50], dim=0
            )
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item10 = l1_out[10]
            cat = torch.cat(
                (
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                    item9,
                    item10,
                ),
                dim=0,
            )
            cat_1 = torch.cat((item0, item1), dim=0)
            cat_2 = torch.cat((item0, item10), dim=0)
            l2_out = torch.split(cat_1, [20, 30, 50], dim=0)
            l3_out = torch.split(cat_2, [10, 60, 30], dim=0)
            item11 = l2_out[0]
            item12 = l2_out[1]
            item13 = l2_out[2]
            item14 = l3_out[0]
            item15 = l3_out[1]
            item16 = l3_out[2]

            output = torch.cat(
                [
                    item11,
                    item12,
                    item13,
                    item14,
                    item15,
                    item16,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                ],
                dim=0,
            )
            return torch.cat((output, cat), dim=0)

        def mutate_cat_node_with_some_getitmes(x):
            l1_out = torch.split(
                x, [50, 50, 200, 20, 20, 20, 20, 20, 40, 10, 50], dim=0
            )
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item10 = l1_out[10]
            cat = torch.cat(
                (
                    item6,
                    item7,
                    item8,
                    item9,
                    item10,
                    item2,
                    item3,
                    item4,
                    item5,
                ),
                dim=0,
            )
            cat_1 = torch.cat((item0, item1), dim=0)
            cat_2 = torch.cat((item0, item10), dim=0)
            l2_out = torch.split(cat_1, [20, 30, 50], dim=0)
            l3_out = torch.split(cat_2, [10, 60, 30], dim=0)
            item11 = l2_out[0]
            item12 = l2_out[1]
            item13 = l2_out[2]
            item14 = l3_out[0]
            item15 = l3_out[1]
            item16 = l3_out[2]

            output = torch.cat(
                [
                    item11,
                    item12,
                    item13,
                    item14,
                    item15,
                    item16,
                    item2,
                ],
                dim=0,
            )
            return torch.cat((output, cat), dim=0)

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "split_cat_to_slices_pass": {},
            },
            post_grad_fusion_options={},
        )
        def split_cat_to_slices(x):
            x_c = x.clone()
            x_c_2 = x.clone()
            l1_out = torch.split(x, [50, 50, 50, 50, 50, 50, 50, 50, 50, 50], dim=0)
            l2_out = torch.split(x_c, [50, 50, 50, 50, 50, 50, 50, 50, 50, 50], dim=0)
            l3_out = torch.split(x_c_2, [100, 100, 100, 100, 100], dim=0)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item0_c = l2_out[0]
            item1_c = l2_out[1]
            item2_c = l2_out[2]
            item3_c = l2_out[3]
            item4_c = l2_out[4]
            item5_c = l2_out[5]
            item6_c = l2_out[6]
            item7_c = l2_out[7]
            item8_c = l2_out[8]
            item9_c = l2_out[9]
            item0_c_2 = l3_out[0]
            item1_c_2 = l3_out[1]
            item2_c_2 = l3_out[2]
            item3_c_2 = l3_out[3]
            item4_c_2 = l3_out[4]
            other = item0.clone()
            return torch.cat(
                [
                    other,
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                    item9,
                    item4_c,
                    item5_c,
                    item6_c,
                    item7_c,
                    item8_c,
                    item9_c,
                    item0_c,
                    item1_c,
                    item2_c,
                    item3_c,
                    item0_c_2,
                    item1_c_2,
                    item2_c_2,
                    item3_c_2,
                    item4_c_2,
                ],
                dim=0,
            )

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "unbind_cat_to_view_pass": {},
            },
            post_grad_fusion_options={},
        )
        def unbind_cat_to_view(x):
            y = x.view(10, 50, 500)
            z = x.view(10, 50, 500)
            l1_out = torch.unbind(y, dim=0)
            l2_out = torch.unbind(z, dim=0)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item2_0 = l2_out[0]
            item2_1 = l2_out[1]
            item2_2 = l2_out[2]
            item2_3 = l2_out[3]
            item2_4 = l2_out[4]
            item2_5 = l2_out[5]
            item2_6 = l2_out[6]
            item2_7 = l2_out[7]
            item2_8 = l2_out[8]
            item2_9 = l2_out[9]
            other1 = item7.clone()
            other2 = item8.clone()
            other3 = item9.clone()
            cat = torch.cat(
                [
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    other1,
                    item2_0,
                    item2_1,
                    item2_2,
                    item2_3,
                    item2_4,
                    item2_5,
                    item2_6,
                    item2_7,
                    item2_8,
                    item2_9,
                    other2,
                    other3,
                ],
                dim=1,
            )
            return cat

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "split_stack_to_cats_pass": {},
            },
            post_grad_fusion_options={},
        )
        def split_stack_to_cats_same_dim(x):
            x_c = x.view(10, 50, 500)
            l1_out = torch.unbind(x_c, dim=0)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            split1 = torch.split(item0, [250, 250], dim=1)
            split2 = torch.split(item1, [250, 250], dim=1)
            split3 = torch.split(item2, [250, 250], dim=1)
            split4 = torch.split(item3, [250, 250], dim=1)
            split5 = torch.split(item4, [250, 250], dim=1)
            split6 = torch.split(item5, [250, 250], dim=1)
            getitem0, getitem1 = split1[0], split1[1]
            getitem2, getitem3 = split2[0], split2[1]
            getitem4, getitem5 = split3[0], split3[1]
            getitem6, getitem7 = split4[0], split4[1]
            getitem8, getitem9 = split5[0], split5[1]
            getitem10, getitem11 = split6[0], split6[1]
            getitem0_c = getitem0.clone()
            getitem1_c = getitem1.clone()
            getitem2_c = getitem2.clone()
            return torch.stack(
                (
                    getitem0,
                    getitem1,
                    getitem2,
                    getitem3,
                    getitem4,
                    getitem5,
                    getitem0_c,
                    getitem1_c,
                    getitem6,
                    getitem7,
                    getitem8,
                    getitem9,
                    getitem10,
                    getitem11,
                    getitem2_c,
                ),
                dim=1,
            )

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "split_stack_to_cats_pass": {},
            },
            post_grad_fusion_options={},
        )
        def split_stack_to_cats_different_dim(x):
            l1_out = torch.split(x, [100, 100, 100, 100, 100], dim=1)
            x_c = x.clone()
            l2_out = torch.split(x_c, [100, 100, 100, 100, 100], dim=1)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item0_c = l2_out[0]
            item1_c = l2_out[1]
            item2_c = l2_out[2]
            item3_c = l2_out[3]
            item4_c = l2_out[4]
            other_1 = item0.clone()
            other_2 = item1.clone()
            other_3 = item2.clone()
            return torch.stack(
                (
                    other_1,
                    other_2,
                    other_3,
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item0_c,
                    item1_c,
                    item2_c,
                    item3_c,
                    item4_c,
                ),
                dim=2,
            )

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "unbind_stack_to_slices_pass": {},
            },
            post_grad_fusion_options={},
        )
        def unbind_stack_to_slices(x):
            x_1 = x.view(50, 10, 500)
            l1_out = torch.unbind(x_1, dim=1)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            other_1 = item0.clone()
            other_2 = item1.clone()
            other_3 = item2.clone()
            return torch.stack(
                (
                    other_1,
                    other_2,
                    other_3,
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                    item9,
                ),
                dim=1,
            )

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "normalization_pass": {},
                "move_reshape_out_of_split_stack_pass": {},
            },
            post_grad_fusion_options={},
        )
        def move_reshape_out_of_split_stack(x):
            x_c = x.view(50000, 5)
            l1_out = torch.split(x_c, [1, 1, 1, 1, 1], dim=1)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            reshape0 = item0.reshape(-1, 5)
            reshape1 = item1.reshape(-1, 5)
            reshape2 = item2.reshape(-1, 5)
            reshape3 = item3.reshape(-1, 5)
            reshape4 = item4.reshape(-1, 5)
            other0 = reshape0.clone()
            other1 = reshape1.clone()
            other2 = reshape2.clone()
            other3 = reshape3.clone()
            return torch.stack(
                (
                    other0,
                    other1,
                    other2,
                    reshape0,
                    reshape1,
                    reshape2,
                    reshape3,
                    reshape4,
                    other3,
                ),
                dim=0,
            )

        args = [
            torch.randn(500, 500),
        ]
        for (
            fn,
            expected_getitem_cat_merged,
            expected_cat_removed,
            expected_split_cat_to_slices,
            exptected_unbind_to_cat_view,
            expected_split_stack_to_cats,
            exptected_unbind_stack_to_slices,
            expected_move_reshape_out_of_split_stack,
        ) in [
            (split_cat_split, 2, 0, 0, 0, 0, 0, 0),
            (split_cat_split_kwarg, 2, 0, 0, 0, 0, 0, 0),
            (remove_cat_node_with_all_getitmes, 0, 2, 0, 0, 0, 0, 0),
            (mutate_cat_node_with_some_getitmes, 0, 1, 0, 0, 0, 0, 0),
            (split_cat_to_slices, 0, 0, 1, 0, 0, 0, 0),
            (unbind_cat_to_view, 0, 0, 0, 1, 0, 0, 0),
            (split_stack_to_cats_same_dim, 0, 0, 0, 0, 1, 0, 0),
            (split_stack_to_cats_different_dim, 0, 0, 0, 0, 1, 0, 0),
            (unbind_stack_to_slices, 0, 0, 0, 0, 0, 1, 0),
            (move_reshape_out_of_split_stack, 0, 0, 0, 0, 0, 0, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["merge_getitem_cat_pass"],
                expected_getitem_cat_merged,
            )
            self.assertEqual(
                counters["inductor"]["mutate_cat_pass"],
                expected_cat_removed,
            )
            self.assertEqual(
                counters["inductor"]["split_cat_to_slices_pass"],
                expected_split_cat_to_slices,
            )
            self.assertEqual(
                counters["inductor"]["unbind_cat_to_view_pass"],
                exptected_unbind_to_cat_view,
            )
            self.assertEqual(
                counters["inductor"]["split_stack_to_cats_pass"],
                expected_split_stack_to_cats,
            )
            self.assertEqual(
                counters["inductor"]["unbind_stack_to_slices_pass"],
                exptected_unbind_stack_to_slices,
            )
            self.assertEqual(
                counters["inductor"]["move_reshape_out_of_split_stack_pass"],
                expected_move_reshape_out_of_split_stack,
            )
            counters.clear()

    def test_numpy_compat_normalization(self):
        def fn(x, y):
            a = torch.stack([x, y], axis=1)
            b = torch.mul(x, x2=y)
            c = torch.mul(x, x2=y)
            d = torch.mul(x, x2=y)
            e = torch.max(x, dim=1, keepdims=True)
            f = torch.dropout(x=x, p=0.5, train=True)
            return a, b, c, d, e, f

        fn_t = torch.fx.symbolic_trace(fn)
        numpy_compat_normalization(fn_t.graph)

        for n in fn_t.graph.nodes:
            for k in n.kwargs.keys():
                self.assertTrue(k not in {"x", "x1", "x2", "a", "axis", "keepdims"})

    @patch
    @requires_gpu
    def test_stack_normalization_axis_kwarg(self):
        def fn(x, y):
            return torch.stack([x, y], axis=1)

        x, y = (torch.rand((4, 4), device=GPU_TYPE) for _ in range(2))
        expected = fn(x, y)
        actual = torch.compile(fn)(x, y)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
