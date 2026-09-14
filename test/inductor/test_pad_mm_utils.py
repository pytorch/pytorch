# Owner(s): ["module: inductor"]
import operator
from unittest import mock

import torch
from torch._inductor import config as inductor_config
from torch._inductor.fx_passes.pad_mm import (
    _padding_bench_fn,
    _padding_plan_result_decoder_factory,
    _padding_plan_result_encoder_factory,
    _select_padding_plan_uncached,
    _selected_padding_plan,
    addmm_replace,
    bmm_replace,
    FORCE_PADDING,
    get_non_view_def,
    get_normal_padding_plans,
    get_padding_lengths,
    K_N_PADDING,
    K_PADDING,
    M_N_PADDING,
    M_PADDING,
    mm_replace,
    N_PADDING,
    NO_PADDING,
    pad_addmm,
    pad_bmm,
    pad_mm,
    should_pad,
    should_pad_addmm,
    should_pad_bench_key,
)
from torch._inductor.runtime.caching import encoders
from torch._inductor.test_case import run_tests, TestCase


class PadMMUtilsTest(TestCase):
    def test_get_non_view_def_traverses_getitem(self):
        graph = torch.fx.Graph()
        arg = graph.placeholder("arg")
        sort = graph.call_function(torch.ops.aten.sort.default, (arg,))
        getitem = graph.call_function(operator.getitem, (sort, 0))

        self.assertIs(get_non_view_def(getitem), sort)

    def test_normal_plan_family_for_every_dimension_mask(self):
        expected = {
            (False, False, False): (NO_PADDING,),
            (False, False, True): (NO_PADDING, N_PADDING),
            (False, True, False): (NO_PADDING, K_PADDING),
            (False, True, True): (
                NO_PADDING,
                K_PADDING,
                N_PADDING,
                K_N_PADDING,
            ),
            (True, False, False): (NO_PADDING,),
            (True, False, True): (NO_PADDING, N_PADDING),
            (True, True, False): (NO_PADDING, K_PADDING),
            (True, True, True): (
                NO_PADDING,
                K_PADDING,
                N_PADDING,
                K_N_PADDING,
            ),
        }
        for mask, plans in expected.items():
            m = 17 if mask[0] else 16
            k = 17 if mask[1] else 16
            n = 17 if mask[2] else 16
            mat1 = torch.randn(m, k)
            mat2 = torch.randn(k, n)
            actual = get_normal_padding_plans(mat1, mat2, torch.ops.aten.mm)
            self.assertEqual(actual, plans, msg=f"mask={mask}")
            self.assertLessEqual(len(actual), 4)
            self.assertFalse(any(plan.pad_m for plan in actual))

    def test_m_plan_requires_stride_repair_without_k_padding(self):
        dtype = torch.bfloat16
        mat2_n_tail = torch.empty(4096, 1157, dtype=dtype)

        # Production layout: M padding materializes an A whose column-major
        # leading stride is not 16-byte aligned.
        transposed = torch.empty(4096, 1157, dtype=dtype).t()
        self.assertEqual(
            get_normal_padding_plans(transposed, mat2_n_tail, torch.ops.aten.mm),
            (NO_PADDING, N_PADDING, M_N_PADDING),
        )

        # Contiguous A does not need a layout-repair candidate.
        self.assertEqual(
            get_normal_padding_plans(
                torch.empty(1157, 4096, dtype=dtype),
                mat2_n_tail,
                torch.ops.aten.mm,
            ),
            (NO_PADDING, N_PADDING),
        )

        # K padding already materializes A, so adding M would duplicate that
        # layout repair and grow the family beyond four choices.
        k_tail_a = torch.empty(4097, 1157, dtype=dtype).t()
        self.assertEqual(
            get_normal_padding_plans(
                k_tail_a,
                torch.empty(4097, 1157, dtype=dtype),
                torch.ops.aten.mm,
            ),
            (NO_PADDING, K_PADDING, N_PADDING, K_N_PADDING),
        )

        # The physical column-major leading stride is already aligned.
        aligned_storage = torch.empty(4096, 1160, dtype=dtype)
        aligned_leading_stride = aligned_storage[:, :1157].t()
        self.assertEqual(aligned_leading_stride.stride(), (1, 1160))
        self.assertEqual(
            get_normal_padding_plans(
                aligned_leading_stride, mat2_n_tail, torch.ops.aten.mm
            ),
            (NO_PADDING, N_PADDING),
        )

        # The same matrix-layout predicate applies to the final two BMM dims.
        bmm_a = torch.empty(2, 128, 290, dtype=dtype).transpose(1, 2)
        bmm_b = torch.empty(2, 128, 128, dtype=dtype)
        self.assertEqual(bmm_a.stride(), (37120, 1, 290))
        self.assertEqual(
            get_normal_padding_plans(bmm_a, bmm_b, torch.ops.aten.bmm),
            (NO_PADDING, M_PADDING),
        )

    def test_dimension_specific_plan_correctness(self):
        cases = (
            (torch.ops.aten.mm, torch.randn(17, 33), torch.randn(33, 65)),
            (
                torch.ops.aten.bmm,
                torch.randn(3, 17, 33),
                torch.randn(3, 33, 65),
            ),
        )
        for op, mat1, mat2 in cases:
            expected = op(mat1, mat2)
            for plan in (
                M_PADDING,
                K_PADDING,
                N_PADDING,
                M_N_PADDING,
                K_N_PADDING,
                FORCE_PADDING,
            ):
                lengths = get_padding_lengths(mat1, mat2, op, plan)
                actual = (
                    pad_bmm(mat1, mat2, *lengths)
                    if op is torch.ops.aten.bmm
                    else pad_mm(mat1, mat2, *lengths)
                )
                torch.testing.assert_close(actual, expected)

        bias = torch.randn(65)
        mat1 = torch.randn(17, 33)
        mat2 = torch.randn(33, 65)
        expected = torch.addmm(bias, mat1, mat2)
        for plan in (
            M_PADDING,
            K_PADDING,
            N_PADDING,
            M_N_PADDING,
            K_N_PADDING,
            FORCE_PADDING,
        ):
            lengths = get_padding_lengths(mat1, mat2, torch.ops.aten.addmm, plan)
            torch.testing.assert_close(pad_addmm(bias, mat1, mat2, *lengths), expected)

    def test_selected_plan_is_carried_into_replacement(self):
        mat1 = torch.randn(17, 33)
        mat2 = torch.randn(33, 65)
        for plan, expected_lengths in (
            (M_PADDING, (3, 0, 0)),
            (K_PADDING, (0, 3, 0)),
            (N_PADDING, (0, 0, 3)),
            (M_N_PADDING, (3, 0, 3)),
            (K_N_PADDING, (0, 3, 3)),
        ):
            token = _selected_padding_plan.set(plan)
            try:
                with mock.patch(
                    "torch._inductor.fx_passes.pad_mm.pad_mm",
                    return_value=torch.empty(17, 65),
                ) as mocked_pad_mm:
                    mm_replace(mat1, mat2)
                    self.assertIsNone(_selected_padding_plan.get())
            finally:
                _selected_padding_plan.reset(token)
            self.assertEqual(mocked_pad_mm.call_args.args[2:5], expected_lengths)

        _selected_padding_plan.set(None)
        with self.assertRaisesRegex(AssertionError, "without a selected plan"):
            mm_replace(mat1, mat2)

    def test_addmm_benchmark_preserves_scalars_and_keys_them(self):
        mat1 = torch.randn(17, 33)
        mat2 = torch.randn(33, 65)
        bias = torch.randn(65)
        match = mock.MagicMock()
        match.kwargs = {"alpha": 0.25, "beta": 0.5}
        with mock.patch(
            "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time",
            return_value=False,
        ):
            actual = _padding_bench_fn(
                match,
                mat1,
                mat2,
                torch.ops.aten.addmm,
                K_N_PADDING,
                bias,
            )()
        torch.testing.assert_close(
            actual, torch.addmm(bias, mat1, mat2, beta=0.5, alpha=0.25)
        )

        with mock.patch(
            "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time",
            return_value=False,
        ):
            first_key = should_pad_bench_key(
                match, mat1, mat2, torch.ops.aten.addmm, bias
            )
            first_encoded = encoders.should_pad_params_encoder(
                match, mat1, mat2, torch.ops.aten.addmm, bias
            )
            match.kwargs = {"alpha": 0.5, "beta": 0.5}
            second_key = should_pad_bench_key(
                match, mat1, mat2, torch.ops.aten.addmm, bias
            )
            second_encoded = encoders.should_pad_params_encoder(
                match, mat1, mat2, torch.ops.aten.addmm, bias
            )
        self.assertNotEqual(first_key, second_key)
        self.assertNotEqual(first_encoded, second_encoded)

    def test_padding_cache_keys_include_device_identity_and_v4(self):
        mat1 = torch.randn(16, 17)
        mat2 = torch.randn(17, 16)
        match = mock.MagicMock()
        match.kwargs = {}
        with (
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time",
                return_value=False,
            ),
            mock.patch(
                "torch._inductor.runtime.caching.encoders.get_device_identity",
                return_value=("cuda:0", "B200", "sm_100"),
            ),
        ):
            first_key = should_pad_bench_key(match, mat1, mat2, torch.ops.aten.mm)
            first_encoded = encoders.should_pad_params_encoder(
                match, mat1, mat2, torch.ops.aten.mm
            )
        with (
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time",
                return_value=False,
            ),
            mock.patch(
                "torch._inductor.runtime.caching.encoders.get_device_identity",
                return_value=("cuda:0", "H100", "sm_90"),
            ),
        ):
            second_key = should_pad_bench_key(match, mat1, mat2, torch.ops.aten.mm)
            second_encoded = encoders.should_pad_params_encoder(
                match, mat1, mat2, torch.ops.aten.mm
            )

        self.assertNotEqual(first_key, second_key)
        self.assertNotEqual(first_encoded, second_encoded)
        self.assertIn("padding_plan_v4", first_key)
        self.assertEqual(first_encoded["padding_plan_version"], 4)

    @inductor_config.patch(force_shape_pad=True)
    def test_force_padding_bypasses_plan_benchmark(self):
        match = mock.MagicMock()
        match.output_node.return_value.meta = {}
        token = _selected_padding_plan.set(None)
        try:
            with (
                mock.patch(
                    "torch._inductor.fx_passes.pad_mm.can_pad", return_value=True
                ),
                mock.patch("torch._inductor.fx_passes.pad_mm._should_pad") as select,
            ):
                self.assertTrue(
                    should_pad(
                        match,
                        torch.randn(17, 33),
                        torch.randn(33, 65),
                        torch.ops.aten.mm,
                    )
                )
            select.assert_not_called()
            self.assertEqual(_selected_padding_plan.get(), FORCE_PADDING)
        finally:
            _selected_padding_plan.reset(token)

    def test_plan_benchmark_selects_only_profitable_winner(self):
        match = mock.MagicMock()

        def select(op, mat1, mat2, times, input=None):
            times = iter(times)
            with (
                mock.patch(
                    "torch._inductor.fx_passes.pad_mm.get_do_bench",
                    return_value=lambda fn: next(times),
                ),
                mock.patch(
                    "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound",
                    return_value=True,
                ),
                mock.patch(
                    "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time",
                    return_value=False,
                ),
                mock.patch(
                    "torch._inductor.fx_passes.pad_mm.get_cached_base_mm_benchmark_time",
                    return_value=None,
                ),
                mock.patch(
                    "torch._inductor.fx_passes.pad_mm.get_cached_padding_plan",
                    return_value=None,
                ),
                mock.patch(
                    "torch._inductor.fx_passes.pad_mm.set_cached_base_mm_benchmark_time"
                ),
                mock.patch("torch._inductor.fx_passes.pad_mm.set_cached_padding_plan"),
                mock.patch(
                    "torch._inductor.fx_passes.pad_mm._should_run_pad_autoheuristic",
                    return_value=False,
                ),
            ):
                return _select_padding_plan_uncached(match, mat1, mat2, op, input)

        self.assertEqual(
            select(
                torch.ops.aten.mm,
                torch.randn(128, 33),
                torch.randn(33, 64),
                (10.0, 5.0),
            ),
            K_PADDING,
        )
        self.assertEqual(
            select(
                torch.ops.aten.addmm,
                torch.randn(128, 32),
                torch.randn(32, 65),
                (10.0, 5.0),
                torch.randn(65),
            ),
            N_PADDING,
        )
        # Timing order is none, K-only, N-only, K+N.
        self.assertEqual(
            select(
                torch.ops.aten.bmm,
                torch.randn(2, 128, 33),
                torch.randn(2, 33, 65),
                (10.0, 8.0, 7.0, 6.0),
            ),
            K_N_PADDING,
        )
        self.assertEqual(
            select(
                torch.ops.aten.mm,
                torch.randn(129, 33),
                torch.randn(33, 65),
                (10.0, 9.5, 9.3, 9.2, 9.1),
            ),
            NO_PADDING,
        )

        # M+N legacy winner from the transposed-A counterexample.
        self.assertEqual(
            select(
                torch.ops.aten.mm,
                torch.randn(32, 17).t(),
                torch.randn(32, 65),
                (10.0, 9.0, 5.0),
            ),
            M_N_PADDING,
        )
        # The non-M N plan wins for the contiguous control.
        self.assertEqual(
            select(
                torch.ops.aten.mm,
                torch.randn(17, 32),
                torch.randn(32, 65),
                (10.0, 5.0, 6.0),
            ),
            N_PADDING,
        )
        # Exact ties prefer the earlier, less invasive non-M plan over M+N.
        self.assertEqual(
            select(
                torch.ops.aten.mm,
                torch.randn(32, 17).t(),
                torch.randn(32, 65),
                (10.0, 5.0, 5.0),
            ),
            N_PADDING,
        )
        # M-only is not offered for an ordinary contiguous operand.
        self.assertEqual(
            select(
                torch.ops.aten.mm,
                torch.randn(17, 32),
                torch.randn(32, 64),
                (),
            ),
            NO_PADDING,
        )

    def test_padding_plan_result_codec(self):
        encode = _padding_plan_result_encoder_factory(lambda: NO_PADDING)()
        decode = _padding_plan_result_decoder_factory(lambda: NO_PADDING)()
        for plan in (
            NO_PADDING,
            M_PADDING,
            K_PADDING,
            N_PADDING,
            M_N_PADDING,
            K_N_PADDING,
            FORCE_PADDING,
        ):
            self.assertEqual(decode(encode(plan)), plan)
        self.assertEqual(encode(M_PADDING), "m")
        self.assertEqual(encode(M_N_PADDING), "m+n")
        self.assertEqual(encode(FORCE_PADDING), "legacy-all")
        self.assertEqual(decode(True), NO_PADDING)
        self.assertEqual(decode("unknown"), NO_PADDING)

    @inductor_config.patch(post_grad_fusion_options={"pad_aten_mm_pass": {}})
    def test_bf16_large_k_override_preserves_legacy_all_plan(self):
        match = mock.MagicMock()
        with mock.patch(
            "torch._inductor.fx_passes.pad_mm.should_pad_mm_bf16",
            return_value=True,
        ):
            self.assertEqual(
                _select_padding_plan_uncached(
                    match,
                    torch.randn(17, 32),
                    torch.randn(32, 65),
                    torch.ops.aten.mm,
                ),
                FORCE_PADDING,
            )
            self.assertEqual(
                _select_padding_plan_uncached(
                    match,
                    torch.randn(16, 32),
                    torch.randn(32, 65),
                    torch.ops.aten.mm,
                ),
                N_PADDING,
            )

    def test_autoheuristic_defers_to_benchmark_for_multiple_candidates(self):
        times = iter((10.0, 8.0, 7.0))
        with (
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_do_bench",
                return_value=lambda fn: next(times),
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound",
                return_value=True,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time",
                return_value=False,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_base_mm_benchmark_time",
                return_value=None,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_padding_plan",
                return_value=None,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.set_cached_base_mm_benchmark_time"
            ),
            mock.patch("torch._inductor.fx_passes.pad_mm.set_cached_padding_plan"),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm._should_run_pad_autoheuristic",
                return_value=True,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.run_autoheuristic"
            ) as autoheuristic,
        ):
            plan = _select_padding_plan_uncached(
                mock.MagicMock(),
                torch.randn(32, 17).t(),
                torch.randn(32, 65),
                torch.ops.aten.mm,
            )
        self.assertEqual(plan, M_N_PADDING)
        autoheuristic.assert_not_called()

    @inductor_config.patch(deterministic=True)
    def test_deterministic_autoheuristic_uses_full_non_m_plan(self):
        with (
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound",
                return_value=True,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time",
                return_value=False,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_base_mm_benchmark_time",
                return_value=None,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_padding_plan",
                return_value=None,
            ),
            mock.patch("torch._inductor.fx_passes.pad_mm.set_cached_padding_plan"),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm._should_run_pad_autoheuristic",
                return_value=True,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.run_autoheuristic",
                return_value=True,
            ) as autoheuristic,
        ):
            plan = _select_padding_plan_uncached(
                mock.MagicMock(),
                torch.randn(17, 33),
                torch.randn(33, 65),
                torch.ops.aten.mm,
            )
        self.assertEqual(plan, K_N_PADDING)
        self.assertEqual(autoheuristic.call_args.args[4:7], (0, 3, 3))

    def test_multi_site_handoff_and_early_rejection_cleanup(self):
        match = mock.MagicMock()
        match.output_node.return_value.meta = {}
        mat1 = torch.randn(17, 33)
        mat2 = torch.randn(33, 65)
        bmat1 = torch.randn(2, 17, 33)
        bmat2 = torch.randn(2, 33, 65)
        bias = torch.randn(65)

        with (
            mock.patch("torch._inductor.fx_passes.pad_mm.can_pad", return_value=True),
            mock.patch(
                "torch._inductor.kernel.mm_common._use_small_mm_pointwise",
                return_value=False,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm._should_pad",
                side_effect=(K_PADDING, N_PADDING, FORCE_PADDING),
            ),
        ):
            self.assertTrue(should_pad(match, mat1, mat2, torch.ops.aten.mm))
            self.assertEqual(mm_replace(mat1, mat2), mat1 @ mat2)
            self.assertIsNone(_selected_padding_plan.get())

            self.assertTrue(should_pad(match, bmat1, bmat2, torch.ops.aten.bmm))
            self.assertEqual(bmm_replace(bmat1, bmat2), bmat1 @ bmat2)
            self.assertIsNone(_selected_padding_plan.get())

            self.assertTrue(
                should_pad(
                    match,
                    mat1,
                    mat2,
                    torch.ops.aten.addmm,
                    input=bias,
                )
            )
            self.assertEqual(
                addmm_replace(bias, mat1, mat2), torch.addmm(bias, mat1, mat2)
            )
            self.assertIsNone(_selected_padding_plan.get())

        rejected = mock.MagicMock()
        rejected_input = mock.MagicMock()
        rejected_input.shape = torch.Size((2, 2, 2))
        rejected_input.is_cuda = True
        rejected.kwargs = {
            "mat1": mock.MagicMock(meta={"val": mat1}),
            "mat2": mock.MagicMock(meta={"val": mat2}),
            "input": mock.MagicMock(meta={"val": rejected_input}),
            "beta": 0,
        }
        _selected_padding_plan.set(FORCE_PADDING)
        self.assertFalse(should_pad_addmm(rejected))
        self.assertIsNone(_selected_padding_plan.get())

    def test_failed_selection_clears_stale_handoff(self):
        match = mock.MagicMock()
        match.output_node.return_value.meta = {}
        _selected_padding_plan.set(FORCE_PADDING)
        with (
            mock.patch("torch._inductor.fx_passes.pad_mm.can_pad", return_value=True),
            mock.patch(
                "torch._inductor.kernel.mm_common._use_small_mm_pointwise",
                return_value=False,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm._should_pad",
                side_effect=RuntimeError("injected failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                should_pad(
                    match,
                    torch.randn(17, 33),
                    torch.randn(33, 65),
                    torch.ops.aten.mm,
                )
        self.assertIsNone(_selected_padding_plan.get())

    def test_padding_plan_cache_hit_skips_benchmark(self):
        with (
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound",
                return_value=True,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_padding_plan",
                return_value=N_PADDING,
            ),
            mock.patch(
                "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time",
                return_value=False,
            ),
            mock.patch("torch._inductor.fx_passes.pad_mm.get_do_bench") as get_do_bench,
        ):
            plan = _select_padding_plan_uncached(
                mock.MagicMock(),
                torch.randn(128, 32),
                torch.randn(32, 65),
                torch.ops.aten.mm,
            )
        self.assertEqual(plan, N_PADDING)
        get_do_bench.assert_not_called()


if __name__ == "__main__":
    run_tests()
