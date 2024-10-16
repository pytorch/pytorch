# Owner(s): ["module: functorch"]
import typing
import unittest

from torch._C import (
    _dispatch_get_registrations_for_dispatch_key as get_registrations_for_dispatch_key,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


xfail_functorch_batched = {
    "aten::is_nonzero",
    "aten::item",
    "aten::linalg_slogdet",
    "aten::masked_select_backward",
    "aten::one_hot",
    "aten::silu_backward",
    "aten::where",
}

xfail_functorch_batched_decomposition = {
    "aten::alias_copy",
    "aten::as_strided_copy",
    "aten::diagonal_copy",
    "aten::is_same_size",
    "aten::unfold_copy",
}

xfail_not_implemented = {
    "aten::affine_grid_generator_backward",
    "aten::align_as",
    "aten::align_tensors",
    "aten::align_to",
    "aten::align_to.ellipsis_idx",
    "aten::alpha_dropout",
    "aten::alpha_dropout_",
    "aten::argwhere",
    "aten::bilinear",
    "aten::can_cast",
    "aten::cat.names",
    "aten::chain_matmul",
    "aten::chalf",
    "aten::choose_qparams_optimized",
    "aten::clip_",
    "aten::clip_.Tensor",
    "aten::coalesce",
    "aten::column_stack",
    "aten::concat.names",
    "aten::concatenate.names",
    "aten::conj",
    "aten::conv_tbc_backward",
    "aten::ctc_loss.IntList",
    "aten::ctc_loss.Tensor",
    "aten::cudnn_is_acceptable",
    "aten::cummaxmin_backward",
    "aten::data",
    "aten::diagflat",
    "aten::divide.out_mode",
    "aten::divide_.Scalar",
    "aten::dropout_",
    "aten::embedding_bag",
    "aten::embedding_bag.padding_idx",
    "aten::feature_alpha_dropout",
    "aten::feature_alpha_dropout_",
    "aten::feature_dropout",
    "aten::feature_dropout_",
    "aten::fft_ihfft2",
    "aten::fft_ihfftn",
    "aten::fill_diagonal_",
    "aten::fix_",
    "aten::flatten.named_out_dim",
    "aten::flatten.using_names",
    "aten::flatten_dense_tensors",
    "aten::float_power_.Scalar",
    "aten::float_power_.Tensor",
    "aten::floor_divide_.Scalar",
    "aten::frobenius_norm",
    "aten::fused_moving_avg_obs_fake_quant",
    "aten::get_gradients",
    "aten::greater_.Scalar",
    "aten::greater_.Tensor",
    "aten::greater_equal_.Scalar",
    "aten::greater_equal_.Tensor",
    "aten::gru.data",
    "aten::gru.input",
    "aten::gru_cell",
    "aten::histogramdd",
    "aten::histogramdd.TensorList_bins",
    "aten::histogramdd.int_bins",
    "aten::infinitely_differentiable_gelu_backward",
    "aten::isclose",
    "aten::istft",
    "aten::item",
    "aten::kl_div",
    "aten::ldexp_",
    "aten::less_.Scalar",
    "aten::less_.Tensor",
    "aten::less_equal_.Scalar",
    "aten::less_equal_.Tensor",
    "aten::linalg_cond.p_str",
    "aten::linalg_eigh.eigvals",
    "aten::linalg_matrix_rank",
    "aten::linalg_matrix_rank.out_tol_tensor",
    "aten::linalg_matrix_rank.tol_tensor",
    "aten::linalg_pinv.out_rcond_tensor",
    "aten::linalg_pinv.rcond_tensor",
    "aten::linalg_slogdet",
    "aten::linalg_svd.U",
    "aten::linalg_tensorsolve",
    "aten::logsumexp.names",
    "aten::lstm.data",
    "aten::lstm.input",
    "aten::lstm_cell",
    "aten::lu_solve",
    "aten::margin_ranking_loss",
    "aten::masked_select_backward",
    "aten::matrix_exp",
    "aten::matrix_exp_backward",
    "aten::max.names_dim",
    "aten::max.names_dim_max",
    "aten::mean.names_dim",
    "aten::median.names_dim",
    "aten::median.names_dim_values",
    "aten::min.names_dim",
    "aten::min.names_dim_min",
    "aten::mish_backward",
    "aten::moveaxis.int",
    "aten::multilabel_margin_loss",
    "aten::nanmedian.names_dim",
    "aten::nanmedian.names_dim_values",
    "aten::nanquantile",
    "aten::nanquantile.scalar",
    "aten::narrow.Tensor",
    "aten::native_channel_shuffle",
    "aten::negative_",
    "aten::nested_to_padded_tensor",
    "aten::nonzero_numpy",
    "aten::norm.names_ScalarOpt_dim",
    "aten::norm.names_ScalarOpt_dim_dtype",
    "aten::norm_except_dim",
    "aten::not_equal_.Scalar",
    "aten::not_equal_.Tensor",
    "aten::one_hot",
    "aten::output_nr",
    "aten::pad_sequence",
    "aten::pdist",
    "aten::pin_memory",
    "aten::promote_types",
    "aten::qr.Q",
    "aten::quantile",
    "aten::quantile.scalar",
    "aten::refine_names",
    "aten::rename",
    "aten::rename_",
    "aten::requires_grad_",
    "aten::retain_grad",
    "aten::retains_grad",
    "aten::rnn_relu.data",
    "aten::rnn_relu.input",
    "aten::rnn_relu_cell",
    "aten::rnn_tanh.data",
    "aten::rnn_tanh.input",
    "aten::rnn_tanh_cell",
    "aten::set_.source_Tensor_storage_offset",
    "aten::set_data",
    "aten::silu_backward",
    "aten::slow_conv3d",
    "aten::smm",
    "aten::special_chebyshev_polynomial_t.n_scalar",
    "aten::special_chebyshev_polynomial_t.x_scalar",
    "aten::special_chebyshev_polynomial_u.n_scalar",
    "aten::special_chebyshev_polynomial_u.x_scalar",
    "aten::special_chebyshev_polynomial_v.n_scalar",
    "aten::special_chebyshev_polynomial_v.x_scalar",
    "aten::special_chebyshev_polynomial_w.n_scalar",
    "aten::special_chebyshev_polynomial_w.x_scalar",
    "aten::special_hermite_polynomial_h.n_scalar",
    "aten::special_hermite_polynomial_h.x_scalar",
    "aten::special_hermite_polynomial_he.n_scalar",
    "aten::special_hermite_polynomial_he.x_scalar",
    "aten::special_laguerre_polynomial_l.n_scalar",
    "aten::special_laguerre_polynomial_l.x_scalar",
    "aten::special_legendre_polynomial_p.n_scalar",
    "aten::special_legendre_polynomial_p.x_scalar",
    "aten::special_shifted_chebyshev_polynomial_t.n_scalar",
    "aten::special_shifted_chebyshev_polynomial_t.x_scalar",
    "aten::special_shifted_chebyshev_polynomial_u.n_scalar",
    "aten::special_shifted_chebyshev_polynomial_u.x_scalar",
    "aten::special_shifted_chebyshev_polynomial_v.n_scalar",
    "aten::special_shifted_chebyshev_polynomial_v.x_scalar",
    "aten::special_shifted_chebyshev_polynomial_w.n_scalar",
    "aten::special_shifted_chebyshev_polynomial_w.x_scalar",
    "aten::square_",
    "aten::sspaddmm",
    "aten::std.correction_names",
    "aten::std.names_dim",
    "aten::std_mean.correction_names",
    "aten::std_mean.names_dim",
    "aten::stft",
    "aten::stft.center",
    "aten::stride.int",
    "aten::subtract.Scalar",
    "aten::subtract_.Scalar",
    "aten::subtract_.Tensor",
    "aten::svd.U",
    "aten::sym_size.int",
    "aten::sym_stride.int",
    "aten::sym_numel",
    "aten::sym_storage_offset",
    "aten::tensor_split.tensor_indices_or_sections",
    "aten::thnn_conv2d",
    "aten::to_dense",
    "aten::to_dense_backward",
    "aten::to_mkldnn_backward",
    "aten::trace_backward",
    "aten::triplet_margin_loss",
    "aten::unflatten_dense_tensors",
    "aten::vander",
    "aten::var.correction_names",
    "aten::var.names_dim",
    "aten::var_mean.correction_names",
    "aten::var_mean.names_dim",
    "aten::where",
    "aten::wrapped_linear_prepack",
    "aten::wrapped_quantized_linear_prepacked",
}


def dispatch_registrations(
    dispatch_key: str, xfails: set, filter_func: typing.Callable = lambda reg: True
):
    registrations = sorted(get_registrations_for_dispatch_key(dispatch_key))
    subtests = [
        subtest(
            reg,
            name=f"[{reg}]",
            decorators=([unittest.expectedFailure] if reg in xfails else []),
        )
        for reg in registrations
        if filter_func(reg)
    ]
    return parametrize("registration", subtests)


CompositeImplicitAutogradRegistrations = set(
    get_registrations_for_dispatch_key("CompositeImplicitAutograd")
)
FuncTorchBatchedRegistrations = set(
    get_registrations_for_dispatch_key("FuncTorchBatched")
)
FuncTorchBatchedDecompositionRegistrations = set(
    get_registrations_for_dispatch_key("FuncTorchBatchedDecomposition")
)


def filter_vmap_implementable(reg):
    reg = reg.lower()
    if not reg.startswith("aten::"):
        return False
    if reg.startswith("aten::_"):
        return False
    if reg.endswith(".out"):
        return False
    if reg.endswith("_out"):
        return False
    if ".dimname" in reg:
        return False
    if "_dimname" in reg:
        return False
    if "fbgemm" in reg:
        return False
    if "quantize" in reg:
        return False
    if "sparse" in reg:
        return False
    if "::is_" in reg:
        return False
    return True


class TestFunctorchDispatcher(TestCase):
    @dispatch_registrations("CompositeImplicitAutograd", xfail_functorch_batched)
    def test_register_a_batching_rule_for_composite_implicit_autograd(
        self, registration
    ):
        assert registration not in FuncTorchBatchedRegistrations, (
            f"You've added a batching rule for a CompositeImplicitAutograd operator {registration}. "
            "The correct way to add vmap support for it is to put it into BatchRulesDecomposition to "
            "reuse the CompositeImplicitAutograd decomposition"
        )

    @dispatch_registrations(
        "FuncTorchBatchedDecomposition", xfail_functorch_batched_decomposition
    )
    def test_register_functorch_batched_decomposition(self, registration):
        assert registration in CompositeImplicitAutogradRegistrations, (
            f"The registrations in BatchedDecompositions.cpp must be for CompositeImplicitAutograd "
            f"operations. If your operation {registration} is not CompositeImplicitAutograd, "
            "then please register it to the FuncTorchBatched key in another file."
        )

    @dispatch_registrations(
        "CompositeImplicitAutograd", xfail_not_implemented, filter_vmap_implementable
    )
    def test_unimplemented_batched_registrations(self, registration):
        assert registration in FuncTorchBatchedDecompositionRegistrations, (
            f"Please check that there is an OpInfo that covers the operator {registration} "
            "and add a registration in BatchedDecompositions.cpp. "
            "If your operator isn't user facing, please add it to the xfail list"
        )


instantiate_parametrized_tests(TestFunctorchDispatcher)

if __name__ == "__main__":
    run_tests()
