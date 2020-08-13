# Generated for selective build without using static dispatch.
# Manually run the script to update (under fbsource):
# xplat/caffe2/fb/update_pt_deps.sh \
# --fbcode --base_ops xplat/caffe2/fb/pt_mobile_base_ops.txt
TORCH_DEPS = {
    "__BASE__": [
        "aten::_coalesced_",
        "aten::_copy_from",
        "aten::_empty_affine_quantized",
        "aten::_empty_per_channel_affine_quantized",
        "aten::_indices",
        "aten::_nnz",
        "aten::_values",
        "aten::add",
        "aten::add_",
        "aten::arange",
        "aten::as_strided",
        "aten::as_strided_",
        "aten::cat",
        "aten::clone",
        "aten::coalesce",
        "aten::contiguous",
        "aten::copy_",
        "aten::copy_sparse_to_sparse_",
        "aten::dense_dim",
        "aten::dequantize",
        "aten::div",
        "aten::div_",
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::eq",
        "aten::equal",
        "aten::expand",
        "aten::fill_",
        "aten::is_coalesced",
        "aten::is_complex",
        "aten::is_floating_point",
        "aten::is_leaf",
        "aten::is_nonzero",
        "aten::item",
        "aten::max",
        "aten::min",
        "aten::mul",
        "aten::mul_",
        "aten::narrow",
        "aten::ne",
        "aten::permute",
        "aten::q_per_channel_axis",
        "aten::q_per_channel_scales",
        "aten::q_per_channel_zero_points",
        "aten::q_scale",
        "aten::q_zero_point",
        "aten::qscheme",
        "aten::quantize_per_tensor",
        "aten::reshape",
        "aten::resize_",
        "aten::resize_as_",
        "aten::scalar_tensor",
        "aten::select",
        "aten::set_",
        "aten::set_quantizer_",
        "aten::size",
        "aten::slice",
        "aten::sparse_dim",
        "aten::sparse_resize_and_clear_",
        "aten::squeeze",
        "aten::squeeze_",
        "aten::stride",
        "aten::sub",
        "aten::sub_",
        "aten::sum",
        "aten::t",
        "aten::to",
        "aten::unsqueeze",
        "aten::view",
        "aten::zero_",
        "aten::zeros",
        "aten::zeros_like",
    ],
    "__ROOT__": [
        "aten::_sparse_coo_tensor_unsafe",
        "aten::_version",
        "aten::any",
        "aten::chunk",
        "aten::detach",
        "aten::isnan",
        "aten::lt",
        "aten::mm",
        "aten::ones_like",
        "aten::output_nr",
        "aten::rsqrt",
        "aten::set_data",
    ],
    "_test::leaky_relu": [
        "aten::leaky_relu",
    ],
    "aten::__and__": [
        "aten::bitwise_and",
    ],
    "aten::__iand__": [
        "aten::bitwise_and_",
    ],
    "aten::__ior__": [
        "aten::bitwise_or_",
    ],
    "aten::__ixor__": [
        "aten::bitwise_xor_",
    ],
    "aten::__or__": [
        "aten::bitwise_or",
    ],
    "aten::__xor__": [
        "aten::bitwise_xor",
    ],
    "aten::_batch_norm_impl_index": [
        "aten::cudnn_batch_norm",
        "aten::miopen_batch_norm",
        "aten::native_batch_norm",
    ],
    "aten::_batch_norm_impl_index_backward": [
        "aten::cudnn_batch_norm_backward",
        "aten::miopen_batch_norm_backward",
        "aten::native_batch_norm_backward",
    ],
    "aten::_cdist_forward": [
        "aten::_euclidean_dist",
    ],
    "aten::_cholesky_helper": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::_cholesky_solve_helper": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::_convolution": [
        "aten::_convolution_nogroup",
        "aten::_unsafe_view",
        "aten::convolution_overrideable",
        "aten::cudnn_convolution",
        "aten::cudnn_convolution_transpose",
        "aten::miopen_convolution",
        "aten::miopen_convolution_transpose",
        "aten::miopen_depthwise_convolution",
        "aten::slow_conv3d",
        "aten::thnn_conv_depthwise2d",
    ],
    "aten::_convolution_double_backward": [
        "aten::_convolution",
        "aten::transpose",
    ],
    "aten::_convolution_nogroup": [
        "aten::_nnpack_available",
        "aten::_nnpack_spatial_convolution",
        "aten::slow_conv3d",
        "aten::slow_conv_dilated2d",
        "aten::slow_conv_dilated3d",
        "aten::slow_conv_transpose2d",
        "aten::slow_conv_transpose3d",
        "aten::thnn_conv2d",
    ],
    "aten::_ctc_loss_backward": [
        "aten::full_like",
    ],
    "aten::_embedding_bag": [
        "aten::cumsum",
        "aten::expand_as",
        "aten::index_add_",
        "aten::ones_like",
    ],
    "aten::_embedding_bag_backward": [
        "aten::_embedding_bag_dense_backward",
        "aten::_embedding_bag_sparse_backward",
        "aten::cumsum",
        "aten::index_add_",
        "aten::ones_like",
    ],
    "aten::_embedding_bag_dense_backward": [
        "aten::index_add_",
        "aten::index_select",
        "aten::nonzero",
        "aten::sort",
    ],
    "aten::_embedding_bag_forward_only": [
        "aten::cumsum",
        "aten::expand_as",
        "aten::index_add_",
        "aten::ones_like",
    ],
    "aten::_embedding_bag_per_sample_weights_backward": [
        "aten::cumsum",
        "aten::index_add_",
        "aten::ones_like",
    ],
    "aten::_embedding_bag_sparse_backward": [
        "aten::embedding_dense_backward",
        "aten::embedding_sparse_backward",
        "aten::index_select",
    ],
    "aten::_euclidean_dist": [
        "aten::clamp_min_",
        "aten::matmul",
        "aten::ones_like",
        "aten::pow",
        "aten::sqrt_",
        "aten::transpose",
    ],
    "aten::_fake_quantize_learnable_per_channel_affine_backward": [
        "aten::clamp",
        "aten::unbind",
    ],
    "aten::_gather_sparse_backward": [
        "aten::_sparse_coo_tensor_unsafe",
        "aten::repeat",
    ],
    "aten::_index_put_impl_": [
        "aten::nonzero",
    ],
    "aten::_inverse_helper": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::_lu_solve_helper": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::_lu_with_info": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::_nnpack_spatial_convolution_backward": [
        "aten::_nnpack_spatial_convolution_backward_input",
        "aten::_nnpack_spatial_convolution_backward_weight",
    ],
    "aten::_pack_padded_sequence": [
        "aten::transpose",
    ],
    "aten::_pack_padded_sequence_backward": [
        "aten::transpose",
    ],
    "aten::_pad_packed_sequence": [
        "aten::full",
        "aten::transpose",
    ],
    "aten::_qr_helper": [
        "aten::expand_as",
        "aten::eye",
    ],
    "aten::_sobol_engine_initialize_state_": [
        "aten::pow",
    ],
    "aten::_sobol_engine_scramble_": [
        "aten::diagonal",
        "aten::expand_as",
        "aten::pow",
    ],
    "aten::_solve_helper": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::_sparse_addmm": [
        "aten::addmm",
    ],
    "aten::_sparse_coo_tensor_unsafe": [
        "aten::_sparse_coo_tensor_with_dims_and_tensors",
    ],
    "aten::_sparse_mm": [
        "aten::_sparse_addmm",
    ],
    "aten::_sparse_sum": [
        "aten::_sparse_coo_tensor_with_dims_and_tensors",
        "aten::values",
    ],
    "aten::_std": [
        "aten::mean",
    ],
    "aten::_svd_helper": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::_symeig_helper": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::_thnn_differentiable_gru_cell_backward": [
        "aten::sigmoid",
        "aten::sigmoid_backward",
        "aten::tanh",
        "aten::tanh_backward",
        "aten::unsafe_chunk",
    ],
    "aten::_thnn_differentiable_lstm_cell_backward": [
        "aten::sigmoid",
        "aten::sigmoid_backward",
        "aten::tanh",
        "aten::tanh_backward",
        "aten::unsafe_chunk",
    ],
    "aten::_triangular_solve_helper": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::_trilinear": [
        "aten::bmm",
    ],
    "aten::_var": [
        "aten::mean",
    ],
    "aten::_weight_norm": [
        "aten::_weight_norm_cuda_interface",
        "aten::norm_except_dim",
    ],
    "aten::abs": [
        "aten::real",
    ],
    "aten::abs_": [
        "aten::abs",
    ],
    "aten::absolute": [
        "aten::abs",
        "aten::real",
    ],
    "aten::absolute_": [
        "aten::abs",
    ],
    "aten::acos_": [
        "aten::acos",
    ],
    "aten::acosh_": [
        "aten::acosh",
    ],
    "aten::adaptive_avg_pool1d": [
        "aten::adaptive_avg_pool2d",
    ],
    "aten::adaptive_avg_pool2d": [
        "aten::_adaptive_avg_pool2d",
        "aten::mean",
        "aten::mkldnn_adaptive_avg_pool2d",
    ],
    "aten::adaptive_max_pool1d": [
        "aten::adaptive_max_pool2d",
    ],
    "aten::add": [
        "aten::empty_meta",
    ],
    "aten::addbmm": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::addbmm_": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::addcdiv_": [
        "aten::addcdiv",
    ],
    "aten::addcmul_": [
        "aten::addcmul",
    ],
    "aten::addmm": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::addmm_": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::addmv": [
        "aten::_addmv_impl_",
    ],
    "aten::addmv_": [
        "aten::_addmv_impl_",
    ],
    "aten::addr": [
        "aten::_addr",
    ],
    "aten::addr_": [
        "aten::_addr_",
    ],
    "aten::affine_grid_generator": [
        "aten::bmm",
        "aten::linspace",
        "aten::transpose",
        "aten::unsqueeze_",
    ],
    "aten::affine_grid_generator_backward": [
        "aten::bmm",
        "aten::linspace",
        "aten::transpose",
        "aten::unsqueeze_",
    ],
    "aten::align_tensors": [
        "aten::rename",
    ],
    "aten::allclose": [
        "aten::all",
        "aten::isclose",
    ],
    "aten::alpha_dropout": [
        "aten::bernoulli_",
    ],
    "aten::alpha_dropout_": [
        "aten::bernoulli_",
    ],
    "aten::angle": [
        "aten::real",
    ],
    "aten::argsort": [
        "aten::sort",
    ],
    "aten::asin_": [
        "aten::asin",
    ],
    "aten::asinh_": [
        "aten::asinh",
    ],
    "aten::atan_": [
        "aten::atan",
    ],
    "aten::atanh_": [
        "aten::atanh",
    ],
    "aten::avg_pool1d": [
        "aten::avg_pool2d",
    ],
    "aten::baddbmm": [
        "aten::addmm_",
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::baddbmm_": [
        "aten::addmm_",
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::batch_norm": [
        "aten::_batch_norm_impl_index",
    ],
    "aten::bernoulli": [
        "aten::bernoulli_",
    ],
    "aten::bilinear": [
        "aten::_trilinear",
    ],
    "aten::binary_cross_entropy": [
        "aten::mean",
    ],
    "aten::binary_cross_entropy_with_logits": [
        "aten::clamp_min_",
        "aten::exp_",
        "aten::log_",
        "aten::mean",
        "aten::neg",
    ],
    "aten::binary_cross_entropy_with_logits_backward": [
        "aten::sigmoid",
    ],
    "aten::bitwise_and_": [
        "aten::bitwise_and",
    ],
    "aten::bitwise_not_": [
        "aten::bitwise_not",
    ],
    "aten::bitwise_or_": [
        "aten::bitwise_or",
    ],
    "aten::bitwise_xor_": [
        "aten::bitwise_xor",
    ],
    "aten::blackman_window": [
        "aten::cos_",
    ],
    "aten::bmm": [
        "aten::addmm_",
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::cartesian_prod": [
        "aten::flatten",
        "aten::meshgrid",
        "aten::stack",
    ],
    "aten::cat": [
        "aten::_cat",
        "aten::_sparse_coo_tensor_with_dims_and_tensors",
    ],
    "aten::cdist": [
        "aten::_cdist_forward",
        "aten::_euclidean_dist",
    ],
    "aten::ceil_": [
        "aten::ceil",
    ],
    "aten::celu": [
        "aten::elu",
    ],
    "aten::celu_": [
        "aten::elu_",
    ],
    "aten::chain_matmul": [
        "aten::mm",
    ],
    "aten::cholesky": [
        "aten::_cholesky_helper",
        "aten::tril_",
        "aten::triu_",
    ],
    "aten::cholesky_solve": [
        "aten::_cholesky_solve_helper",
    ],
    "aten::chunk": [
        "aten::split",
        "aten::split_with_sizes",
    ],
    "aten::clamp": [
        "aten::clamp_max",
        "aten::clamp_min",
    ],
    "aten::clamp_": [
        "aten::clamp",
    ],
    "aten::clamp_max_": [
        "aten::clamp_max",
    ],
    "aten::clamp_min_": [
        "aten::clamp_min",
    ],
    "aten::combinations": [
        "aten::full",
        "aten::le",
        "aten::lt",
        "aten::masked_select",
        "aten::meshgrid",
        "aten::stack",
    ],
    "aten::conv1d": [
        "aten::convolution",
    ],
    "aten::conv2d": [
        "aten::convolution",
    ],
    "aten::conv3d": [
        "aten::convolution",
    ],
    "aten::conv_tbc": [
        "aten::addmm_",
    ],
    "aten::conv_tbc_backward": [
        "aten::addmm_",
    ],
    "aten::conv_transpose1d": [
        "aten::convolution",
    ],
    "aten::conv_transpose2d": [
        "aten::convolution",
    ],
    "aten::conv_transpose3d": [
        "aten::convolution",
    ],
    "aten::convolution": [
        "aten::_convolution",
    ],
    "aten::cos_": [
        "aten::cos",
    ],
    "aten::cosh_": [
        "aten::cosh",
    ],
    "aten::cosine_embedding_loss": [
        "aten::clamp_min_",
        "aten::mean",
        "aten::sqrt_",
        "aten::where",
    ],
    "aten::cosine_similarity": [
        "aten::clamp_min_",
        "aten::sqrt_",
    ],
    "aten::ctc_loss": [
        "aten::_ctc_loss",
        "aten::_cudnn_ctc_loss",
        "aten::_use_cudnn_ctc_loss",
        "aten::clamp_min",
        "aten::mean",
        "aten::where",
    ],
    "aten::cummax": [
        "aten::_cummax_helper",
    ],
    "aten::cummin": [
        "aten::_cummin_helper",
    ],
    "aten::cumprod": [
        "aten::_cumprod",
    ],
    "aten::cumsum": [
        "aten::_cumsum",
    ],
    "aten::deg2rad_": [
        "aten::deg2rad",
    ],
    "aten::det": [
        "aten::_lu_with_info",
        "aten::all",
        "aten::diagonal",
        "aten::fmod_",
        "aten::ge",
        "aten::prod",
    ],
    "aten::diag_embed": [
        "aten::diagonal",
    ],
    "aten::diagflat": [
        "aten::diag",
    ],
    "aten::diagonal": [
        "aten::refine_names",
    ],
    "aten::dist": [
        "aten::norm",
    ],
    "aten::dropout": [
        "aten::_fused_dropout",
        "aten::bernoulli_",
    ],
    "aten::dropout_": [
        "aten::bernoulli_",
    ],
    "aten::einsum": [
        "aten::bmm",
        "aten::diagonal",
    ],
    "aten::elu_": [
        "aten::elu",
    ],
    "aten::embedding": [
        "aten::index_select",
    ],
    "aten::embedding_backward": [
        "aten::embedding_dense_backward",
        "aten::embedding_sparse_backward",
    ],
    "aten::embedding_bag": [
        "aten::_embedding_bag",
        "aten::_embedding_bag_forward_only",
    ],
    "aten::embedding_renorm_": [
        "aten::norm",
    ],
    "aten::embedding_sparse_backward": [
        "aten::_sparse_coo_tensor_unsafe",
        "aten::index",
    ],
    "aten::equal": [
        "aten::is_same_size",
    ],
    "aten::erf_": [
        "aten::erf",
    ],
    "aten::erfc_": [
        "aten::erfc",
    ],
    "aten::erfinv_": [
        "aten::erfinv",
    ],
    "aten::exp_": [
        "aten::exp",
    ],
    "aten::expm1_": [
        "aten::expm1",
    ],
    "aten::feature_alpha_dropout": [
        "aten::bernoulli_",
    ],
    "aten::feature_alpha_dropout_": [
        "aten::bernoulli_",
    ],
    "aten::feature_dropout": [
        "aten::bernoulli_",
    ],
    "aten::feature_dropout_": [
        "aten::bernoulli_",
    ],
    "aten::fft": [
        "aten::_fft_with_size",
    ],
    "aten::fliplr": [
        "aten::flip",
    ],
    "aten::flipud": [
        "aten::flip",
    ],
    "aten::floor_": [
        "aten::floor",
    ],
    "aten::floor_divide": [
        "aten::trunc_",
    ],
    "aten::floor_divide_": [
        "aten::floor_divide",
        "aten::trunc_",
    ],
    "aten::fmod_": [
        "aten::fmod",
    ],
    "aten::frac_": [
        "aten::frac",
    ],
    "aten::frobenius_norm": [
        "aten::conj",
        "aten::norm",
        "aten::real",
        "aten::sqrt",
    ],
    "aten::gcd_": [
        "aten::gcd",
    ],
    "aten::ge_": [
        "aten::ge",
    ],
    "aten::ger": [
        "aten::_addr",
    ],
    "aten::glu_backward": [
        "aten::sigmoid",
    ],
    "aten::grid_sampler": [
        "aten::cudnn_grid_sampler",
        "aten::grid_sampler_2d",
        "aten::grid_sampler_3d",
    ],
    "aten::group_norm": [
        "aten::native_group_norm",
    ],
    "aten::gru": [
        "aten::_thnn_fused_gru_cell",
        "aten::cudnn_is_acceptable",
        "aten::dropout",
        "aten::linear",
        "aten::matmul",
        "aten::sigmoid_",
        "aten::stack",
        "aten::tanh_",
        "aten::transpose",
        "aten::transpose_",
        "aten::unbind",
        "aten::unsafe_chunk",
    ],
    "aten::gru_cell": [
        "aten::_thnn_fused_gru_cell",
        "aten::linear",
        "aten::matmul",
        "aten::sigmoid_",
        "aten::tanh_",
        "aten::unsafe_chunk",
    ],
    "aten::gt_": [
        "aten::gt",
    ],
    "aten::hamming_window": [
        "aten::cos_",
    ],
    "aten::hann_window": [
        "aten::cos_",
    ],
    "aten::hardtanh": [
        "aten::clamp",
    ],
    "aten::hardtanh_": [
        "aten::clamp_",
    ],
    "aten::hinge_embedding_loss": [
        "aten::clamp_min_",
        "aten::mean",
        "aten::where",
    ],
    "aten::ifft": [
        "aten::_fft_with_size",
    ],
    "aten::imag": [
        "aten::view_as_real",
    ],
    "aten::index": [
        "aten::nonzero",
    ],
    "aten::index_add": [
        "aten::index_add_",
    ],
    "aten::index_copy": [
        "aten::index_copy_",
    ],
    "aten::index_copy_": [
        "aten::_index_copy_",
    ],
    "aten::index_fill": [
        "aten::index_fill_",
    ],
    "aten::index_put": [
        "aten::index_put_",
    ],
    "aten::index_put_": [
        "aten::_index_put_impl_",
    ],
    "aten::instance_norm": [
        "aten::alias",
        "aten::batch_norm",
        "aten::mean",
        "aten::repeat",
    ],
    "aten::inverse": [
        "aten::_inverse_helper",
    ],
    "aten::irfft": [
        "aten::_fft_with_size",
    ],
    "aten::isclose": [
        "aten::__iand__",
        "aten::__ior__",
        "aten::abs",
        "aten::isfinite",
        "aten::le",
    ],
    "aten::isfinite": [
        "aten::abs",
        "aten::ones_like",
    ],
    "aten::isinf": [
        "aten::__ior__",
        "aten::abs",
        "aten::imag",
        "aten::real",
    ],
    "aten::isreal": [
        "aten::imag",
        "aten::ones_like",
    ],
    "aten::istft": [
        "aten::_fft_with_size",
        "aten::abs",
        "aten::constant_pad_nd",
        "aten::conv_transpose1d",
        "aten::eye",
        "aten::ones",
        "aten::pow",
        "aten::repeat",
        "aten::transpose",
    ],
    "aten::item": [
        "aten::_local_scalar_dense",
    ],
    "aten::kl_div": [
        "aten::exp",
        "aten::gt",
        "aten::log",
        "aten::mean",
        "aten::where",
    ],
    "aten::kl_div_backward": [
        "aten::exp",
        "aten::expand_as",
        "aten::neg",
    ],
    "aten::kthvalue": [
        "aten::unsqueeze_",
    ],
    "aten::l1_loss": [
        "aten::abs_",
        "aten::mean",
    ],
    "aten::l1_loss_backward": [
        "aten::sign_",
    ],
    "aten::layer_norm": [
        "aten::native_layer_norm",
    ],
    "aten::lcm_": [
        "aten::lcm",
    ],
    "aten::le_": [
        "aten::le",
    ],
    "aten::leaky_relu_": [
        "aten::leaky_relu",
    ],
    "aten::lgamma_": [
        "aten::lgamma",
    ],
    "aten::linear": [
        "aten::addmm",
        "aten::matmul",
        "aten::mkldnn_linear",
    ],
    "aten::log10_": [
        "aten::log10",
    ],
    "aten::log1p_": [
        "aten::log1p",
    ],
    "aten::log2_": [
        "aten::log2",
    ],
    "aten::log_": [
        "aten::log",
    ],
    "aten::log_sigmoid": [
        "aten::log_sigmoid_forward",
    ],
    "aten::log_softmax": [
        "aten::_log_softmax",
    ],
    "aten::logcumsumexp": [
        "aten::_logcumsumexp",
    ],
    "aten::logdet": [
        "aten::_lu_with_info",
        "aten::abs_",
        "aten::all",
        "aten::diagonal",
        "aten::fmod_",
        "aten::full",
        "aten::ge",
        "aten::index_put_",
        "aten::log_",
        "aten::lt",
        "aten::nonzero_numpy",
        "aten::prod",
        "aten::sign",
    ],
    "aten::logical_and_": [
        "aten::logical_and",
    ],
    "aten::logical_not_": [
        "aten::logical_not",
    ],
    "aten::logical_or_": [
        "aten::logical_or",
    ],
    "aten::logical_xor_": [
        "aten::logical_xor",
    ],
    "aten::logit_": [
        "aten::logit",
    ],
    "aten::logsumexp": [
        "aten::abs",
        "aten::exp",
        "aten::log_",
        "aten::masked_fill_",
        "aten::max_values",
    ],
    "aten::lstm": [
        "aten::_thnn_fused_lstm_cell",
        "aten::cudnn_is_acceptable",
        "aten::dropout",
        "aten::linear",
        "aten::matmul",
        "aten::sigmoid_",
        "aten::stack",
        "aten::tanh",
        "aten::tanh_",
        "aten::transpose",
        "aten::unbind",
        "aten::unsafe_chunk",
    ],
    "aten::lstm_cell": [
        "aten::_thnn_fused_lstm_cell",
        "aten::linear",
        "aten::matmul",
        "aten::sigmoid_",
        "aten::tanh",
        "aten::tanh_",
        "aten::unsafe_chunk",
    ],
    "aten::lt_": [
        "aten::lt",
    ],
    "aten::lu_solve": [
        "aten::_lu_solve_helper",
    ],
    "aten::margin_ranking_loss": [
        "aten::clamp_min_",
        "aten::mean",
        "aten::neg",
    ],
    "aten::masked_fill": [
        "aten::masked_fill_",
    ],
    "aten::masked_scatter": [
        "aten::masked_scatter_",
    ],
    "aten::matmul": [
        "aten::_unsafe_view",
        "aten::bmm",
        "aten::dot",
        "aten::mm",
        "aten::mv",
        "aten::transpose",
    ],
    "aten::matrix_power": [
        "aten::_unsafe_view",
        "aten::bmm",
        "aten::dot",
        "aten::expand_as",
        "aten::eye",
        "aten::inverse",
        "aten::mm",
        "aten::mv",
        "aten::transpose",
    ],
    "aten::matrix_rank": [
        "aten::abs",
        "aten::gt",
        "aten::svd",
        "aten::symeig",
    ],
    "aten::max": [
        "aten::_make_per_tensor_quantized_tensor",
        "aten::int_repr",
        "aten::unsqueeze_",
    ],
    "aten::max_pool1d": [
        "aten::max_pool1d_with_indices",
    ],
    "aten::max_pool1d_with_indices": [
        "aten::max_pool2d_with_indices",
    ],
    "aten::max_pool2d": [
        "aten::max_pool2d_with_indices",
        "aten::mkldnn_max_pool2d",
        "aten::quantized_max_pool2d",
    ],
    "aten::max_pool3d": [
        "aten::max_pool3d_with_indices",
        "aten::mkldnn_max_pool3d",
    ],
    "aten::median": [
        "aten::kthvalue",
    ],
    "aten::min": [
        "aten::_make_per_tensor_quantized_tensor",
        "aten::int_repr",
        "aten::unsqueeze_",
    ],
    "aten::mm": [
        "aten::transpose",
        "aten::transpose_",
    ],
    "aten::mode": [
        "aten::_mode",
    ],
    "aten::mse_loss": [
        "aten::mean",
    ],
    "aten::multilabel_margin_loss": [
        "aten::multilabel_margin_loss_forward",
    ],
    "aten::multinomial": [
        "aten::bitwise_and",
        "aten::ge",
        "aten::log_",
        "aten::lt",
        "aten::topk",
        "aten::uniform_",
    ],
    "aten::mv": [
        "aten::_addmv_impl_",
    ],
    "aten::mvlgamma": [
        "aten::all",
        "aten::gt",
        "aten::lgamma_",
    ],
    "aten::mvlgamma_": [
        "aten::all",
        "aten::gt",
        "aten::lgamma_",
    ],
    "aten::neg_": [
        "aten::neg",
    ],
    "aten::new_full": [
        "aten::full",
    ],
    "aten::nll_loss": [
        "aten::nll_loss_forward",
    ],
    "aten::nll_loss2d": [
        "aten::nll_loss2d_forward",
    ],
    "aten::nonzero_numpy": [
        "aten::nonzero",
        "aten::unbind",
    ],
    "aten::norm": [
        "aten::native_norm",
    ],
    "aten::norm_except_dim": [
        "aten::norm",
        "aten::transpose",
    ],
    "aten::normal": [
        "aten::full",
        "aten::normal_",
        "aten::view_as_real",
    ],
    "aten::normal_": [
        "aten::view_as_real",
    ],
    "aten::nuclear_norm": [
        "aten::svd",
        "aten::unsqueeze_",
    ],
    "aten::one_hot": [
        "aten::scatter_",
    ],
    "aten::pairwise_distance": [
        "aten::norm",
    ],
    "aten::pdist": [
        "aten::_pdist_forward",
    ],
    "aten::pin_memory": [
        "aten::is_pinned",
    ],
    "aten::pinverse": [
        "aten::diag_embed",
        "aten::gt",
        "aten::matmul",
        "aten::reciprocal",
        "aten::svd",
        "aten::transpose",
        "aten::where",
    ],
    "aten::poisson_nll_loss": [
        "aten::exp",
        "aten::le",
        "aten::log",
        "aten::masked_fill",
        "aten::mean",
    ],
    "aten::polygamma_": [
        "aten::polygamma",
    ],
    "aten::pow": [
        "aten::can_cast",
        "aten::result_type",
    ],
    "aten::pow_": [
        "aten::can_cast",
        "aten::result_type",
    ],
    "aten::qr": [
        "aten::_qr_helper",
    ],
    "aten::quantized_gru": [
        "aten::_thnn_fused_gru_cell",
        "aten::dropout",
        "aten::fbgemm_linear_int8_weight_fp32_activation",
        "aten::sigmoid_",
        "aten::stack",
        "aten::tanh_",
        "aten::transpose",
        "aten::transpose_",
        "aten::unbind",
        "aten::unsafe_chunk",
    ],
    "aten::quantized_gru_cell": [
        "aten::_thnn_fused_gru_cell",
        "aten::fbgemm_linear_int8_weight_fp32_activation",
        "aten::sigmoid_",
        "aten::tanh_",
        "aten::unsafe_chunk",
    ],
    "aten::quantized_lstm": [
        "aten::_thnn_fused_lstm_cell",
        "aten::dropout",
        "aten::fbgemm_linear_int8_weight_fp32_activation",
        "aten::sigmoid_",
        "aten::stack",
        "aten::tanh",
        "aten::tanh_",
        "aten::transpose",
        "aten::unbind",
        "aten::unsafe_chunk",
    ],
    "aten::quantized_lstm_cell": [
        "aten::_thnn_fused_lstm_cell",
        "aten::fbgemm_linear_int8_weight_fp32_activation",
        "aten::sigmoid_",
        "aten::tanh",
        "aten::tanh_",
        "aten::unsafe_chunk",
    ],
    "aten::quantized_rnn_relu_cell": [
        "aten::fbgemm_linear_int8_weight_fp32_activation",
        "aten::relu",
    ],
    "aten::quantized_rnn_tanh_cell": [
        "aten::fbgemm_linear_int8_weight_fp32_activation",
        "aten::tanh",
    ],
    "aten::rad2deg_": [
        "aten::rad2deg",
    ],
    "aten::rand": [
        "aten::uniform_",
    ],
    "aten::rand_like": [
        "aten::uniform_",
    ],
    "aten::randint": [
        "aten::random_",
    ],
    "aten::randint_like": [
        "aten::random_",
    ],
    "aten::randn": [
        "aten::normal_",
    ],
    "aten::randn_like": [
        "aten::normal_",
    ],
    "aten::real": [
        "aten::view_as_real",
    ],
    "aten::reciprocal_": [
        "aten::reciprocal",
    ],
    "aten::refine_names": [
        "aten::alias",
    ],
    "aten::relu": [
        "aten::threshold",
    ],
    "aten::relu_": [
        "aten::threshold_",
    ],
    "aten::rename": [
        "aten::alias",
    ],
    "aten::repeat": [
        "aten::alias",
        "aten::empty_quantized",
        "aten::expand_as",
        "aten::unfold",
    ],
    "aten::repeat_interleave": [
        "aten::all",
        "aten::cumsum",
        "aten::flatten",
        "aten::ge",
        "aten::index_select",
    ],
    "aten::reshape": [
        "aten::_mkldnn_reshape",
        "aten::_unsafe_view",
    ],
    "aten::rfft": [
        "aten::_fft_with_size",
    ],
    "aten::rnn_relu": [
        "aten::cudnn_is_acceptable",
        "aten::dropout",
        "aten::linear",
        "aten::matmul",
        "aten::relu",
        "aten::stack",
        "aten::transpose",
        "aten::transpose_",
        "aten::unbind",
    ],
    "aten::rnn_relu_cell": [
        "aten::linear",
        "aten::matmul",
        "aten::relu",
    ],
    "aten::rnn_tanh": [
        "aten::cudnn_is_acceptable",
        "aten::dropout",
        "aten::linear",
        "aten::matmul",
        "aten::stack",
        "aten::tanh",
        "aten::transpose",
        "aten::transpose_",
        "aten::unbind",
    ],
    "aten::rnn_tanh_cell": [
        "aten::linear",
        "aten::matmul",
        "aten::tanh",
    ],
    "aten::rot90": [
        "aten::flip",
        "aten::transpose_",
    ],
    "aten::round_": [
        "aten::round",
    ],
    "aten::rrelu": [
        "aten::rrelu_with_noise",
    ],
    "aten::rrelu_": [
        "aten::rrelu_with_noise_",
    ],
    "aten::rrelu_with_noise": [
        "aten::leaky_relu",
    ],
    "aten::rrelu_with_noise_": [
        "aten::leaky_relu",
    ],
    "aten::rrelu_with_noise_backward": [
        "aten::leaky_relu_backward",
    ],
    "aten::rsqrt_": [
        "aten::rsqrt",
    ],
    "aten::scatter": [
        "aten::scatter_",
    ],
    "aten::scatter_add": [
        "aten::scatter_add_",
    ],
    "aten::select": [
        "aten::_sparse_coo_tensor_with_dims_and_tensors",
        "aten::index_select",
        "aten::nonzero",
    ],
    "aten::selu": [
        "aten::elu",
    ],
    "aten::selu_": [
        "aten::elu_",
    ],
    "aten::sigmoid_": [
        "aten::sigmoid",
    ],
    "aten::sign_": [
        "aten::sign",
    ],
    "aten::silu": [
        "aten::sigmoid",
    ],
    "aten::silu_": [
        "aten::sigmoid",
    ],
    "aten::silu_backward": [
        "aten::sigmoid",
    ],
    "aten::sin_": [
        "aten::sin",
    ],
    "aten::sinh_": [
        "aten::sinh",
    ],
    "aten::slogdet": [
        "aten::_lu_with_info",
        "aten::abs_",
        "aten::all",
        "aten::diagonal",
        "aten::fmod_",
        "aten::ge",
        "aten::log_",
        "aten::prod",
        "aten::sign",
    ],
    "aten::slow_conv3d": [
        "aten::slow_conv3d_forward",
    ],
    "aten::slow_conv3d_backward": [
        "aten::addmm_",
        "aten::baddbmm_",
        "aten::bmm",
        "aten::mm",
        "aten::transpose",
    ],
    "aten::slow_conv3d_forward": [
        "aten::addmm_",
        "aten::baddbmm_",
        "aten::bmm",
        "aten::mm",
    ],
    "aten::smm": [
        "aten::sspaddmm",
    ],
    "aten::smooth_l1_loss": [
        "aten::mean",
    ],
    "aten::soft_margin_loss": [
        "aten::exp_",
        "aten::log_",
        "aten::mean",
        "aten::neg",
    ],
    "aten::soft_margin_loss_backward": [
        "aten::exp",
        "aten::neg",
    ],
    "aten::softmax": [
        "aten::_softmax",
    ],
    "aten::solve": [
        "aten::_solve_helper",
    ],
    "aten::sort": [
        "aten::_make_per_tensor_quantized_tensor",
        "aten::int_repr",
    ],
    "aten::sparse_coo_tensor": [
        "aten::_sparse_coo_tensor_with_dims",
        "aten::_sparse_coo_tensor_with_dims_and_tensors",
    ],
    "aten::sqrt_": [
        "aten::sqrt",
    ],
    "aten::square": [
        "aten::pow",
    ],
    "aten::square_": [
        "aten::pow",
    ],
    "aten::std": [
        "aten::_std",
        "aten::imag",
        "aten::real",
        "aten::sqrt",
    ],
    "aten::std_mean": [
        "aten::imag",
        "aten::real",
        "aten::sqrt",
    ],
    "aten::stft": [
        "aten::rfft",
        "aten::transpose_",
    ],
    "aten::svd": [
        "aten::_svd_helper",
    ],
    "aten::symeig": [
        "aten::_symeig_helper",
    ],
    "aten::t": [
        "aten::transpose",
    ],
    "aten::t_": [
        "aten::transpose_",
    ],
    "aten::tan_": [
        "aten::tan",
    ],
    "aten::tanh_": [
        "aten::tanh",
    ],
    "aten::tensordot": [
        "aten::mm",
    ],
    "aten::thnn_conv2d": [
        "aten::thnn_conv2d_forward",
    ],
    "aten::thnn_conv2d_backward": [
        "aten::addmm_",
        "aten::transpose",
    ],
    "aten::thnn_conv2d_forward": [
        "aten::addmm_",
        "aten::mm",
    ],
    "aten::thnn_conv_depthwise2d": [
        "aten::thnn_conv_depthwise2d_forward",
    ],
    "aten::to_dense_backward": [
        "aten::sparse_mask",
        "aten::to_mkldnn",
    ],
    "aten::to_mkldnn_backward": [
        "aten::to_dense",
    ],
    "aten::to_sparse": [
        "aten::chunk",
        "aten::index",
        "aten::nonzero",
        "aten::sparse_coo_tensor",
        "aten::transpose",
        "aten::unique_dim",
    ],
    "aten::transpose": [
        "aten::_mkldnn_transpose",
    ],
    "aten::transpose_": [
        "aten::_mkldnn_transpose_",
    ],
    "aten::triangular_solve": [
        "aten::_triangular_solve_helper",
    ],
    "aten::triplet_margin_loss": [
        "aten::clamp_min",
        "aten::mean",
        "aten::pairwise_distance",
    ],
    "aten::trunc_": [
        "aten::trunc",
    ],
    "aten::uniform_": [
        "aten::view_as_real",
    ],
    "aten::unique_consecutive": [
        "aten::stack",
        "aten::transpose",
        "aten::unbind",
    ],
    "aten::unique_dim": [
        "aten::stack",
        "aten::transpose",
        "aten::unbind",
    ],
    "aten::unique_dim_consecutive": [
        "aten::stack",
        "aten::transpose",
        "aten::unbind",
    ],
    "aten::unsafe_chunk": [
        "aten::unsafe_split",
        "aten::unsafe_split_with_sizes",
    ],
    "aten::unsqueeze": [
        "aten::_cat",
        "aten::_sparse_coo_tensor_with_dims_and_tensors",
    ],
    "aten::vander": [
        "aten::cumprod",
        "aten::flip",
        "aten::promote_types",
    ],
    "aten::var": [
        "aten::_var",
        "aten::imag",
        "aten::real",
        "aten::sqrt",
        "aten::std",
    ],
    "aten::var_mean": [
        "aten::imag",
        "aten::real",
        "aten::sqrt",
    ],
    "aten::where": [
        "aten::_s_where",
        "aten::nonzero_numpy",
    ],
    "quantized::make_quantized_cell_params": [
        "aten::fbgemm_linear_int8_weight_fp32_activation",
    ],
    "quantized::max_pool2d": [
        "aten::max_pool2d",
    ],
    "quantized::quantized_gru_cell_dynamic": [
        "aten::_thnn_fused_gru_cell",
        "aten::sigmoid_",
        "aten::tanh_",
        "aten::unsafe_chunk",
    ],
    "quantized::quantized_lstm_cell_dynamic": [
        "aten::_thnn_fused_lstm_cell",
        "aten::sigmoid_",
        "aten::tanh",
        "aten::tanh_",
        "aten::unsafe_chunk",
    ],
    "quantized::quantized_rnn_relu_cell_dynamic": [
        "aten::relu",
    ],
    "quantized::quantized_rnn_tanh_cell_dynamic": [
        "aten::tanh",
    ],
}
