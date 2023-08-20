import argparse
import datetime
import re
import sys
import warnings
from collections import defaultdict

import torch
from torch._C import parse_schema


# How to run this test locally:
# 1 Have two virtual environments (eg conda env), one without PyTorch installed (venv_nightly)
#   one with your local changes (venv_yours).
# In venv_nightly:
# 2. First ensure that Pytorch is uninstalled, but all prereqs are installed
# 3. Install torch nightly build with
#    `pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html`
# 4. Generate original schemas with
#    `python test/forward_backward_compatibility/dump_all_function_schemas.py --filename nightly_schemas.txt`
# Now in venv_yours:
# 5. Run this test with
#    `python test/forward_backward_compatibility/check_forward_backward_compatibility.py --existing-schemas nightly_schemas.txt`

# The date specifies how long the allowlist exclusion should apply to.
#
#   - If we NEVER give BC guarantee for an operator, you can put the
#     date arbitrarily far in the future.
#   - Otherwise, pick a date that is far enough in the future that you
#     believe you can land your diff before then.
#
# Allowlist entries can be removed after the date listed on them passes.
#
# Allowlist item format:
# [
#   0: function name regex
#   1: date until which the allowlist entry is valid
#   2: (optional) function argument regex
# ]
#
# NB: function name DOES NOT include overload name!
ALLOW_LIST = [
    ("c10_experimental", datetime.date(9999, 1, 1)),
    # Internal
    ("static", datetime.date(9999, 1, 1)),
    ("prim::ModuleDictIndex", datetime.date(9999, 1, 1)),
    ("prim::MKLDNNRelu6", datetime.date(9999, 1, 1)),
    ("prim::MKLDNNRelu6_", datetime.date(9999, 1, 1)),
    ("prim::Concat", datetime.date(9999, 1, 1)),
    ("aten::_NestedTensor_GeneralizedBMM", datetime.date(9999, 1, 1)),
    # Internal, profiler-specific ops
    ("profiler::_call_end_callbacks_on_jit_fut*", datetime.date(9999, 1, 1)),
    ("profiler::_record_function_enter", datetime.date(9999, 1, 1)),
    ("aten::_cholesky_helper", datetime.date(9999, 1, 1)),
    ("aten::_lstsq_helper", datetime.date(9999, 1, 1)),
    ("aten::_syevd_helper", datetime.date(9999, 1, 1)),
    ("aten::_linalg_solve_out_helper_", datetime.date(9999, 1, 1)),
    ("aten::select_backward", datetime.date(9999, 1, 1)),
    ("aten::lstsq", datetime.date(9999, 1, 1)),
    ("aten::lstsq.X", datetime.date(9999, 1, 1)),
    ("aten::slice_backward", datetime.date(9999, 1, 1)),
    ("aten::diagonal_backward", datetime.date(9999, 1, 1)),
    ("aten::rowwise_prune", datetime.date(9999, 1, 1)),
    ("aten::eig", datetime.date(9999, 1, 1)),
    ("aten::eig.e", datetime.date(9999, 1, 1)),
    ("aten::adaptive_avg_pool3d_backward", datetime.date(9999, 1, 1)),
    ("aten::_embedding_bag_dense_backward", datetime.date(9999, 1, 1)),
    ("aten::matrix_rank", datetime.date(9999, 1, 1)),
    ("aten::matrix_rank.tol", datetime.date(9999, 1, 1)),
    ("aten::randperm", datetime.date(9999, 1, 1)),
    ("aten::solve", datetime.date(9999, 1, 1)),
    ("aten::solve.solution", datetime.date(9999, 1, 1)),
    ("aten::_solve_helper", datetime.date(9999, 1, 1)),
    ("aten::_convolution_nogroup", datetime.date(9999, 1, 1)),
    ("aten::miopen_convolution_backward", datetime.date(9999, 1, 1)),
    ("aten::miopen_convolution_backward_bias", datetime.date(9999, 1, 1)),
    ("aten::miopen_convolution_backward_input", datetime.date(9999, 1, 1)),
    ("aten::miopen_convolution_backward_weight", datetime.date(9999, 1, 1)),
    ("aten::miopen_convolution_transpose_backward", datetime.date(9999, 1, 1)),
    ("aten::miopen_convolution_transpose_backward_input", datetime.date(9999, 1, 1)),
    ("aten::miopen_convolution_transpose_backward_weight", datetime.date(9999, 1, 1)),
    ("aten::miopen_depthwise_convolution_backward", datetime.date(9999, 1, 1)),
    ("aten::miopen_depthwise_convolution_backward_input", datetime.date(9999, 1, 1)),
    ("aten::miopen_depthwise_convolution_backward_weight", datetime.date(9999, 1, 1)),
    ("aten::_nested_tensor", datetime.date(9999, 1, 1)),
    ("prepacked::unpack_prepacked_sizes_conv2d", datetime.date(9999, 1, 1)),
    ("prepacked::unpack_prepacked_sizes_linear", datetime.date(9999, 1, 1)),
    ("aten::_symeig_helper", datetime.date(9999, 1, 1)),
    ("aten::symeig", datetime.date(9999, 1, 1)),
    ("aten::symeig.e", datetime.date(9999, 1, 1)),
    ("aten::native_multi_head_self_attention", datetime.date(9999, 1, 1)),
    ("aten::_native_multi_head_self_attention", datetime.date(9999, 1, 1)),
    ("aten::grid_sampler_3d_backward", datetime.date(9999, 1, 1)),
    ("aten::_transform_bias_rescale_qkv", datetime.date(9999, 1, 1)),
    ("prim::infer_squeeze_size.dim", datetime.date(9999, 1, 1)),
    ("prim::infer_squeeze_size", datetime.date(9999, 1, 1)),
    ("aten::_weight_norm_cuda_interface", datetime.date(9999, 1, 1)),
    ("aten::_weight_norm_cuda_interface_backward", datetime.date(9999, 1, 1)),
    ("aten::empty.SymInt", datetime.date(9999, 1, 1)),
    # nested tensor temporary auxiliary ops
    ("aten::_reshape_nested", datetime.date(9999, 1, 1)),
    ("aten::_reshape_nested_backward", datetime.date(9999, 1, 1)),
    ("aten::mps_linear", datetime.date(9999, 1, 1)),
    ("aten::_mps_linear", datetime.date(9999, 1, 1)),
    ("aten::_mps_max_pool2d", datetime.date(9999, 1, 1)),
    ("aten::_mps_max_pool2d.out", datetime.date(9999, 1, 1)),
    ("aten::mps_max_pool2d_backward", datetime.date(9999, 1, 1)),
    ("aten::mps_max_pool2d_backward.out", datetime.date(9999, 1, 1)),
    # TODO: FIXME: prims shouldn't be checked
    ("prims::.*", datetime.date(9999, 1, 1)),
    ("aten::_amp_foreach_non_finite_check_and_unscale.out", datetime.date(2022, 9, 1)),
    ("aten::_amp_foreach_non_finite_check_and_unscale_", datetime.date(2022, 9, 1)),
    ("aten::_cudnn_rnn_backward.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_abs.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_abs_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_acos.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_acos_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_add.List_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_add.ScalarList_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_add.Scalar_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_add_.List", datetime.date(2022, 9, 1)),
    ("aten::_foreach_add_.Scalar", datetime.date(2022, 9, 1)),
    ("aten::_foreach_add_.ScalarList", datetime.date(2022, 9, 1)),
    ("aten::_foreach_addcdiv.ScalarList_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_addcdiv.Scalar_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_addcdiv_.Scalar", datetime.date(2022, 9, 1)),
    ("aten::_foreach_addcdiv_.ScalarList", datetime.date(2022, 9, 1)),
    ("aten::_foreach_addcmul.ScalarList_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_addcmul.Scalar_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_addcmul_.Scalar", datetime.date(2022, 9, 1)),
    ("aten::_foreach_addcmul_.ScalarList", datetime.date(2022, 9, 1)),
    ("aten::_foreach_asin.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_asin_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_atan.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_atan_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_ceil.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_ceil_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_cos.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_cos_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_cosh.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_cosh_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_div.List_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_div.ScalarList_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_div.Scalar_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_div_.List", datetime.date(2022, 9, 1)),
    ("aten::_foreach_div_.Scalar", datetime.date(2022, 9, 1)),
    ("aten::_foreach_div_.ScalarList", datetime.date(2022, 9, 1)),
    ("aten::_foreach_erf.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_erf_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_erfc.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_erfc_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_exp.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_exp_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_expm1.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_expm1_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_floor.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_floor_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_frac.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_frac_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_lgamma.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_lgamma_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_log.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_log10.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_log10_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_log1p.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_log1p_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_log2.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_log2_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_log_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_maximum.List_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_maximum_.List", datetime.date(2022, 9, 1)),
    ("aten::_foreach_minimum.List_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_minimum_.List", datetime.date(2022, 9, 1)),
    ("aten::_foreach_mul.List_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_mul.ScalarList_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_mul.Scalar_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_mul_.List", datetime.date(2022, 9, 1)),
    ("aten::_foreach_mul_.Scalar", datetime.date(2022, 9, 1)),
    ("aten::_foreach_mul_.ScalarList", datetime.date(2022, 9, 1)),
    ("aten::_foreach_neg.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_neg_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_norm.Scalar_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_reciprocal.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_reciprocal_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_round.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_round_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sigmoid.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sigmoid_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sin.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sin_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sinh.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sinh_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sqrt.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sqrt_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sub.List_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sub.ScalarList_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sub.Scalar_out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sub_.List", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sub_.Scalar", datetime.date(2022, 9, 1)),
    ("aten::_foreach_sub_.ScalarList", datetime.date(2022, 9, 1)),
    ("aten::_foreach_tan.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_tan_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_tanh.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_tanh_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_trunc.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_trunc_", datetime.date(2022, 9, 1)),
    ("aten::_foreach_zero.out", datetime.date(2022, 9, 1)),
    ("aten::_foreach_zero_", datetime.date(2022, 9, 1)),
    ("aten::_histogramdd_bin_edges.out", datetime.date(2022, 9, 1)),
    ("aten::chunk", datetime.date(2022, 9, 1)),
    ("aten::dequantize.tensors_out", datetime.date(2022, 9, 1)),
    ("aten::dsplit.array", datetime.date(2022, 9, 1)),
    ("aten::dsplit.int", datetime.date(2022, 9, 1)),
    ("aten::hsplit.array", datetime.date(2022, 9, 1)),
    ("aten::hsplit.int", datetime.date(2022, 9, 1)),
    ("aten::lstm_mps_backward.out", datetime.date(2022, 9, 1)),
    ("aten::miopen_rnn_backward.out", datetime.date(2022, 9, 1)),
    ("aten::quantize_per_tensor.tensors_out", datetime.date(2022, 9, 1)),
    ("aten::split", datetime.date(2022, 9, 1)),
    ("aten::split.Tensor", datetime.date(2022, 9, 1)),
    ("aten::split.sizes", datetime.date(2022, 9, 1)),
    ("aten::split_copy.Tensor_out", datetime.date(2022, 9, 1)),
    ("aten::split_with_sizes", datetime.date(2022, 9, 1)),
    ("aten::split_with_sizes_copy.out", datetime.date(2022, 9, 1)),
    ("aten::tensor_split.indices", datetime.date(2022, 9, 1)),
    ("aten::tensor_split.sections", datetime.date(2022, 9, 1)),
    ("aten::tensor_split.tensor_indices_or_sections", datetime.date(2022, 9, 1)),
    ("aten::unbind.Dimname", datetime.date(2022, 9, 1)),
    ("aten::unbind.int", datetime.date(2022, 9, 1)),
    ("aten::unbind_copy.int_out", datetime.date(2022, 9, 1)),
    ("aten::unsafe_split.Tensor_out", datetime.date(2022, 9, 1)),
    ("aten::unsafe_split_with_sizes.out", datetime.date(2022, 9, 1)),
    ("aten::vsplit.array", datetime.date(2022, 9, 1)),
    ("aten::vsplit.int", datetime.date(2022, 9, 1)),
    ("aten::sym_numel", datetime.date(2022, 10, 1)),
    ("aten::to_padded_tensor", datetime.date(2022, 10, 1)),
    ("aten::nested_to_padded_tensor", datetime.date(2022, 10, 1)),
    ("aten::nested_tensor", datetime.date(2022, 10, 15)),
    ("aten::_nested_tensor_layer_norm", datetime.date(2022, 10, 15)),
    ("aten::_torch_cuda_cu_linker_symbol_op", datetime.date(2022, 11, 1)),
    ("aten::_test_inductor_realize", datetime.date(2023, 1, 1)),

    ("aten::upsample_linear1d_backward", datetime.date(2022, 12, 15)),
    ("aten::upsample_bicubic2d_backward", datetime.date(2022, 12, 15)),
    ("aten::upsample_trilinear3d", datetime.date(2022, 12, 15)),
    ("aten::upsample_bilinear2d", datetime.date(2022, 12, 15)),
    ("aten::upsample_nearest3d", datetime.date(2022, 12, 15)),
    ("aten::upsample_nearest2d_backward", datetime.date(2022, 12, 15)),
    ("aten::upsample_bilinear2d_backward", datetime.date(2022, 12, 15)),
    ("aten::upsample_trilinear3d_backward", datetime.date(2022, 12, 15)),
    ("aten::upsample_nearest2d", datetime.date(2022, 12, 15)),
    ("aten::upsample_bicubic2d", datetime.date(2022, 12, 15)),
    ("aten::upsample_nearest1d_backward", datetime.date(2022, 12, 15)),
    ("aten::upsample_nearest3d_backward", datetime.date(2022, 12, 15)),
    ("aten::upsample_linear1d", datetime.date(2022, 12, 15)),
    ("aten::upsample_nearest1d", datetime.date(2022, 12, 15)),
    ("aten::_upsample_nearest_exact3d", datetime.date(2022, 12, 15)),
    ("aten::_upsample_nearest_exact3d_backward", datetime.date(2022, 12, 15)),
    ("aten::_upsample_bilinear2d_aa", datetime.date(2022, 12, 15)),
    ("aten::_upsample_bilinear2d_aa_backward", datetime.date(2022, 12, 15)),
    ("aten::_upsample_bicubic2d_aa", datetime.date(2022, 12, 15)),
    ("aten::_upsample_bicubic2d_aa_backward", datetime.date(2022, 12, 15)),
    ("aten::_upsample_nearest_exact1d", datetime.date(2022, 12, 15)),
    ("aten::_upsample_nearest_exact1d_backward", datetime.date(2022, 12, 15)),
    ("aten::_upsample_nearest_exact2d", datetime.date(2022, 12, 15)),
    ("aten::_upsample_nearest_exact2d_backward", datetime.date(2022, 12, 15)),
    ("aten::_scaled_dot_product_attention", datetime.date(2023, 8, 1)),
    ("aten::_chunk_grad_outputs_efficient_attention", datetime.date(2023, 8, 1)),
    ("aten::_scaled_dot_product_flash_attention", datetime.date(2023, 5, 15)),
    ("aten::_scaled_dot_product_efficient_attention", datetime.date(2023, 8, 15)),
    ("aten::_scaled_dot_product_efficient_attention_backward", datetime.date(2023, 8, 15)),
    ("aten::_sparse_mask_helper", datetime.date(2023, 3, 15)),
    ("aten::_fused_sdp_choice", datetime.date(2023, 3, 15)),
    ("aten::_flash_attention_forward", datetime.date(2023, 5, 15)),
    ("aten::_flash_attention_backward", datetime.date(2023, 5, 15)),
    ("aten::_efficient_attention_forward", datetime.date(2023, 7, 1)),
    ("aten::_efficient_attention_backward", datetime.date(2023, 8, 1)),
    ("mkldnn::_convolution_pointwise.binary", datetime.date(2022, 12, 15)),
    ("prim::CudaFusionIvalGuard", datetime.date(2023, 2, 1)),
    ("prim::CudaFusionGuard", datetime.date(2023, 2, 1)),
    ("prim::CudaFusionGroup", datetime.date(2023, 2, 1)),
    ("prim::CudaFusionViewGuard", datetime.date(2023, 2, 1)),
    ("prim::CudaFusionSizeEq", datetime.date(2023, 2, 1)),
    ("prim::transpose_copy.int", datetime.date(2023, 2, 1)),
    ("prim::expand_as_copy", datetime.date(2023, 2, 1)),
    ("prim::squeeze_copy", datetime.date(2023, 2, 1)),
    ("prim::squeeze_copy.dim", datetime.date(2023, 2, 1)),
    ("prim::unsqueeze_copy", datetime.date(2023, 2, 1)),
    ("prim::expand_copy", datetime.date(2023, 2, 1)),
    ("prim::flatten_copy", datetime.date(2023, 2, 1)),
    ("prim::add_optional", datetime.date(2023, 2, 1)),
    ("prim::reshape_copy", datetime.date(2023, 2, 1)),
    ("prim::permute_copy", datetime.date(2023, 2, 1)),
    ("prim::infer_unsqueeze_size", datetime.date(2023, 2, 1)),
    ("prim::t_copy", datetime.date(2023, 2, 1)),
    ("prim::view_copy", datetime.date(2023, 2, 1)),
    # BetterTransformer 1.0 internal operators
    ("aten::_transformer_decoder_only_layer_fwd", datetime.date(9999, 1, 1)),
    ("aten::_native_decoder_only_multi_head_attention",
     datetime.date(9999, 1, 1)),
    ("mkldnn::_convolution_pointwise_.binary", datetime.date(2023, 7, 1)),
    # These ops were moved to python under the c10d_functional namespace
    ("c10d::allreduce_", datetime.date(2023, 7, 30)),
    ("aten::wait_tensor", datetime.date(9999, 1, 30)),
    ("aten::reduce_scatter_tensor", datetime.date(9999, 1, 30)),
    ("aten::all_gather_into_tensor", datetime.date(9999, 1, 30)),
    ("aten::all_reduce", datetime.date(9999, 1, 30)),
    ("aten::to_sparse.out", datetime.date(2023, 12, 31)),
    ("aten::to_sparse.sparse_dim_out", datetime.date(2023, 12, 31)),
    ("aten::to_sparse_bsc.out", datetime.date(2023, 12, 31)),
    ("aten::to_sparse_bsr.out", datetime.date(2023, 12, 31)),
    ("aten::to_sparse_csc.out", datetime.date(2023, 12, 31)),
    ("aten::to_sparse_csr.out", datetime.date(2023, 12, 31)),
    ("aten::_structured_sparse_linear", datetime.date(2023, 12, 31)),
    ("aten::batch_norm_backward_elemt.out", datetime.date(2023, 12, 31)),
    ("aten::batch_norm_backward_elemt", datetime.date(2023, 12, 31)),
    ("aten::sym_constrain_range", datetime.date(2023, 12, 31)),
]

ALLOW_LIST_COMPILED = [
    (
        re.compile(item[0]),
        item[1],
        re.compile(item[2]) if len(item) > 2 else None,
    ) for item in ALLOW_LIST if item[1] >= datetime.date.today()
]

def allow_listed(schema):
    for item in ALLOW_LIST_COMPILED:
        if item[0].search(str(schema)):
            if len(item) > 2 and item[2] is not None:
                # if arguments regex is present, use it
                return bool(item[2].search(str(schema)))
            return True
    return False


# The nightly will fail to parse newly added syntax to schema declarations
# Add new schemas that will fail the nightly here
dont_parse_list = [
    ("_TorchScriptTesting.*", datetime.date(2099, 9, 17)),
    ("test_backend", datetime.date(2099, 9, 17)),
    ("dist_c10d", datetime.date(2099, 9, 17)),
    ("__backends__.nnc", datetime.date(2099, 9, 17)),
]

def has_valid_upgraders(schema, version_map):
    # we want to parse through the map to find if
    # the schema has valid upgraders. Since the
    # version map has entry for each overload
    # we need to do some ugly parsing.

    # the name of the operator
    schema_name = schema.name

    if schema_name not in version_map:
        return False

    entries = version_map[schema_name]

    possible_overloads = []
    possible_schemas = []
    for key, upgrader_schema_entries in entries.items():
        possible_overloads.append(key)
        possible_schemas.extend(upgrader_schema_entries)

    # let's make sure this existing schema is part of possible
    # schemas
    for old_schema in possible_schemas:
        if old_schema == schema:
            return True

    return False

def dont_parse(schema_line):
    for item in dont_parse_list:
        if item[1] < datetime.date.today():
            continue
        regexp = re.compile(item[0])
        if regexp.search(schema_line):
            return True
    return False

def load_schemas_to_dict():
    new_schemas = torch._C._jit_get_all_schemas()
    new_schemas += torch._C._jit_get_custom_class_schemas()
    new_schema_dict = defaultdict(list)
    for s in new_schemas:
        new_schema_dict[s.name].append(s)
    return new_schema_dict

def process_version_map(version_map):
    # version map maps full schema name to
    # list of upgraders. Since we only have
    # the name of the schema (aka no overload)
    # we want to first process the map to make
    # the key lookup easier. After this it will be:
    # Dict[schema_name, Dict[overload, List[schema]]]

    output = defaultdict(dict)
    for (key, entries) in version_map.items():
        operator_name = key.split(".")[0]
        schema_entries = [parse_schema(entry.old_schema) for entry in entries]
        output[operator_name][key] = schema_entries
    return output

def check_bc(existing_schemas):
    new_schema_dict = load_schemas_to_dict()
    version_map = process_version_map(torch._C._get_operator_version_map())
    is_bc = True
    broken_ops = []
    for existing_schema in existing_schemas:
        if allow_listed(existing_schema):
            print("schema: ", str(existing_schema), " found on allowlist, skipping")
            continue
        if has_valid_upgraders(existing_schema, version_map):
            print("schema: ", str(existing_schema), " has valid upgrader, skipping")
            continue
        print("processing existing schema: ", str(existing_schema))
        matching_new_schemas = new_schema_dict.get(existing_schema.name, [])
        found = False
        for matching_new_schema in matching_new_schemas:
            if matching_new_schema.is_backward_compatible_with(existing_schema):
                found = True
                break
        if not found:
            print(
                "Can NOT find backward compatible schemas after changes "
                "for schema {} from the following candidates:\n[\n{}\n]".format(
                    str(existing_schema),
                    "\n\t".join(str(s) for s in matching_new_schemas),
                )
            )
            # TODO Print out more details about why candidates don't match.
            broken_ops.append(str(existing_schema))
            is_bc = False
    if is_bc:
        print("Found backward compatible schemas for all existing schemas")
    else:
        print(
            "The PR is introducing backward incompatible changes to the "
            "operator library. Please contact PyTorch team to confirm "
            "whether this change is wanted or not. \n\nBroken ops: "
            "[\n\t{}\n]".format("\n\t".join(broken_ops))
        )
    return is_bc

def check_fc(existing_schemas):
    new_schema_dict = load_schemas_to_dict()
    is_fc = True
    broken_ops = []
    for existing_schema in existing_schemas:
        if allow_listed(existing_schema):
            print("schema: ", str(existing_schema), " found on allowlist, skipping")
            continue
        print("processing existing schema: ", str(existing_schema))
        matching_new_schemas = new_schema_dict.get(existing_schema.name, [])
        found = False
        possible_failure_reasons = []
        for matching_new_schema in matching_new_schemas:
            is_compatible, reason = matching_new_schema.check_forward_compatible_with(existing_schema)
            if is_compatible:
                found = True
                break
            if reason != "":
                possible_failure_reasons.append(reason)
        if not found:
            print(
                "Can NOT find forward compatible schemas after changes "
                "for schema {} from the following candidates:\n[\n{}\n]".format(
                    str(existing_schema),
                    "\n\t".join(str(s) for s in matching_new_schemas),
                )
            )
            print(
                "Refer to following reasons for failure "
                "to find FC schema:\n[\n{}\n]".format(
                    "\n\t".join(str(r) for r in possible_failure_reasons)
                )
            )
            broken_ops.append(str(existing_schema))
            is_fc = False
    if is_fc:
        print("Found forward compatible schemas for all existing schemas")
    else:
        warnings.warn(
            "The PR is introducing a potentially forward incompatible changes to the "
            "operator library. Please contact PyTorch team to confirm "
            "whether this change is wanted or not. \n\nBroken ops: "
            "[\n\t{}\n]".format("\n\t".join(broken_ops)), stacklevel=TO_BE_DETERMINED
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--existing-schemas",
        help="filename to load existing schemas",
        type=str,
        default="schemas.txt",
    )
    args = parser.parse_args()
    existing_schema_dict = {}
    slist = []
    with open(args.existing_schemas) as f:
        while True:
            line = f.readline()
            if not line:
                break

            if dont_parse(line.strip()):
                print("Not parsing schema line: ", line.strip())
                continue
            s = parse_schema(line.strip())
            slist.append(s)

    # TODO in case there is FC breaking changes,
    # we just warn for now until there is a policy.
    check_fc(slist)

    if not check_bc(slist):
        sys.exit(1)
