
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at::functorch {

#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  OP_DECOMPOSE(alpha_dropout_);
  OP_DECOMPOSE(dropout_);
  OP_DECOMPOSE(feature_alpha_dropout_);
  OP_DECOMPOSE(feature_dropout_);
}

static void unsupportedData(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false, "mutating directly with `.data` under vmap transform is not allowed.");
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatchedDecomposition, m) {
  OP_DECOMPOSE2(__and__, Scalar);
  OP_DECOMPOSE2(__and__, Tensor);
  OP_DECOMPOSE2(__iand__, Tensor);
  OP_DECOMPOSE2(__iand__, Scalar);
  OP_DECOMPOSE2(__ior__, Tensor);
  OP_DECOMPOSE2(__ior__, Scalar);
  OP_DECOMPOSE2(__ixor__, Tensor);
  OP_DECOMPOSE2(__ixor__, Scalar);
  OP_DECOMPOSE2(__or__, Tensor);
  OP_DECOMPOSE2(__or__, Scalar);
  OP_DECOMPOSE2(__xor__, Tensor);
  OP_DECOMPOSE2(__xor__, Scalar);
  OP_DECOMPOSE(_batch_norm_impl_index);
  OP_DECOMPOSE(absolute);
  OP_DECOMPOSE(absolute_);
  OP_DECOMPOSE(arctan2);
  OP_DECOMPOSE(arctan2_);
  OP_DECOMPOSE(argsort);
  OP_DECOMPOSE(avg_pool1d);
  OP_DECOMPOSE(adaptive_max_pool1d);
  OP_DECOMPOSE(adaptive_avg_pool1d);
  m.impl("adaptive_avg_pool2d", native::adaptive_avg_pool2d_symint);
  m.impl("adaptive_avg_pool3d", native::adaptive_avg_pool3d_symint);
  OP_DECOMPOSE(adjoint);
  OP_DECOMPOSE(arccos);
  OP_DECOMPOSE(arccos_);
  OP_DECOMPOSE(arccosh);
  OP_DECOMPOSE(arccosh_);
  OP_DECOMPOSE(arcsin);
  OP_DECOMPOSE(arcsin_);
  OP_DECOMPOSE(arcsinh);
  OP_DECOMPOSE(arcsinh_);
  OP_DECOMPOSE(arctan);
  OP_DECOMPOSE(arctan_);
  OP_DECOMPOSE(arctanh);
  OP_DECOMPOSE(arctanh_);
  OP_DECOMPOSE(atleast_1d);
  OP_DECOMPOSE2(atleast_1d, Sequence);
  OP_DECOMPOSE(atleast_2d);
  OP_DECOMPOSE2(atleast_2d, Sequence);
  OP_DECOMPOSE(atleast_3d);
  OP_DECOMPOSE2(atleast_3d, Sequence);
  OP_DECOMPOSE(batch_norm);
  OP_DECOMPOSE(broadcast_tensors);
  m.impl("broadcast_to", native::broadcast_to_symint);
  OP_DECOMPOSE(cartesian_prod);
  OP_DECOMPOSE(cdist);
  OP_DECOMPOSE(chunk);
  OP_DECOMPOSE(clip);
  OP_DECOMPOSE2(clip, Tensor );
  OP_DECOMPOSE(concat);
  OP_DECOMPOSE(conj_physical);
  OP_DECOMPOSE(contiguous);
  OP_DECOMPOSE(combinations);
  OP_DECOMPOSE(corrcoef);
  OP_DECOMPOSE(cosine_embedding_loss);
  OP_DECOMPOSE(cosine_similarity);
  OP_DECOMPOSE(cov);
  OP_DECOMPOSE(cross);
  m.impl("cross_entropy_loss", native::cross_entropy_loss_symint);
  OP_DECOMPOSE2(cumulative_trapezoid, x);
  OP_DECOMPOSE2(cumulative_trapezoid, dx);
  OP_DECOMPOSE2(dsplit, int);
  OP_DECOMPOSE2(dsplit, array);
  OP_DECOMPOSE(det);
  OP_DECOMPOSE(diff);
  OP_DECOMPOSE(diag);
  OP_DECOMPOSE(dstack);
  OP_DECOMPOSE(einsum);
  m.impl("embedding_backward", native::embedding_backward_symint);
  OP_DECOMPOSE(expand_as);
  m.impl("fft_fft", native::fft_fft_symint);
  OP_DECOMPOSE(fft_fftshift);
  m.impl("fft_fft2", native::fft_fft2_symint);
  m.impl("fft_fftn", native::fft_fftn_symint);
  m.impl("fft_hfft", native::fft_hfft_symint);
  m.impl("fft_hfft2", native::fft_hfft2_symint);
  m.impl("fft_hfftn", native::fft_hfftn_symint);
  m.impl("fft_ifft", native::fft_ifft_symint);
  OP_DECOMPOSE(fft_ifftshift);
  m.impl("fft_ifft2", native::fft_ifft2_symint);
  m.impl("fft_ifftn", native::fft_ifftn_symint);
  m.impl("fft_ihfft", native::fft_ihfft_symint);
  m.impl("fft_irfft", native::fft_irfft_symint);
  m.impl("fft_irfft2", native::fft_irfft2_symint);
  m.impl("fft_irfftn", native::fft_irfftn_symint);
  m.impl("fft_rfft", native::fft_rfft_symint);
  m.impl("fft_rfft2", native::fft_rfft2_symint);
  m.impl("fft_rfftn", native::fft_rfftn_symint);
  OP_DECOMPOSE(fix);
  OP_DECOMPOSE(fliplr);
  OP_DECOMPOSE(flipud);
  OP_DECOMPOSE2(flatten, using_ints);
  OP_DECOMPOSE2(float_power, Tensor_Tensor);
  OP_DECOMPOSE2(float_power, Tensor_Scalar);
  OP_DECOMPOSE2(float_power, Scalar);
  OP_DECOMPOSE(gather_backward);
  OP_DECOMPOSE(ger);
  OP_DECOMPOSE2(gradient, scalarint);
  OP_DECOMPOSE2(gradient, scalararray);
  OP_DECOMPOSE2(gradient, array);
  OP_DECOMPOSE2(gradient, scalarrayint);
  OP_DECOMPOSE2(gradient, scalarrayarray);
  OP_DECOMPOSE2(gradient, tensorarrayint);
  OP_DECOMPOSE2(gradient, tensorarray);
  OP_DECOMPOSE2(greater_equal, Tensor );
  OP_DECOMPOSE2(greater_equal, Scalar );
  OP_DECOMPOSE2(greater, Tensor );
  OP_DECOMPOSE(grid_sampler);
  OP_DECOMPOSE(group_norm);
  OP_DECOMPOSE(hinge_embedding_loss);
  OP_DECOMPOSE2(hsplit, int);
  OP_DECOMPOSE2(hsplit, array);
  OP_DECOMPOSE(hstack);
  m.impl("index_select_backward", native::index_select_backward_symint);
  OP_DECOMPOSE(inner);
  OP_DECOMPOSE(inverse);
  OP_DECOMPOSE(isfinite);
  OP_DECOMPOSE(isreal);
  OP_DECOMPOSE(concatenate);
  OP_DECOMPOSE(instance_norm);
  OP_DECOMPOSE(kron);
  OP_DECOMPOSE(l1_loss);
  m.impl("layer_norm", native::layer_norm_symint);
  OP_DECOMPOSE2(ldexp, Tensor);
  OP_DECOMPOSE2(less_equal, Tensor );
  OP_DECOMPOSE2(less, Tensor );
  OP_DECOMPOSE(linear);
  OP_DECOMPOSE(linalg_cond);
  OP_DECOMPOSE(linalg_cholesky);
  OP_DECOMPOSE(linalg_det);
  OP_DECOMPOSE(linalg_eigvalsh);
  OP_DECOMPOSE(linalg_eigvals);
  OP_DECOMPOSE(linalg_inv);
  OP_DECOMPOSE(linalg_lu_factor);
  OP_DECOMPOSE(linalg_matmul);
  OP_DECOMPOSE(linalg_matrix_norm);
  OP_DECOMPOSE2(linalg_matrix_norm, str_ord);
  OP_DECOMPOSE(linalg_multi_dot);
  OP_DECOMPOSE(linalg_norm);
  OP_DECOMPOSE2(linalg_norm, ord_str);
  OP_DECOMPOSE(linalg_eigh);
  OP_DECOMPOSE(linalg_solve);
  OP_DECOMPOSE(linalg_solve_ex);
  OP_DECOMPOSE(linalg_svd);
  OP_DECOMPOSE(linalg_svdvals);
  OP_DECOMPOSE(linalg_pinv);
  OP_DECOMPOSE(linalg_tensorinv);
  OP_DECOMPOSE2(linalg_pinv, atol_rtol_float);
  m.impl("linalg_vander", native::linalg_vander_symint);
  OP_DECOMPOSE(cumprod_backward);
  OP_DECOMPOSE(linalg_matrix_power);
  OP_DECOMPOSE(linalg_vecdot);
  OP_DECOMPOSE(log_sigmoid);
  OP_DECOMPOSE(logdet);
  OP_DECOMPOSE2(log_softmax, int);
  OP_DECOMPOSE(_lu_with_info);
  OP_DECOMPOSE(matmul);
  OP_DECOMPOSE(matrix_H);
  OP_DECOMPOSE(matrix_power);
  OP_DECOMPOSE2(max, other );
  OP_DECOMPOSE(max_pool1d);
  OP_DECOMPOSE(max_pool1d_with_indices);
  OP_DECOMPOSE(max_pool2d);
  OP_DECOMPOSE(max_pool3d);
  OP_DECOMPOSE(meshgrid);
  OP_DECOMPOSE2(meshgrid, indexing);
  OP_DECOMPOSE(mH);
  OP_DECOMPOSE2(min, other );
  OP_DECOMPOSE2(moveaxis, intlist);
  OP_DECOMPOSE2(movedim, int);
  OP_DECOMPOSE2(movedim, intlist);
  OP_DECOMPOSE(msort);
  OP_DECOMPOSE(mT);
  OP_DECOMPOSE(nanmean);
  m.impl("narrow", native::narrow_symint);
  OP_DECOMPOSE(negative);
  OP_DECOMPOSE2(frobenius_norm, dim);
  OP_DECOMPOSE2(nuclear_norm, dim);
  OP_DECOMPOSE(nuclear_norm);
  m.impl("nll_loss_nd", native::nll_loss_nd_symint);
  m.impl("nll_loss", native::nll_loss_symint);
  m.impl("nll_loss2d", native::nll_loss2d_symint);
  OP_DECOMPOSE2(not_equal, Tensor );
  OP_DECOMPOSE(outer);
  OP_DECOMPOSE(pairwise_distance);
  OP_DECOMPOSE(pinverse);
  OP_DECOMPOSE(poisson_nll_loss);
  OP_DECOMPOSE(positive);
  OP_DECOMPOSE(qr);
  OP_DECOMPOSE(ravel);
  m.impl("repeat_interleave.self_int", static_cast<decltype(&ATEN_FN2(repeat_interleave, self_int))>(native::repeat_interleave_symint));
  m.impl("repeat_interleave.self_Tensor", static_cast<decltype(&ATEN_FN2(repeat_interleave, self_Tensor))>(native::repeat_interleave_symint));
  m.impl("reshape", native::reshape_symint);
  OP_DECOMPOSE(resolve_conj);
  OP_DECOMPOSE(resolve_neg);
  OP_DECOMPOSE(rms_norm);
  OP_DECOMPOSE(row_stack);
  OP_DECOMPOSE(rrelu);
  OP_DECOMPOSE(rrelu_);
  OP_DECOMPOSE(relu6);
  OP_DECOMPOSE(relu6_);
  OP_DECOMPOSE(prelu);
  OP_DECOMPOSE2(softmax, int);
  OP_DECOMPOSE(scaled_dot_product_attention);
  OP_DECOMPOSE(special_gammainc);
  OP_DECOMPOSE(special_gammaincc);
  OP_DECOMPOSE(special_logit);
  OP_DECOMPOSE(special_log_softmax);
  OP_DECOMPOSE(special_logsumexp);
  OP_DECOMPOSE(special_multigammaln);
  OP_DECOMPOSE(special_polygamma);
  OP_DECOMPOSE(special_softmax);
  OP_DECOMPOSE(special_digamma);
  OP_DECOMPOSE(special_erf);
  OP_DECOMPOSE(special_erfc);
  OP_DECOMPOSE(special_erfinv);
  OP_DECOMPOSE(special_exp2);
  OP_DECOMPOSE(special_expm1);
  OP_DECOMPOSE(special_expit);
  OP_DECOMPOSE(special_gammaln);
  OP_DECOMPOSE(special_i0);
  OP_DECOMPOSE(special_log1p);
  OP_DECOMPOSE(special_ndtr);
  OP_DECOMPOSE(special_psi);
  OP_DECOMPOSE(special_round);
  OP_DECOMPOSE(special_sinc);
  OP_DECOMPOSE(special_xlogy);
  OP_DECOMPOSE2(special_xlogy, other_scalar);
  OP_DECOMPOSE2(special_xlogy, self_scalar);


  m.impl("split.sizes", native::split_symint);
  OP_DECOMPOSE(square);
  OP_DECOMPOSE(numpy_T);
  OP_DECOMPOSE(reshape_as);
  OP_DECOMPOSE(slogdet);
  OP_DECOMPOSE2(result_type, Tensor);
  OP_DECOMPOSE2(result_type, Scalar);
  OP_DECOMPOSE2(result_type, Scalar_Tensor);
  OP_DECOMPOSE2(result_type, Scalar_Scalar);
  OP_DECOMPOSE(is_same_size);
  OP_DECOMPOSE(view_as);
  OP_DECOMPOSE2(size, int);
  OP_DECOMPOSE(is_complex);
  OP_DECOMPOSE(std);
  OP_DECOMPOSE(selu);
  OP_DECOMPOSE(selu_);
  OP_DECOMPOSE2(std, dim);
  OP_DECOMPOSE(std_mean);
  OP_DECOMPOSE2(std_mean, dim);
  OP_DECOMPOSE(swapaxes);
  OP_DECOMPOSE2(subtract, Tensor);
  m.impl("sum_to_size", native::sum_to_size_symint);
  OP_DECOMPOSE(svd);
  OP_DECOMPOSE(swapdims);
  OP_DECOMPOSE(take_along_dim);
  OP_DECOMPOSE(tensordot);
  m.impl("tensor_split.indices", native::tensor_split_indices_symint);
  m.impl("tensor_split.sections", native::tensor_split_sections_symint);
  OP_DECOMPOSE(_test_check_tensor);
  m.impl("tile", native::tile_symint);
  OP_DECOMPOSE2(trapezoid, x);
  OP_DECOMPOSE2(trapezoid, dx);
  OP_DECOMPOSE2(trapz, x);
  OP_DECOMPOSE2(trapz, dx);
  OP_DECOMPOSE(unsafe_chunk);
  m.impl("value_selecting_reduction_backward", native::value_selecting_reduction_backward_symint);
  OP_DECOMPOSE(var);
  OP_DECOMPOSE2(var, dim);
  OP_DECOMPOSE(var_mean);
  OP_DECOMPOSE2(var_mean, dim);
  OP_DECOMPOSE2(vsplit, int);
  OP_DECOMPOSE2(vsplit, array);
  OP_DECOMPOSE(vstack);
  OP_DECOMPOSE2(where, ScalarOther);
  OP_DECOMPOSE2(where, ScalarSelf);
  OP_DECOMPOSE2(where, Scalar);
  OP_DECOMPOSE(orgqr);
  m.impl("unflatten.int", native::unflatten_symint);
  m.impl("_convolution_double_backward", native::_convolution_double_backward);
  m.impl("conv_transpose1d", native::conv_transpose1d_symint);
  m.impl("conv_transpose2d.input", native::conv_transpose2d_symint);
  m.impl("conv_transpose3d.input", native::conv_transpose3d_symint);
  m.impl("conv1d", native::conv1d_symint);
  m.impl("conv2d", native::conv2d_symint);
  m.impl("conv3d", native::conv3d_symint);
  m.impl("conv1d.padding", native::conv1d_padding_symint);
  m.impl("conv2d.padding", native::conv2d_padding_symint);
  m.impl("conv3d.padding", native::conv3d_padding_symint);
  m.impl("_convolution_mode", native::_convolution_mode_symint);
  OP_DECOMPOSE(type_as);
  OP_DECOMPOSE(linalg_diagonal);
  OP_DECOMPOSE(diagonal_copy);
  OP_DECOMPOSE(alias_copy);
  m.impl("pad", native::pad_symint);
  m.impl("_pad_circular", native::_pad_circular_symint);
  OP_DECOMPOSE(swapdims_);
  OP_DECOMPOSE(swapaxes_);
  OP_DECOMPOSE(unfold_copy);
  // Easy way to decompose upsample*.vec overloads instead of introducing *_symint methods
  // if used OP_DECOMPOSE2.
  m.impl("upsample_bilinear2d.vec", native::upsample_bilinear2d);
  m.impl("upsample_bicubic2d.vec", native::upsample_bicubic2d);
  m.impl("_upsample_bilinear2d_aa.vec", native::_upsample_bilinear2d_aa);
  m.impl("_upsample_bicubic2d_aa.vec", native::_upsample_bicubic2d_aa);
  m.impl("upsample_linear1d.vec", native::upsample_linear1d);
  m.impl("upsample_nearest1d.vec", native::upsample_nearest1d);
  m.impl("upsample_nearest2d.vec", native::upsample_nearest2d);
  m.impl("upsample_nearest3d.vec", native::upsample_nearest3d);
  m.impl("upsample_trilinear3d.vec", native::upsample_trilinear3d);

  // views on complex tensor
  OP_DECOMPOSE(imag);
  OP_DECOMPOSE(real);

  // divide, alias for div
  OP_DECOMPOSE2(divide, Tensor);
  OP_DECOMPOSE2(divide_, Tensor);
  OP_DECOMPOSE2(divide, Scalar);
  OP_DECOMPOSE2(divide, Tensor_mode);
  OP_DECOMPOSE2(divide_, Tensor_mode);
  OP_DECOMPOSE2(divide, Scalar_mode);
  OP_DECOMPOSE2(divide_, Scalar_mode);

  // divide, alias for div
  OP_DECOMPOSE2(true_divide, Tensor);
  OP_DECOMPOSE2(true_divide_, Tensor);
  OP_DECOMPOSE2(true_divide, Scalar);
  OP_DECOMPOSE2(true_divide_, Scalar);

  // multiply, alias for mul
  OP_DECOMPOSE2(multiply, Tensor)
  OP_DECOMPOSE2(multiply_, Tensor)
  OP_DECOMPOSE2(multiply, Scalar)
  OP_DECOMPOSE2(multiply_, Scalar)

  OP_DECOMPOSE2(linalg_matrix_rank, atol_rtol_tensor);
  OP_DECOMPOSE2(linalg_matrix_rank, atol_rtol_float);
  OP_DECOMPOSE(linalg_ldl_factor);

  // comparison ops
  OP_DECOMPOSE2(greater, Scalar);
  OP_DECOMPOSE2(less_equal, Scalar);
  OP_DECOMPOSE2(less, Scalar);
  OP_DECOMPOSE2(not_equal, Scalar);
  m.impl("_has_compatible_shallow_copy_type", torch::CppFunction::makeFromBoxedFunction<&unsupportedData>());

  // to.*
  OP_DECOMPOSE2(to, device);
  OP_DECOMPOSE2(to, dtype);
  OP_DECOMPOSE2(to, dtype_layout);
  OP_DECOMPOSE2(to, other);
}

} // namespace at::functorch
