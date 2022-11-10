
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Operators.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at { namespace functorch {

#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  OP_DECOMPOSE(alpha_dropout_);
  OP_DECOMPOSE(dropout_);
  OP_DECOMPOSE(feature_alpha_dropout_);
  OP_DECOMPOSE(feature_dropout_);
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
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
  OP_DECOMPOSE(arctan2);
  OP_DECOMPOSE(avg_pool1d);
  OP_DECOMPOSE(adaptive_max_pool1d);
  OP_DECOMPOSE(adaptive_avg_pool1d);
  m.impl("adaptive_avg_pool2d", native::adaptive_avg_pool2d_symint);
  OP_DECOMPOSE(adaptive_avg_pool3d);
  OP_DECOMPOSE(adjoint);
  OP_DECOMPOSE(arccos);
  OP_DECOMPOSE(arccosh);
  OP_DECOMPOSE(arcsin);
  OP_DECOMPOSE(arcsinh);
  OP_DECOMPOSE(arctan);
  OP_DECOMPOSE(arctanh);
  OP_DECOMPOSE(atleast_1d);
  OP_DECOMPOSE2(atleast_1d, Sequence);
  OP_DECOMPOSE(atleast_2d);
  OP_DECOMPOSE2(atleast_2d, Sequence);
  OP_DECOMPOSE(atleast_3d);
  OP_DECOMPOSE2(atleast_3d, Sequence);
  OP_DECOMPOSE(batch_norm);
  OP_DECOMPOSE2(bitwise_or, Scalar);
  OP_DECOMPOSE2(bitwise_xor, Scalar);
  OP_DECOMPOSE(broadcast_tensors);
  m.impl("broadcast_to", native::broadcast_to_symint);
  OP_DECOMPOSE(cartesian_prod);
  OP_DECOMPOSE(cdist);
  OP_DECOMPOSE(clip);
  OP_DECOMPOSE2(clip, Tensor );
  OP_DECOMPOSE(concat);
  OP_DECOMPOSE(conj_physical);
  OP_DECOMPOSE(combinations);
  OP_DECOMPOSE(corrcoef);
  OP_DECOMPOSE(cosine_embedding_loss);
  OP_DECOMPOSE(cosine_similarity);
  OP_DECOMPOSE(cov);
  m.impl("cross_entropy_loss", native::cross_entropy_loss_symint);
  OP_DECOMPOSE2(cumulative_trapezoid, x);
  OP_DECOMPOSE2(cumulative_trapezoid, dx);
  OP_DECOMPOSE2(dsplit, int);
  OP_DECOMPOSE2(dsplit, array);
  OP_DECOMPOSE(det);
  OP_DECOMPOSE(diff);
  OP_DECOMPOSE(dstack);
  OP_DECOMPOSE(einsum);
  m.impl("embedding_backward", native::embedding_backward_symint);
  OP_DECOMPOSE(expand_as);
  OP_DECOMPOSE(fft_fft);
  OP_DECOMPOSE(fft_fftshift);
  OP_DECOMPOSE(fft_fft2);
  OP_DECOMPOSE(fft_fftn);
  OP_DECOMPOSE(fft_hfft);
  OP_DECOMPOSE(fft_hfft2);
  OP_DECOMPOSE(fft_hfftn);
  OP_DECOMPOSE(fft_ifft);
  OP_DECOMPOSE(fft_ifftshift);
  OP_DECOMPOSE(fft_ifft2);
  OP_DECOMPOSE(fft_ifftn);
  OP_DECOMPOSE(fft_ihfft);
  OP_DECOMPOSE(fft_irfft);
  OP_DECOMPOSE(fft_irfft2);
  OP_DECOMPOSE(fft_irfftn);
  OP_DECOMPOSE(fft_rfft);
  OP_DECOMPOSE(fft_rfft2);
  OP_DECOMPOSE(fft_rfftn);
  OP_DECOMPOSE(fix);
  OP_DECOMPOSE(fliplr);
  OP_DECOMPOSE(flipud);
  OP_DECOMPOSE2(float_power, Tensor_Tensor);
  OP_DECOMPOSE2(float_power, Tensor_Scalar);
  OP_DECOMPOSE(ger);
  OP_DECOMPOSE2(gradient, scalarint);
  OP_DECOMPOSE2(gradient, scalararray);
  OP_DECOMPOSE2(gradient, array);
  OP_DECOMPOSE2(gradient, scalarrayint);
  OP_DECOMPOSE2(gradient, scalarrayarray);
  OP_DECOMPOSE2(gradient, tensorarrayint);
  OP_DECOMPOSE2(gradient, tensorarray);
  OP_DECOMPOSE2(greater_equal, Tensor );
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
  OP_DECOMPOSE(concatenate);
  OP_DECOMPOSE(instance_norm);
  OP_DECOMPOSE(kron);
  OP_DECOMPOSE(l1_loss);
  OP_DECOMPOSE(layer_norm);
  OP_DECOMPOSE2(ldexp, Tensor);
  OP_DECOMPOSE2(less_equal, Tensor );
  OP_DECOMPOSE2(less, Tensor );
  OP_DECOMPOSE(linalg_cond);
  OP_DECOMPOSE(linalg_cholesky);
  OP_DECOMPOSE(linalg_det);
  OP_DECOMPOSE(linalg_eigvalsh);
  OP_DECOMPOSE(linalg_eigvals);
  OP_DECOMPOSE(linalg_inv);
  OP_DECOMPOSE(linalg_matmul);
  OP_DECOMPOSE(linalg_matrix_norm);
  OP_DECOMPOSE2(linalg_matrix_norm, str_ord);
  OP_DECOMPOSE(linalg_multi_dot);
  OP_DECOMPOSE(linalg_norm);
  OP_DECOMPOSE2(linalg_norm, ord_str);
  OP_DECOMPOSE(linalg_solve);
  OP_DECOMPOSE(linalg_solve_ex);
  OP_DECOMPOSE(linalg_svd);
  OP_DECOMPOSE(linalg_svdvals);
  OP_DECOMPOSE(linalg_tensorinv);
  OP_DECOMPOSE(_lu_with_info);
  OP_DECOMPOSE(matmul);
  OP_DECOMPOSE(matrix_H);
  OP_DECOMPOSE(matrix_power);
  OP_DECOMPOSE2(max, other );
  OP_DECOMPOSE(max_pool1d_with_indices);
  OP_DECOMPOSE(max_pool2d);
  OP_DECOMPOSE(meshgrid);
  OP_DECOMPOSE2(meshgrid, indexing);
  OP_DECOMPOSE(mH);
  OP_DECOMPOSE2(min, other );
  OP_DECOMPOSE2(moveaxis, intlist);
  OP_DECOMPOSE2(movedim, int);
  OP_DECOMPOSE(msort);
  OP_DECOMPOSE(mT);
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
  OP_DECOMPOSE2(repeat_interleave, self_int);
  OP_DECOMPOSE2(repeat_interleave, self_Tensor);
  m.impl("reshape", native::reshape_symint);
  OP_DECOMPOSE(resolve_conj);
  OP_DECOMPOSE(resolve_neg);
  OP_DECOMPOSE(row_stack);
  OP_DECOMPOSE(rrelu);
  OP_DECOMPOSE2(softmax, int);
  OP_DECOMPOSE(special_gammainc);
  OP_DECOMPOSE(special_gammaincc);
  OP_DECOMPOSE(special_logit);
  OP_DECOMPOSE(special_log_softmax);
  OP_DECOMPOSE(special_logsumexp);
  OP_DECOMPOSE(special_multigammaln);
  OP_DECOMPOSE(special_polygamma);
  OP_DECOMPOSE(special_softmax);
  m.impl("split.sizes", native::split_symint);
  OP_DECOMPOSE(square);
  OP_DECOMPOSE(numpy_T);
  OP_DECOMPOSE(reshape_as);
  OP_DECOMPOSE(slogdet);
  OP_DECOMPOSE(t);
  OP_DECOMPOSE2(result_type, Tensor);
  OP_DECOMPOSE2(result_type, Scalar);
  OP_DECOMPOSE2(result_type, Scalar_Tensor);
  OP_DECOMPOSE2(result_type, Scalar_Scalar);
  OP_DECOMPOSE(is_same_size);
  OP_DECOMPOSE(view_as);
  OP_DECOMPOSE2(size, int);
  OP_DECOMPOSE(is_complex);
  OP_DECOMPOSE(std);
  OP_DECOMPOSE2(std, dim);
  OP_DECOMPOSE(std_mean);
  OP_DECOMPOSE2(std_mean, dim);
  OP_DECOMPOSE(swapaxes);
  OP_DECOMPOSE2(subtract, Tensor);
  OP_DECOMPOSE(sum_to_size);
  OP_DECOMPOSE(svd);
  OP_DECOMPOSE(swapdims);
  OP_DECOMPOSE(take_along_dim);
  OP_DECOMPOSE(tensordot);
  OP_DECOMPOSE(tile);
  OP_DECOMPOSE2(trapezoid, x);
  OP_DECOMPOSE2(trapezoid, dx);
  OP_DECOMPOSE2(trapz, x);
  OP_DECOMPOSE2(trapz, dx);
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
  OP_DECOMPOSE(orgqr);
  OP_DECOMPOSE2(unflatten, int);
  m.impl("_convolution_double_backward", native::_convolution_double_backward);
  OP_DECOMPOSE(conv_transpose1d);
  OP_DECOMPOSE2(conv_transpose2d, input);
  OP_DECOMPOSE2(conv_transpose3d, input);
  OP_DECOMPOSE(conv1d);
  OP_DECOMPOSE(conv2d);
  OP_DECOMPOSE(conv3d);
  OP_DECOMPOSE2(conv1d, padding);
  OP_DECOMPOSE2(conv2d, padding);
  OP_DECOMPOSE2(conv3d, padding);
  OP_DECOMPOSE(_convolution_mode);
  OP_DECOMPOSE(frobenius_norm);
  OP_DECOMPOSE(type_as);
  OP_DECOMPOSE(linalg_diagonal);
  OP_DECOMPOSE(diagonal_copy);
  m.impl("pad", native::pad_symint);
  m.impl("_pad_circular", native::_pad_circular_symint);
  OP_DECOMPOSE(t_);
  OP_DECOMPOSE(swapdims_);
  OP_DECOMPOSE(swapaxes_);
  OP_DECOMPOSE(unfold_copy);

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
}

}}
