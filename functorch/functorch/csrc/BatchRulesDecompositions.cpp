
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <ATen/Operators.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace functorch {

#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  OP_DECOMPOSE(absolute);
  OP_DECOMPOSE(adaptive_avg_pool2d);
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
  OP_DECOMPOSE(broadcast_tensors);
  OP_DECOMPOSE(broadcast_to);
  OP_DECOMPOSE(clip);
  OP_DECOMPOSE2(clip, Tensor );
  OP_DECOMPOSE(concat);
  OP_DECOMPOSE(conj_physical);
  OP_DECOMPOSE(corrcoef);
  OP_DECOMPOSE(cosine_similarity);
  OP_DECOMPOSE(cov);
  OP_DECOMPOSE2(cumulative_trapezoid, x);
  OP_DECOMPOSE2(cumulative_trapezoid, dx);
  OP_DECOMPOSE(det);
  OP_DECOMPOSE(diff);
  OP_DECOMPOSE2(divide, Tensor );
  OP_DECOMPOSE(einsum);
  OP_DECOMPOSE(expand_as);
  OP_DECOMPOSE(fft_fft);
  OP_DECOMPOSE(fft_ifft);
  OP_DECOMPOSE(fft_ihfft);
  OP_DECOMPOSE(fft_irfft);
  OP_DECOMPOSE(fft_irfftn);
  OP_DECOMPOSE(fft_rfft);
  OP_DECOMPOSE(fft_rfftn);
  OP_DECOMPOSE(fix);
  OP_DECOMPOSE(fliplr);
  OP_DECOMPOSE(flipud);
  OP_DECOMPOSE2(float_power, Tensor_Tensor);
  OP_DECOMPOSE2(float_power, Tensor_Scalar);
  OP_DECOMPOSE(ger);
  OP_DECOMPOSE2(greater_equal, Tensor );
  OP_DECOMPOSE2(greater, Tensor );
  OP_DECOMPOSE(grid_sampler);
  OP_DECOMPOSE(inner);
  OP_DECOMPOSE(kron);
  OP_DECOMPOSE2(less_equal, Tensor );
  OP_DECOMPOSE2(less, Tensor );
  OP_DECOMPOSE(linalg_cond);
  OP_DECOMPOSE(linalg_det);
  OP_DECOMPOSE(linalg_matmul);
  OP_DECOMPOSE(linalg_svd);
  OP_DECOMPOSE(matmul);
  OP_DECOMPOSE2(max, other );
  OP_DECOMPOSE(max_pool2d);
  OP_DECOMPOSE2(meshgrid, indexing);
  OP_DECOMPOSE(mH);
  OP_DECOMPOSE2(min, other );
  OP_DECOMPOSE2(moveaxis, intlist);
  OP_DECOMPOSE2(movedim, int);
  OP_DECOMPOSE(msort);
  OP_DECOMPOSE(mT);
  OP_DECOMPOSE2(multiply, Tensor );
  OP_DECOMPOSE(narrow);
  OP_DECOMPOSE(negative);
  OP_DECOMPOSE2(not_equal, Tensor );
  OP_DECOMPOSE(outer);
  OP_DECOMPOSE(qr);
  OP_DECOMPOSE(ravel);
  OP_DECOMPOSE(reshape);
  OP_DECOMPOSE(resolve_conj);
  OP_DECOMPOSE(resolve_neg);
  OP_DECOMPOSE2(softmax, int);
  OP_DECOMPOSE(special_gammainc);
  OP_DECOMPOSE(special_gammaincc);
  OP_DECOMPOSE(special_logit);
  OP_DECOMPOSE(special_log_softmax);
  OP_DECOMPOSE(special_logsumexp);
  OP_DECOMPOSE(special_multigammaln);
  OP_DECOMPOSE(special_polygamma);
  OP_DECOMPOSE(special_softmax);
  OP_DECOMPOSE(square);
  OP_DECOMPOSE(std);
  OP_DECOMPOSE2(std, dim);
  OP_DECOMPOSE(std_mean);
  OP_DECOMPOSE2(std_mean, dim);
  OP_DECOMPOSE(swapaxes);
  OP_DECOMPOSE2(subtract, Tensor);
  OP_DECOMPOSE(svd);
  OP_DECOMPOSE(swapdims);
  OP_DECOMPOSE(tensordot);
  OP_DECOMPOSE(tile);
  OP_DECOMPOSE2(trapezoid, x);
  OP_DECOMPOSE2(trapezoid, dx);
  OP_DECOMPOSE2(trapz, x);
  OP_DECOMPOSE2(trapz, dx);
  OP_DECOMPOSE2(true_divide, Tensor);
  OP_DECOMPOSE(var);
  OP_DECOMPOSE2(var, dim);
  OP_DECOMPOSE(var_mean);
  OP_DECOMPOSE2(var_mean, dim);
  OP_DECOMPOSE2(where, self);
}

}}

