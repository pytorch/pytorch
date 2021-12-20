
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <ATen/Operators.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace functorch {

at::Tensor sync_and_unwrap_functional_output(at::Tensor out_functional) {
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(out_functional));
  auto out_wrapper_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(out_functional);
  out_wrapper_impl->sync_();
  auto out_unwrapped = out_wrapper_impl->value();
  return out_unwrapped;
}

c10::List<at::Tensor> sync_and_unwrap_functional_output(const c10::List<at::Tensor>& t_list) {
  c10::List<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(sync_and_unwrap_functional_output(t_list[i]));
  }
  return outputs;
}

void decompose_functional(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();

  const auto num_arguments = schema.arguments().size();
  const auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;
  //
  // Step 1: Wrap any tensor inputs into Functional tensors
  // and put them on the stack at the correct indices.
  for (const auto idx : c10::irange(arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      auto functional_ivalue = at::functionalization::impl::to_functional_tensor(ivalue.toTensor());
      (*stack)[arguments_begin + idx] = std::move(functional_ivalue);
    } else if (ivalue.isTensorList()) {
      auto functional_ivalue = at::functionalization::impl::to_functional_tensor(ivalue.toTensorList());
      (*stack)[arguments_begin + idx] = std::move(functional_ivalue);
    }
  }

  // Step 2: set up TLS such that we hit the functionalization kernels before the batching rules.
  // Note: this relies on the fact that Functionalization > BatchMode in DispatchKey.h
  c10::impl::IncludeDispatchKeyGuard include_guard(c10::DispatchKeySet(c10::DispatchKey::Functionalize));

  // Step 3: redispatch to native kernel
  // TODO: this is technically kind of sketchy, since we're relying on the fact
  // that the composite kernel is registered to a particular dispatch key.
  // In reality, a C++ extension could register their own custom kernels to any dispatch key, which would override
  // the composite kernel entry.
  // I'm using CPU because C++ extensions that register custom kernels to existing composite operators are pretty uncommon,
  // and only really matter for out-of-tree keys like XLA.
  // I wonder if we should make "alias dispatch key kernels" a runtime-accessible property on the OperatorHandle?
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);

  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

  // Step 4: Unwrap each functional output tensor, syncing any pending updates
  for (const auto idx : c10::irange(returns.size())) {
    if (returns[idx].isTensor()) {
      const auto& out_functional = returns[idx].toTensor();
      auto out_unwrapped = sync_and_unwrap_functional_output(out_functional);
      (*stack)[returns_begin + idx] = c10::IValue(out_unwrapped);
    } else if (returns[idx].isTensorList()) {
      const auto& out_functional = returns[idx].toTensorList();
      auto out_unwrapped = sync_and_unwrap_functional_output(out_functional);
      (*stack)[returns_begin + idx] = c10::IValue(out_unwrapped);
    }
  }
}

#define DECOMPOSE_FUNCTIONAL(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&decompose_functional>());


#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
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
  OP_DECOMPOSE(index_select_backward);
  OP_DECOMPOSE(absolute);
  OP_DECOMPOSE(avg_pool1d);
  OP_DECOMPOSE(adaptive_max_pool1d);
  OP_DECOMPOSE(adaptive_avg_pool1d);
  OP_DECOMPOSE(adaptive_avg_pool2d);
  OP_DECOMPOSE(adaptive_avg_pool3d);
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
  OP_DECOMPOSE2(bitwise_or, Scalar);
  OP_DECOMPOSE2(bitwise_xor, Scalar);
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
  OP_DECOMPOSE(nll_loss_nd);
  OP_DECOMPOSE(nll_loss);
  OP_DECOMPOSE(nll_loss2d);
  OP_DECOMPOSE2(not_equal, Tensor );
  OP_DECOMPOSE(outer);
  OP_DECOMPOSE(pairwise_distance);
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
  OP_DECOMPOSE2(unflatten, int);
  OP_DECOMPOSE(cross_entropy_loss);
  OP_DECOMPOSE(arctan2);
  OP_DECOMPOSE(layer_norm);
  OP_DECOMPOSE(diag_backward);
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
  OP_DECOMPOSE(embedding_backward);
  DECOMPOSE_FUNCTIONAL(diag_embed);
  DECOMPOSE_FUNCTIONAL(block_diag);
}

}}

