// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

// [start, start + 1, ..., stop - 1]
static VmapDimVector range(int64_t start, int64_t stop) {
  TORCH_INTERNAL_ASSERT(stop > start);
  VmapDimVector dims;
  dims.reserve(stop - start);
  for (int64_t i = start; i < stop; i++) {
    dims.emplace_back(i);
  }
  return dims;
}

std::tuple<Tensor,optional<int64_t>> sum_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<ScalarType> dtype) {
  if (!self_bdim.has_value()) {
    return { self.sum(dtype), nullopt };
  }
  auto self_dim = self.dim();
  if (self_dim == 1) {
    return { self.clone(), 0 };
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto dims = range(1, self_dim);
  auto result = at::sum(self_, dims, /*keepdim*/false, dtype);
  return { result, 0 };
}

bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}

std::tuple<Tensor,optional<int64_t>> sum_dim_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, IntArrayRef dims, bool keepdim, optional<ScalarType> dtype) {
  if (!self_bdim.has_value()) {
    return { at::sum(self, dims, keepdim, dtype), nullopt };
  }
  auto self_dim = self.dim();
  if (self_dim == 1 && dims.size() == 1 && is_allowed_dim_on_scalar_tensor(dims[0])) {
    return { self.clone(), 0 };
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  VmapDimVector new_dims;
  new_dims.reserve(dims.size());
  for (auto dim: dims) {
    new_dims.push_back(getPhysicalDim(self_, self_bdim.has_value(), dim));
  }
  auto result = at::sum(self_, new_dims, keepdim, dtype);
  return { result, 0 };
}

std::tuple<Tensor,optional<int64_t>> argmax_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<int64_t> dim, bool keepdim) {
  if (!self_bdim.has_value()) {
    return { at::argmax(self, dim, keepdim), nullopt };
  }
  if (self.dim() == 1 && dim && is_allowed_dim_on_scalar_tensor(*dim)) {
    return { self.clone(), 0 };
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  if (!dim) {
    dim = 0;
    self_ = at::flatten(self, 1);
  }
  auto new_dim = getPhysicalDim(self_, self_bdim.has_value(), *dim);
  auto result = at::argmax(self_, new_dim, keepdim);
  return {result, 0};
}

std::tuple<Tensor,optional<int64_t>> mean_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<ScalarType> dtype) {
  if (!self_bdim.has_value()) {
    return { self.sum(dtype), nullopt };
  }
  auto self_dim = self.dim();
  if (self_dim == 1) {
    return { self.clone(), 0 };
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto dims = range(1, self_dim);
  auto result = at::mean(self_, dims, /*keepdim*/false, dtype);
  return { result, 0 };
}

std::tuple<Tensor,optional<int64_t>>
_log_softmax_backward_data(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& output, optional<int64_t> output_bdim,
    int64_t dim,
    const Tensor& self, optional<int64_t> self_bdim) {
  TORCH_INTERNAL_ASSERT(!(output_bdim.has_value() ^ self_bdim.has_value()),
      "output_bdim and self_bdim must be the same");
  if (!grad_output_bdim && !self_bdim) {
    return { at::_log_softmax_backward_data(grad_output, output, dim, self), nullopt };
  }
  if (grad_output_bdim && self_bdim) {
    auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
    auto output_ = moveBatchDimToFront(output, output_bdim);
    auto self_ = moveBatchDimToFront(self, self_bdim);
    dim = getPhysicalDim(grad_output_, /*has_batch_dim*/true, dim);
    return { at::_log_softmax_backward_data(grad_output_, output_, dim, self_), 0 };
  }
  // NB: It turns out that expanding + calling log_softmax_backward is generally
  // faster than the decomposition.
  // Benchmark here: https://gist.github.com/zou3519/ae3b33b5730a84aae8a80a05c89e078a
  // Decomposition is (grad_output - grad_output.sum(dim, keepdim=True) * result.exp())
  // We can squeeze out a last mile of performance by writing custom kernels.
  if (grad_output_bdim && !self_bdim) {
    auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
    dim = getPhysicalDim(grad_output_, /*has_batch_dim*/true, dim);
    auto output_ = output.expand_as(grad_output_);
    auto self_ = self.expand_as(grad_output_);
    return { at::_log_softmax_backward_data(grad_output_, output_, dim, self_), 0 };
  }
  if (!grad_output_bdim && self_bdim) {
    auto output_ = moveBatchDimToFront(output, output_bdim);
    auto self_ = moveBatchDimToFront(self, self_bdim);
    auto grad_output_ = grad_output.expand_as(output_);
    dim = getPhysicalDim(grad_output_, /*has_batch_dim*/true, dim);
    return { at::_log_softmax_backward_data(grad_output_, output_, dim, self_), 0 };
  }
  TORCH_INTERNAL_ASSERT(false);
}


TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("sum", sum_batch_rule);
  VMAP_SUPPORT("sum.dim_IntList", sum_dim_batch_rule);
  VMAP_SUPPORT("argmax", argmax_batch_rule);
  VMAP_SUPPORT("mean", mean_batch_rule);
  VMAP_SUPPORT("_log_softmax_backward_data", _log_softmax_backward_data);
}

}}
