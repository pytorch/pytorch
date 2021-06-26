// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <ATen/Operators.h>

namespace at { namespace functorch {

// [start, start + 1, ..., stop - 1]
static VmapDimVector range(int64_t start, int64_t stop) {
  TORCH_INTERNAL_ASSERT(stop >= start);
  VmapDimVector dims;
  dims.reserve(stop - start);
  for (int64_t i = start; i < stop; i++) {
    dims.emplace_back(i);
  }
  return dims;
}


bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}


template <typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor,optional<int64_t>> reduction_dimarray_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, IntArrayRef dims, ExtraArgs... extra_args) {
  if (!self_bdim.has_value()) {
    return std::make_tuple( Func(self, dims, std::forward<ExtraArgs>(extra_args)...), nullopt );
  }
  auto logical_dim = rankWithoutBatchDim(self, self_bdim);

  // If the dim intlist is empty, that's equivalent to passing in a dim on all dimensions.
  if (dims.size() == 0) {
    dims = range(0, std::max((int64_t)1, logical_dim));
  }

  if (logical_dim == 0 && dims.size() == 1 && is_allowed_dim_on_scalar_tensor(dims[0])) {
    return std::make_tuple( self.clone(), 0 );
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  VmapDimVector new_dims;
  new_dims.reserve(dims.size());
  for (auto dim: dims) {
    new_dims.push_back(getPhysicalDim(self_, self_bdim.has_value(), dim));
  }
  auto result = Func(self_, new_dims, std::forward<ExtraArgs>(extra_args)...);
  return std::make_tuple( result, 0 );
}

// Taken from https://stackoverflow.com/a/41301717
template<typename R, typename... A>
R ret(R(*)(A...));

// Optional implies the weird case with 0-dim tensors i.e. torch.sum(torch.randn(()), 0)
template <typename F, F Func, typename... ExtraArgs>
optional<std::tuple<decltype(ret(Func)), optional<int64_t>>> reduction_dim_batch_rule_impl(const Tensor& self, optional<int64_t> self_bdim, int64_t dim, ExtraArgs... extra_args) {
  if (!self_bdim.has_value()) {
    return std::make_tuple(Func(self, dim, std::forward<ExtraArgs>(extra_args)...), nullopt);
  }
  auto logical_dim = rankWithoutBatchDim(self, self_bdim);
  if (logical_dim == 0 && is_allowed_dim_on_scalar_tensor(dim)) {
    return nullopt;
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  int64_t new_dim = getPhysicalDim(self, self_bdim.has_value(), dim);
  auto result = Func(self_, new_dim, std::forward<ExtraArgs>(extra_args)...);
  return std::make_tuple( result, 0 );
}

template <typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor,optional<int64_t>> reduction_dim_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, int64_t dim, ExtraArgs... extra_args) {
  auto out = reduction_dim_batch_rule_impl<F, Func, ExtraArgs...>(self, self_bdim, dim, std::forward<ExtraArgs>(extra_args)...);
  if (!out) {
    return std::make_tuple( self.clone(), 0 );
  }
  return *out;
}

template <typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>> reduction_dim_ret_pair_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, int64_t dim, ExtraArgs... extra_args) {
  auto out = reduction_dim_batch_rule_impl<F, Func, ExtraArgs...>(self, self_bdim, dim, std::forward<ExtraArgs>(extra_args)...);
  if (!out) {
    return std::make_tuple(self.clone(), 0, at::zeros({self.size(0)}, {}, self.options().dtype(kLong)), 0);
  }
  auto tensors = std::get<0>(*out);
  auto bdim = std::get<1>(*out);
  return std::make_tuple(std::get<0>(tensors), bdim, std::get<1>(tensors), bdim);
}

template <typename F, F Func, typename G, G DimRule, typename... ExtraArgs>
std::tuple<Tensor,optional<int64_t>> reduction_no_dim_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, ExtraArgs... extra_args) {
  if (!self_bdim.has_value()) {
    return std::make_tuple(Func(self, std::forward<ExtraArgs>(extra_args)...), nullopt);
  }
  if (self.dim() == 1) {
    return std::make_tuple(self.clone(), 0);
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  self_ = at::flatten(self_, 1);
  auto out = DimRule(self_, 0, 0, false, std::forward<ExtraArgs>(extra_args)...);
  return std::make_tuple(std::get<0>(out), std::get<1>(out));
}

// For now I'm not macroing these (don't see another way to do it), since I'm
// worried about various edge cases popping up that make things more annoying.
// Will re-evaluate after adding more reduction batching rules

std::tuple<Tensor,optional<int64_t>> sum_dim_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, IntArrayRef dims, bool keepdim, optional<ScalarType> dtype) {
  return reduction_dimarray_batch_rule<decltype(&ATEN_FN2(sum, dim_IntList)), &at::sum, bool, optional<ScalarType>>(self, self_bdim, dims, keepdim, dtype);
}

std::tuple<Tensor,optional<int64_t>> sum_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<ScalarType> dtype) {
  return sum_dim_batch_rule(self, self_bdim, range(0, self.dim() - 1), false, dtype);
}

std::tuple<Tensor,optional<int64_t>> mean_dim_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, IntArrayRef dims, bool keepdim, optional<ScalarType> dtype) {
  return reduction_dimarray_batch_rule<decltype(&ATEN_FN2(mean, dim)), &at::mean, bool, optional<ScalarType>>(self, self_bdim, dims, keepdim, dtype);
}

std::tuple<Tensor,optional<int64_t>> mean_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<ScalarType> dtype) {
  return mean_dim_batch_rule(self, self_bdim, range(0, self.dim() - 1), false, dtype);
}

std::tuple<Tensor,optional<int64_t>> nansum_dim_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, IntArrayRef dims, bool keepdim, optional<ScalarType> dtype) {
  return reduction_dimarray_batch_rule<decltype(&ATEN_FN2(nansum, dim_IntList)), &at::nansum, bool, optional<ScalarType>>(self, self_bdim, dims, keepdim, dtype);
}

std::tuple<Tensor,optional<int64_t>> nansum_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<ScalarType> dtype) {
  return nansum_dim_batch_rule(self, self_bdim, range(0, self.dim() - 1), false, dtype);
}

// Wraps so that dim is always provided
Tensor std_correction_wrapper(const Tensor& self, IntArrayRef dim, optional<int64_t> correction, bool keepdim) {
  return at::std(self, dim, correction, keepdim);
}
std::tuple<Tensor,optional<int64_t>> std_correction_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<IntArrayRef> dim, optional<int64_t> correction, bool keepdim) {
  if (!dim.has_value()) {
    dim = range(0, self.dim() - 1);
  }
  return reduction_dimarray_batch_rule<decltype(&std_correction_wrapper), &std_correction_wrapper, optional<int64_t>, bool>(self, self_bdim, *dim, correction, keepdim);
}

Tensor var_correction_wrapper(const Tensor& self, IntArrayRef dim, optional<int64_t> correction, bool keepdim) {
  return at::var(self, dim, correction, keepdim);
}
std::tuple<Tensor,optional<int64_t>> var_correction_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<IntArrayRef> dim, optional<int64_t> correction, bool keepdim) {
  if (!dim.has_value()) {
    dim = range(0, self.dim() - 1);
  }
  return reduction_dimarray_batch_rule<decltype(&var_correction_wrapper), &var_correction_wrapper, optional<int64_t>, bool>(self, self_bdim, *dim, correction, keepdim);
}

std::tuple<Tensor,optional<int64_t>> prod_dim_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, int64_t dim, bool keepdim, optional<ScalarType> dtype) {
  return reduction_dim_batch_rule<decltype(&ATEN_FN2(prod, dim_int)), &at::prod, bool, optional<ScalarType>>(self, self_bdim, dim, keepdim, dtype);
}

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>> max_dim_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, int64_t dim, bool keepdim) {
  return reduction_dim_ret_pair_batch_rule<decltype(&ATEN_FN2(max, dim)), &at::max, bool>(self, self_bdim, dim, keepdim);
}

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>> min_dim_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, int64_t dim, bool keepdim) {
  return reduction_dim_ret_pair_batch_rule<decltype(&ATEN_FN2(min, dim)), &at::min, bool>(self, self_bdim, dim, keepdim);
}
// Wraps topk so that dim is the first argument after self (makes it work with templates)
std::tuple<Tensor, Tensor> wrapped_topk(
    const Tensor& self, int64_t dim, int64_t k, bool largest, bool sorted) {
  return at::topk(self, k, dim, largest, sorted);
}

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>> topk_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, int64_t k, int64_t dim, bool largest, bool sorted) {
  return reduction_dim_ret_pair_batch_rule<decltype(&wrapped_topk), &wrapped_topk, int64_t, bool, bool>(self, self_bdim, dim, k, largest, sorted);
}

std::tuple<Tensor, Tensor> wrapped_sort_stable(
    const Tensor& self, int64_t dim, optional<bool> stable, bool descending) {
  return at::sort(self, stable, dim, descending);
}

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>> sort_stable_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<bool> stable, int64_t dim, bool descending) {
  return reduction_dim_ret_pair_batch_rule<decltype(&wrapped_sort_stable), &wrapped_sort_stable, optional<bool>, bool>(self, self_bdim, dim, stable, descending);
}

// Skipping frobenius/nuclear/all/any since they don't have opinfo tests right now :P

template<typename F, F Func>
std::tuple<Tensor,optional<int64_t>> argx_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<int64_t> dim, bool keepdim) {
  if (!self_bdim.has_value()) {
    return std::make_tuple( Func(self, dim, keepdim), nullopt );
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  if (!dim) {
    // If no dimension is given, then argmax gives you the flattened index of
    // the maximum element. We need to do this flatten/keepdim shenanigans in order
    // to preserve that behavior.
    dim = 0;
    if (self_.dim() > 1) {
      self_ = at::flatten(self, 1);
    }
    keepdim = false;
  }
  auto new_dim = getPhysicalDim(self_, self_bdim.has_value(), *dim);
  if (self_.dim() <= new_dim) {
    // This happens when the original tensor is a scalar
    // vmap(argmax(shape [], 0)) => argmax(shape [5, 1], 1)
    TORCH_INTERNAL_ASSERT(self_.dim() == 1);
    self_ = self_.unsqueeze(-1);
  }
  auto result = Func(self_, new_dim, keepdim);
  return std::make_tuple(result, 0);
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
    return std::make_tuple( at::_log_softmax_backward_data(grad_output, output, dim, self), nullopt );
  }
  if (grad_output_bdim && self_bdim) {
    auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
    auto output_ = moveBatchDimToFront(output, output_bdim);
    auto self_ = moveBatchDimToFront(self, self_bdim);
    dim = getPhysicalDim(grad_output_, /*has_batch_dim*/true, dim);
    return std::make_tuple( at::_log_softmax_backward_data(grad_output_, output_, dim, self_), 0 );
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
    return std::make_tuple( at::_log_softmax_backward_data(grad_output_, output_, dim, self_), 0 );
  }
  if (!grad_output_bdim && self_bdim) {
    auto output_ = moveBatchDimToFront(output, output_bdim);
    auto self_ = moveBatchDimToFront(self, self_bdim);
    auto grad_output_ = grad_output.expand_as(output_);
    dim = getPhysicalDim(grad_output_, /*has_batch_dim*/true, dim);
    return std::make_tuple( at::_log_softmax_backward_data(grad_output_, output_, dim, self_), 0 );
  }
  TORCH_INTERNAL_ASSERT(false);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("amax", SINGLE_ARG(reduction_dimarray_batch_rule<decltype(&at::amax), &at::amax, bool>));
  VMAP_SUPPORT("amin", SINGLE_ARG(reduction_dimarray_batch_rule<decltype(&at::amin), &at::amin, bool>));
  VMAP_SUPPORT("argmax", SINGLE_ARG(argx_batch_rule<decltype(&at::argmax), &at::argmax>));
  VMAP_SUPPORT("argmin", SINGLE_ARG(argx_batch_rule<decltype(&at::argmin), &at::argmin>));
  VMAP_SUPPORT("cumprod", SINGLE_ARG(reduction_dim_batch_rule<decltype(&ATEN_FN(cumprod)), &at::cumprod, optional<ScalarType>>));
  VMAP_SUPPORT("cumsum", SINGLE_ARG(reduction_dim_batch_rule<decltype(&ATEN_FN(cumsum)), &at::cumsum, optional<ScalarType>>));
  VMAP_SUPPORT("log_softmax.int", SINGLE_ARG(reduction_dim_batch_rule<decltype(&ATEN_FN2(log_softmax, int)), &at::log_softmax, optional<ScalarType>>));
  VMAP_SUPPORT("nansum", nansum_batch_rule);
  VMAP_SUPPORT("nansum.dim_IntList", nansum_dim_batch_rule);
  VMAP_SUPPORT("max", SINGLE_ARG(reduction_no_dim_batch_rule<decltype(&ATEN_FN(max)), &at::max, decltype(&max_dim_batch_rule), &max_dim_batch_rule>));
  VMAP_SUPPORT("max.dim", max_dim_batch_rule);
  VMAP_SUPPORT("mean", mean_batch_rule);
  VMAP_SUPPORT("mean.dim", mean_dim_batch_rule);
  VMAP_SUPPORT("min", SINGLE_ARG(reduction_no_dim_batch_rule<decltype(&ATEN_FN(min)), &at::min, decltype(&min_dim_batch_rule), &min_dim_batch_rule>));
  VMAP_SUPPORT("min.dim", min_dim_batch_rule);
  VMAP_SUPPORT("mode", SINGLE_ARG(reduction_dim_ret_pair_batch_rule<decltype(&ATEN_FN(mode)), &at::mode, bool>));
  VMAP_SUPPORT("prod", SINGLE_ARG(reduction_no_dim_batch_rule<decltype(&ATEN_FN(prod)), &at::prod, decltype(&prod_dim_batch_rule), &prod_dim_batch_rule, optional<ScalarType>>));
  VMAP_SUPPORT("prod.dim_int", prod_dim_batch_rule);
  OP_DECOMPOSE(std);
  OP_DECOMPOSE2(std, dim);
  VMAP_SUPPORT("std.correction", std_correction_batch_rule);
  VMAP_SUPPORT("sort", SINGLE_ARG(reduction_dim_ret_pair_batch_rule<decltype(&ATEN_FN(sort)), &at::sort, bool>));
  VMAP_SUPPORT("sort.stable", sort_stable_batch_rule);
  VMAP_SUPPORT("sum", sum_batch_rule);
  VMAP_SUPPORT("sum.dim_IntList", sum_dim_batch_rule);
  VMAP_SUPPORT("topk", topk_batch_rule);
  OP_DECOMPOSE(var);
  OP_DECOMPOSE2(var, dim);
  VMAP_SUPPORT("var.correction", var_correction_batch_rule);
  VMAP_SUPPORT("_log_softmax_backward_data", _log_softmax_backward_data);
}

}}
