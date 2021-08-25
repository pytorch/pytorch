// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>

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

Tensor sum_decomp(
    const Tensor& self, optional<ScalarType> dtype) {
  return at::sum(self, range(0, self.dim()), false, dtype);
}

Tensor mean_decomp(
    const Tensor& self, optional<ScalarType> dtype) {
  return at::mean(self, range(0, self.dim()), false, dtype);
}

Tensor nansum_decomp(
    const Tensor& self, optional<ScalarType> dtype) {
  return at::nansum(self, range(0, self.dim()), false, dtype);
}

Tensor prod_decomp(
    const Tensor& self, optional<ScalarType> dtype) {
  return at::prod(self.flatten(), 0, false, dtype);
}

Tensor max_decomp(
    const Tensor& self) {
  return std::get<0>(at::max(self.flatten(), 0, false));
}

Tensor min_decomp(
    const Tensor& self) {
  return std::get<0>(at::min(self.flatten(), 0, false));
}

Tensor norm_scalar_decomp(
    const Tensor& self, const Scalar& p) {
  return at::norm(self, p, range(0, self.dim()), false);
}

enum ReductionCase { DimArray, Dim };

template<int dim_arg_pos=1>
void boxed_reduction_batch_rule(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();
  auto arguments = torch::jit::pop(*stack, num_arguments);

  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

  std::vector<std::pair<Tensor, optional<int64_t>>> tensor_inputs;
  std::vector<int64_t> tensor_pos;

  TORCH_INTERNAL_ASSERT(arguments[0].isTensor());
  Tensor self;
  optional<int64_t> self_bdim;
  std::tie(self, self_bdim) = unwrapTensorAtLevel(arguments[0].toTensor(), cur_level);

  auto logical_dim = rankWithoutBatchDim(self, self_bdim);
  std::vector<int64_t> dims;
  ReductionCase reduction_case;
  if (arguments[dim_arg_pos].isIntList()) {
    reduction_case = ReductionCase::DimArray;
    dims = arguments[dim_arg_pos].toIntList().vec();
    if (dims.size() == 0) {
      auto all_dims = range(0, std::max((int64_t)1, logical_dim));
      dims = std::vector<int64_t>(all_dims.begin(), all_dims.end());
    }
  } else if (arguments[dim_arg_pos].isInt()) {
    reduction_case = ReductionCase::Dim;
    dims = {arguments[dim_arg_pos].toInt()};
  } else if (arguments[dim_arg_pos].isNone())  {
    reduction_case = ReductionCase::DimArray;
    auto all_dims = range(0, self.dim() - 1);
    dims = std::vector<int64_t>(all_dims.begin(), all_dims.end());
  } else{
    TORCH_INTERNAL_ASSERT(false, "Unexpected dtype found at dims");
    // auto all_dims = range(0, self.dim() - 1);
    // dims = std::vector<int64_t>(all_dims.begin(), all_dims.end());
  }

  self = moveBatchDimToFront(self, self_bdim);
  VmapDimVector new_dims;
  new_dims.reserve(dims.size());
  for (auto dim: dims) {
    new_dims.push_back(getPhysicalDim(self, self_bdim.has_value(), dim));
  }
  bool is_scalar_case = logical_dim == 0 && dims.size() == 1 && is_allowed_dim_on_scalar_tensor(dims[0]);
  if (is_scalar_case) {
    self = self.unsqueeze(-1);
    new_dims = {1};
  }
  arguments[0] = self;
  if (reduction_case == ReductionCase::DimArray) {
    arguments[dim_arg_pos] = std::vector<int64_t>(new_dims.begin(), new_dims.end());
  } else if (reduction_case == ReductionCase::Dim) {
    arguments[dim_arg_pos] = new_dims[0];
  }
  for (const auto arg_idx : c10::irange(0, num_arguments)) {
    torch::jit::push(stack, arguments[arg_idx]);
  }
  op.callBoxed(stack);

  const auto returns = torch::jit::pop(*stack, num_returns);
  for (const auto& ret : returns) {
    if (ret.isTensor()) {
      auto res = ret.toTensor();
      if (is_scalar_case) {
        res = res.squeeze(-1);
      }
      torch::jit::push(stack, makeBatched(res, 0, cur_level));
    } else {
      TORCH_INTERNAL_ASSERT(false, "This boxed batching rule does not currently support ops that return non-tensor values");
    }
  }
}

#define REDUCTION_BOXED(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_reduction_batch_rule>());

#define REDUCTION_BOXED_ARGS(op, dim_pos) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_reduction_batch_rule<dim_pos>>());

// Skipping frobenius/nuclear/all/any since they don't have opinfo tests right now :P

template<typename F, F Func>
std::tuple<Tensor,optional<int64_t>> argx_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<int64_t> dim, bool keepdim) {
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

Tensor dist_decomp(const Tensor& self, const Tensor& other, const Scalar& p) {
  return at::norm((self - other), p);
}

static std::tuple<Tensor, Tensor, Tensor> expand_bdims(
    const Tensor& a, bool a_has_bdim,
    const Tensor& b, bool b_has_bdim,
    const Tensor& c, bool c_has_bdim) {
  int64_t bdim_size = -1;
  Tensor flagpole;
  if (a_has_bdim) {
    bdim_size = a.size(0);
    flagpole = a;
  } else if (b_has_bdim) {
    bdim_size = b.size(0);
    flagpole = b;
  } else if (c_has_bdim) {
    bdim_size = c.size(0);
    flagpole = c;
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
  return std::make_tuple(
      a_has_bdim ? a : a.expand_as(flagpole),
      b_has_bdim ? b : b.expand_as(flagpole),
      c_has_bdim ? c : c.expand_as(flagpole));
}

std::tuple<Tensor,optional<int64_t>> _softmax_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& output, optional<int64_t> output_bdim,
    int64_t dim,
    const Tensor& self, optional<int64_t> self_bdim) {
  TORCH_INTERNAL_ASSERT(!(output_bdim.has_value() ^ self_bdim.has_value()),
    "output_bdim and self_bdim must be the same");
  // NB: self only gets used for its dtype. We handle it anyways just in case.
  // softmax_backward's decomposition is y * gy - y * (y * gy).sum(dim, keepdim=True)
  // NB: the CUDA kernel handles strides so we can just expand
  // all of the tensors and call it a day. The CPU kernel is not as good but
  // idk if the perf on that really matters
  auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
  auto output_ = moveBatchDimToFront(output, output_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim);

  // Expand out that extra dimension for everyone
  std::tie(grad_output_, output_, self_) = expand_bdims(
      grad_output_, grad_output_bdim.has_value(),
      output_, output_bdim.has_value(),
      self_, self_bdim.has_value());

  // Scalar tensor case. softmax turns into the identity when this happens.
  // I don't know why the output is zeros, though, but that's what softmax tells me...
  if (self_.dim() == 1 && (dim == 0 || dim == -1)) {
    return std::make_tuple(at::zeros_like(grad_output_), 0);
  }

  dim = getPhysicalDim(self_, /*has_batch_dim*/true, dim);

  // Not sure why output_ needs to be marked as .contiguous(). Someting must
  // have changed in PyTorch (and output of softmax is probably always contiguous)
  return std::make_tuple(at::_softmax_backward_data(grad_output_, output_.contiguous(), dim, self_), 0);
}

std::tuple<Tensor,optional<int64_t>> _log_softmax_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& output, optional<int64_t> output_bdim,
    int64_t dim,
    const Tensor& self, optional<int64_t> self_bdim) {
  TORCH_INTERNAL_ASSERT(!(output_bdim.has_value() ^ self_bdim.has_value()),
    "output_bdim and self_bdim must be the same");
  // NB: It turns out that expanding + calling log_softmax_backward is generally
  // faster than the decomposition.
  // Benchmark here: https://gist.github.com/zou3519/ae3b33b5730a84aae8a80a05c89e078a
  // Decomposition is (grad_output - grad_output.sum(dim, keepdim=True) * result.exp())
  // We can squeeze out a last mile of performance by writing custom kernels.
  auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
  auto output_ = moveBatchDimToFront(output, output_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim);

  // Expand out that extra dimension for everyone
  std::tie(grad_output_, output_, self_) = expand_bdims(
      grad_output_, grad_output_bdim.has_value(),
      output_, output_bdim.has_value(),
      self_, self_bdim.has_value());

  // Scalar tensor case. log_softmax returns zeros when this happens
  if (self_.dim() == 1 && (dim == 0 || dim == -1)) {
    return std::make_tuple(at::zeros_like(grad_output_), 0);
  }

  dim = getPhysicalDim(self_, /*has_batch_dim*/true, dim);

  return std::make_tuple(at::_log_softmax_backward_data(grad_output_, output_, dim, self_), 0);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  REDUCTION_BOXED(_fft_r2c);
  REDUCTION_BOXED(_fft_c2r);
  REDUCTION_BOXED(_fft_c2c);
  REDUCTION_BOXED(amax);
  REDUCTION_BOXED(amin);
  REDUCTION_BOXED(any.dim);
  VMAP_SUPPORT("argmax", SINGLE_ARG(argx_batch_rule<decltype(&at::argmax), &at::argmax>));
  VMAP_SUPPORT("argmin", SINGLE_ARG(argx_batch_rule<decltype(&at::argmin), &at::argmin>));
  REDUCTION_BOXED(count_nonzero.dim_IntList);
  REDUCTION_BOXED(cummax);
  REDUCTION_BOXED(cummin);
  REDUCTION_BOXED(cumprod);
  REDUCTION_BOXED(cumsum);
  m.impl("dist", dist_decomp);
  REDUCTION_BOXED_ARGS(kthvalue, 2);
  REDUCTION_BOXED_ARGS(linalg_vector_norm, 2);
  REDUCTION_BOXED(log_softmax.int);
  REDUCTION_BOXED(logcumsumexp);
  REDUCTION_BOXED(logsumexp);
  m.impl("max", max_decomp);
  REDUCTION_BOXED(max.dim);
  m.impl("mean", mean_decomp);
  REDUCTION_BOXED(mean.dim);
  REDUCTION_BOXED(median.dim);
  m.impl("min", min_decomp);
  REDUCTION_BOXED(min.dim);
  REDUCTION_BOXED(mode);
  REDUCTION_BOXED(nanmedian.dim);
  m.impl("nansum", nansum_decomp);
  REDUCTION_BOXED(nansum.dim_IntList);
  m.impl("norm.Scalar", norm_scalar_decomp);
  REDUCTION_BOXED_ARGS(norm.ScalarOpt_dim, 2);
  m.impl("prod", prod_decomp);
  REDUCTION_BOXED(prod.dim_int);
  REDUCTION_BOXED(std.correction);
  REDUCTION_BOXED(_softmax);
  REDUCTION_BOXED(sort);
  REDUCTION_BOXED_ARGS(sort.stable, 2);
  m.impl("sum", sum_decomp);
  REDUCTION_BOXED(sum.dim_IntList);
  REDUCTION_BOXED_ARGS(topk, 2);
  REDUCTION_BOXED(var.correction);
  VMAP_SUPPORT("_log_softmax_backward_data", _log_softmax_backward_batch_rule);
  VMAP_SUPPORT("_softmax_backward_data", _softmax_backward_batch_rule);
}

}}
