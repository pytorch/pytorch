// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace functorch {

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

Tensor nanmedian_decomp(
    const Tensor& self) {
  return std::get<0>(at::nanmedian(self.flatten(), 0, false));
}

Tensor median_decomp(
    const Tensor& self) {
  return std::get<0>(at::median(self.flatten(), 0, false));
}

enum ReductionCase { DimArray, Dim };

// dim_arg_pos allows us to specify the location of the dim/dim array argument.
// Defaults to 1
template<int dim_arg_pos=1>
void boxed_reduction_batch_rule(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();

  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

  auto orig_arguments = torch::jit::last(*stack, num_arguments);
  if (std::none_of(orig_arguments.begin(), orig_arguments.end(), ivalueParticipatesInCurrentLevel)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    op.callBoxed(stack);
    return;
  }

  auto arguments = torch::jit::pop(*stack, num_arguments);

  TORCH_INTERNAL_ASSERT(arguments[0].isTensor());
  Tensor self;
  optional<int64_t> self_bdim;
  std::tie(self, self_bdim) = unwrapTensorAtLevel(arguments[0].toTensor(), cur_level);

  self = moveBatchDimToFront(self, self_bdim);

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
    auto param_type = schema.arguments()[dim_arg_pos].type()->expect<OptionalType>()->getElementType();
    if (param_type->kind() == IntType::Kind) {
      reduction_case = ReductionCase::Dim;
      if (self.dim() > 1) {
        self = self.flatten(1);
      }
      dims = {0};
    } else if (param_type->kind() == ListType::Kind) {
      reduction_case = ReductionCase::DimArray;
      if (logical_dim == 0) {
        dims = {0};
      } else {
        auto all_dims = range(0, self.dim() - 1);
        dims = std::vector<int64_t>(all_dims.begin(), all_dims.end());
      }
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unexpected dtype found at dims");
    }
  } else{
    TORCH_INTERNAL_ASSERT(false, "Unexpected dtype found at dims");
  }

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

Tensor dist_decomp(const Tensor& self, const Tensor& other, const Scalar& p) {
  return at::norm((self - other), p);
}

static std::tuple<Tensor, Tensor> expand_bdims(
    const Tensor& a, bool a_has_bdim,
    const Tensor& b, bool b_has_bdim) {
  Tensor flagpole;
  if (a_has_bdim) {
    flagpole = a;
  } else if (b_has_bdim) {
    flagpole = b;
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
  return std::make_tuple(
      a_has_bdim ? a : a.expand_as(flagpole),
      b_has_bdim ? b : b.expand_as(flagpole));
}

std::tuple<Tensor,optional<int64_t>> _softmax_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& output, optional<int64_t> output_bdim,
    int64_t dim,
    ScalarType input_dtype) {
  // softmax_backward's decomposition is y * gy - y * (y * gy).sum(dim, keepdim=True)
  // NB: the CUDA kernel handles strides so we can just expand
  // all of the tensors and call it a day. The CPU kernel is not as good but
  // idk if the perf on that really matters
  auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
  auto output_ = moveBatchDimToFront(output, output_bdim);

  // Expand out that extra dimension for everyone
  std::tie(grad_output_, output_) = expand_bdims(
      grad_output_, grad_output_bdim.has_value(),
      output_, output_bdim.has_value());

  // Scalar tensor case. softmax turns into the identity when this happens.
  // I don't know why the output is zeros, though, but that's what softmax tells me...
  if (output_.dim() == 1 && (dim == 0 || dim == -1)) {
    return std::make_tuple(at::zeros_like(grad_output_), 0);
  }

  dim = getPhysicalDim(output_, /*has_batch_dim*/true, dim);

  // Not sure why output_ needs to be marked as .contiguous(). Someting must
  // have changed in PyTorch (and output of softmax is probably always contiguous)
  return std::make_tuple(at::_softmax_backward_data(grad_output_, output_.contiguous(), dim, input_dtype), 0);
}

std::tuple<Tensor,optional<int64_t>> _log_softmax_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& output, optional<int64_t> output_bdim,
    int64_t dim,
    c10::ScalarType input_dtype) {
  // NB: It turns out that expanding + calling log_softmax_backward is generally
  // faster than the decomposition.
  // Benchmark here: https://gist.github.com/zou3519/ae3b33b5730a84aae8a80a05c89e078a
  // Decomposition is (grad_output - grad_output.sum(dim, keepdim=True) * result.exp())
  // We can squeeze out a last mile of performance by writing custom kernels.
  auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
  auto output_ = moveBatchDimToFront(output, output_bdim);

  // Expand out that extra dimension for everyone
  std::tie(grad_output_, output_) = expand_bdims(
      grad_output_, grad_output_bdim.has_value(),
      output_, output_bdim.has_value());

  // Scalar tensor case. log_softmax returns zeros when this happens
  if (output_.dim() == 1 && (dim == 0 || dim == -1)) {
    return std::make_tuple(at::zeros_like(grad_output_), 0);
  }

  dim = getPhysicalDim(output_, /*has_batch_dim*/true, dim);

  return std::make_tuple(at::_log_softmax_backward_data(grad_output_, output_, dim, input_dtype), 0);
}

// aminmax has divergent behavior for 0-d tenosrs.
// reference: https://github.com/pytorch/pytorch/issues/64008
// TODO: Once the divergent behavior for 0-d scalar is fixed, we should use REDUCTION_BOXED_ARGS
std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>> aminmax_batching_rule(
    const Tensor &self, optional<int64_t> self_bdim, optional<int64_t> dim, bool keep_dim)
{
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto logical_rank = rankWithoutBatchDim(self_, self_bdim);
  if (logical_rank == 0) {
    self_ = self_.unsqueeze(-1);
  }

  if (dim.has_value()) {
    dim = maybe_wrap_dim(dim.value(), logical_rank) + 1;
  } else {
    // flatten the input except for batch-dim
    auto bsize = self_.size(0);
    self_ = self_.view({bsize, -1});
    dim = 1;
  }

  Tensor min, max;
  std::tie(min, max) = at::aminmax(self_, dim, keep_dim);

  if (logical_rank == 0 && self_.device().is_cuda()) {
    // behaviour diverges between cpu and cuda
    min = min.squeeze(-1);
    max = max.squeeze(-1);
  }
  return std::make_tuple(min, 0, max, 0);
}

std::tuple<Tensor,optional<int64_t>> searchsorted_batch_rule(
    const Tensor& sorted_sequence,
    optional<int64_t> sorted_sequence_bdim,
    const Tensor& self,
    optional<int64_t> self_bdim,
    bool out_int32,
    bool right,
    c10::optional<c10::string_view> side,
    const c10::optional<Tensor>& sorter,
    c10::optional<int64_t> sorter_bdim) {
  auto buckets_logical_rank = rankWithoutBatchDim(sorted_sequence, sorted_sequence_bdim);

  // Preprocess sorter and sorted_sequence.
  // If they both exist, and only one has a bdim, then we need to make sure both do.
  // After this step, we can forget about sorter for a bit.
  auto buckets = moveBatchDimToFront(sorted_sequence, sorted_sequence_bdim);
  optional<int64_t> buckets_bdim;
  if (sorted_sequence_bdim.has_value()) {
    buckets_bdim = 0;
  }

  optional<Tensor> sorter_;
  if (sorter.has_value() && sorter->defined()) {
    auto sorter__ = moveBatchDimToFront(*sorter, sorter_bdim);
    if (sorted_sequence_bdim.has_value() != sorter_bdim.has_value()) {
      auto bdim_size = get_bdim_size2(
          sorted_sequence, sorted_sequence_bdim,
          sorter.value(), sorter_bdim);
      sorter__ = ensure_has_bdim(sorter__, sorter_bdim.has_value(), bdim_size);
      buckets = ensure_has_bdim(buckets, sorted_sequence_bdim.has_value(), bdim_size);
      buckets_bdim = 0;
    }
    sorter_ = sorter__;
  }

  // Two cases: buckets_logical_rank is 1, or it is greater than 1.
  // searchsorted is basically two operators with different semantics jammed
  // into one
  if (buckets_logical_rank > 1) {
    // B<...>D, B<...>V -> no change
    if (buckets_bdim.has_value() && self_bdim.has_value()) {
      auto self_ = moveBatchDimToFront(self, self_bdim);
      auto result = at::searchsorted(buckets, self_, out_int32, right, side, sorter_);
      return std::make_tuple(result, 0);
    }
    // B<...>D, <...>V -> B<...>D, B<...>V
    if (buckets_bdim.has_value() && !self_bdim.has_value()) {
      auto self_ = moveBatchDimToFront(self, self_bdim);
      self_ = ensure_has_bdim(self_, self_bdim.has_value(), buckets.size(0));
      auto result = at::searchsorted(buckets, self_, out_int32, right, side, sorter_);
      return std::make_tuple(result, 0);
    }
    // <...>D, B<...>V -> <...>D, <...>(BV)
    if (!buckets_bdim.has_value() && self_bdim.has_value()) {
      auto bdim_size = self.size(*self_bdim);
      auto self_ = reshape_dim_into(*self_bdim, -1, self);
      auto result = at::searchsorted(buckets, self_, out_int32, right, side, sorter_);
      result = reshape_dim_outof(-1, bdim_size, result);
      return std::make_tuple(result, result.dim() - 2);
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  // buckets_logical_rank == 1 case.
  // BD, B* -> BD, B flat(*)
  if (buckets_bdim.has_value() && self_bdim.has_value()) {
    auto self_ = moveBatchDimToFront(self, self_bdim);
    self_ = self_.flatten(1);
    auto result = at::searchsorted(buckets, self_, out_int32, right, side, sorter_);
    result = result.view(self_.sizes());
    return std::make_tuple(result, 0);
  }
  // BD, * -> BD, flat(*) -> BD, B flat(*)
  if (buckets_bdim.has_value() && !self_bdim.has_value()) {
    auto bdim_size = buckets.size(*buckets_bdim);
    auto self_ = ensure_has_bdim(self, false, bdim_size);
    self_ = self_.flatten(1);
    auto result = at::searchsorted(buckets, self_, out_int32, right, side, sorter_);
    result = result.view(self_.sizes());
    return std::make_tuple(result, 0);
  }
  // D, B* -> no change
  if (!buckets_bdim.has_value() && self_bdim.has_value()) {
    auto result = at::searchsorted(buckets, self, out_int32, right, side, sorter_);
    return std::make_tuple(result, self_bdim);
  }
  TORCH_INTERNAL_ASSERT(false);
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT2(searchsorted, Tensor, searchsorted_batch_rule);
  REDUCTION_BOXED(_fft_r2c);
  REDUCTION_BOXED(_fft_c2r);
  REDUCTION_BOXED(_fft_c2c);
  REDUCTION_BOXED(amax);
  // REDUCTION_BOXED(aminmax); Currently fails due to inconsistent scalar semantics.
  REDUCTION_BOXED(amin);
  REDUCTION_BOXED(any.dim);
  REDUCTION_BOXED(argmax);
  REDUCTION_BOXED(argmin);
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
  m.impl("median", median_decomp);
  REDUCTION_BOXED(median.dim);
  m.impl("min", min_decomp);
  REDUCTION_BOXED(min.dim);
  REDUCTION_BOXED(mode);
  m.impl("nanmedian", nanmedian_decomp);
  REDUCTION_BOXED(nanmedian.dim);
  // TODO: re-enable these
  // m.impl("nansum", nansum_decomp);
  // REDUCTION_BOXED(nansum.dim_IntList);
  m.impl("norm.Scalar", norm_scalar_decomp);
  REDUCTION_BOXED_ARGS(norm.ScalarOpt_dim, 2);
  m.impl("prod", prod_decomp);
  REDUCTION_BOXED(prod.dim_int);
  REDUCTION_BOXED(std.correction);
  REDUCTION_BOXED(_softmax);
  REDUCTION_BOXED(sort);
  REDUCTION_BOXED_ARGS(sort.stable, 2);
  REDUCTION_BOXED(argsort);
  REDUCTION_BOXED(std_mean.correction);
  m.impl("sum", sum_decomp);
  REDUCTION_BOXED(sum.dim_IntList);
  REDUCTION_BOXED_ARGS(topk, 2);
  REDUCTION_BOXED(var.correction);
  REDUCTION_BOXED(var_mean.correction);
  REDUCTION_BOXED(_log_softmax);
  REDUCTION_BOXED_ARGS(rot90, 2);
  VMAP_SUPPORT(aminmax, aminmax_batching_rule);
  VMAP_SUPPORT(_log_softmax_backward_data, _log_softmax_backward_batch_rule);
  VMAP_SUPPORT(_softmax_backward_data, _softmax_backward_batch_rule);
}
}}
