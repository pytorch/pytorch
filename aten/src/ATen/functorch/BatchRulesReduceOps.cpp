// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <utility>

namespace at::functorch {

static bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}

static Tensor sum_decomp(
    const Tensor& self, std::optional<ScalarType> dtype) {
  return at::sum(self, range(0, self.dim()), false, dtype);
}

static std::tuple<Tensor, std::optional<int64_t>> _is_all_true_batch_rule(
    const Tensor& self, std::optional<int64_t> self_bdim) {
  return std::make_tuple(at::_is_all_true(self), std::nullopt);
}

static std::tuple<Tensor, std::optional<int64_t>> _is_any_true_batch_rule(
     const Tensor& self, std::optional<int64_t> self_bdim) {
   return std::make_tuple(at::_is_any_true(self), std::nullopt);
 }

static Tensor mean_decomp(
    const Tensor& self, std::optional<ScalarType> dtype) {
  return at::mean(self, range(0, self.dim()), false, dtype);
}

static Tensor prod_decomp(
    const Tensor& self, std::optional<ScalarType> dtype) {
  return at::prod(self.flatten(), 0, false, dtype);
}

static Tensor max_decomp(
    const Tensor& self) {
  return std::get<0>(at::max(self.flatten(), 0, false));
}

static Tensor min_decomp(
    const Tensor& self) {
  return std::get<0>(at::min(self.flatten(), 0, false));
}

static Tensor norm_scalar_decomp(
    const Tensor& self, const Scalar& p) {
  return at::norm(self, p, range(0, self.dim()), false);
}

static Tensor nanmedian_decomp(
    const Tensor& self) {
  return std::get<0>(at::nanmedian(self.flatten(), 0, false));
}

static Tensor median_decomp(
    const Tensor& self) {
  return std::get<0>(at::median(self.flatten(), 0, false));
}

static Tensor all_decomp(const Tensor& self) {
  return at::all(self.flatten(), 0, false);
}

static Tensor any_decomp(const Tensor& self) {
  return at::any(self.flatten(), 0, false);
}

enum class ReductionCase:uint8_t { DimArray, Dim };

// Macros and templates have a difficult time dealing with enums,
// so we didn't turn this into an enum.
// See NOTE: [keepdim cases] for explanation of what these are.
static constexpr int KEEPDIM_CASE_FALSE = 0;
static constexpr int KEEPDIM_CASE_TRUE = 1;
static constexpr int KEEPDIM_CASE_VARIABLE = 2;

// dim_arg_pos allows us to specify the location of the dim/dim array argument.
// For most PyTorch ops, this is equal to 1.
//
// NOTE: [keepdim cases]
// The operator in question either:
// - has a keepdim argument (KeepdimCase.Variable)
//   In this case, `maybe_keepdim_arg_pos` says where the index of the keepdim arg is.
//   example: sum(tensor, dim, keepdim)
// - always does a reduction with no keepdim (KeepdimCase.False)
//   that is, the rank of the output tensor is less than the rank of the input tensor.
// - always does a reduction with keepdim=True semantics (KeepdimCase.True)
//   That is, the rank of the output tensor is always the same as that of the input.
//   examples: log_softmax(tensor, dim), cumsum(tensor, dim)
template<
  int dim_arg_pos,
  int keepdim_case,
  // optional cannot be used in a template, otherwise we would use it here.
  int maybe_keepdim_arg_pos
>
static void boxed_reduction_batch_rule(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();

  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "boxed_reduction_batch_rule");
  int64_t cur_level = maybe_layer->layerId();

  auto orig_arguments = torch::jit::last(*stack, num_arguments);
  if (std::none_of(orig_arguments.begin(), orig_arguments.end(), ivalueParticipatesInCurrentLevel)) {
    c10::impl::ExcludeDispatchKeyGuard guard_2(DispatchKey::FuncTorchBatched);
    op.callBoxed(stack);
    return;
  }

  auto arguments = torch::jit::pop(*stack, num_arguments);

  TORCH_INTERNAL_ASSERT(arguments[0].isTensor());
  auto [self, self_bdim] = unwrapTensorAtLevel(arguments[0].toTensor(), cur_level);

  self = moveBatchDimToFront(self, self_bdim);

  auto logical_dim = rankWithoutBatchDim(self, self_bdim);
  std::vector<int64_t> dims;
  ReductionCase reduction_case{};
  if (arguments[dim_arg_pos].isIntList()) {
    reduction_case = ReductionCase::DimArray;
    dims = arguments[dim_arg_pos].toIntList().vec();
    if (dims.empty()) {
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
  std::optional<bool> maybe_keepdim;
  if (is_scalar_case) {
    // NOTE: [boxed_reduction_batch_rule scalar tensor handling]
    // Reduction operations in PyTorch have an edge case where they allow
    // dim=0 and dim=-1 if the tensor has shape [].
    //
    // This can come up if we do something like
    // vmap(lambda x: x.sum(0))(torch.tensor([10.])),
    //
    // In order to handle this edge case, we unsqueeze a dimension on the Tensor,
    // run the operation (with dim=1 instead), and then process the output tensor.
    // There are two cases:
    // - keepdim = True
    //     unsqueeze   op      squeeze
    //   [B] -> [B, 1] -> [B, 1] -> [B]
    // - keepdim = False
    //     unsqueeze   op     no need to squeeze
    //   [B] -> [B, 1] -> [B]
    // if keepdim is True, then we need to squeeze the dimension of size 1.

    // Determine the value of keepdim
    switch (keepdim_case) {
      case KEEPDIM_CASE_FALSE:
        maybe_keepdim = false;
        break;
      case KEEPDIM_CASE_TRUE:
        maybe_keepdim = true;
        break;
      case KEEPDIM_CASE_VARIABLE:
        TORCH_INTERNAL_ASSERT(maybe_keepdim_arg_pos >= 0);
        maybe_keepdim = arguments[maybe_keepdim_arg_pos].toBool();
        break;
    }
    self = self.unsqueeze(-1);
    new_dims = {1};
  }
  arguments[0] = std::move(self);
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
      // see NOTE: [boxed_reduction_batch_rule scalar tensor handling]
      if (is_scalar_case && maybe_keepdim.value()) {
        // squeeze(-1) is a no-op if the shape of the dim is not 1.
        // To make it safer, we internal assert here.
        TORCH_INTERNAL_ASSERT(res.size(-1) == 1);
        res = res.squeeze(-1);
      }
      torch::jit::push(stack, makeBatched(res, 0, cur_level));
    } else {
      TORCH_INTERNAL_ASSERT(false, "This boxed batching rule does not currently support ops that return non-tensor values");
    }
  }
}

// Skipping all/any since they don't have opinfo tests right now :P

static Tensor dist_decomp(const Tensor& self, const Tensor& other, const Scalar& p) {
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

static std::tuple<Tensor, std::optional<int64_t>> _softmax_backward_batch_rule(
    const Tensor& grad_output, std::optional<int64_t> grad_output_bdim,
    const Tensor& output, std::optional<int64_t> output_bdim,
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

static std::tuple<Tensor, std::optional<int64_t>> _log_softmax_backward_batch_rule(
    const Tensor& grad_output, std::optional<int64_t> grad_output_bdim,
    const Tensor& output, std::optional<int64_t> output_bdim,
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

static std::tuple<Tensor, std::optional<int64_t>> searchsorted_batch_rule(
    const Tensor& sorted_sequence,
    std::optional<int64_t> sorted_sequence_bdim,
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    bool out_int32,
    bool right,
    std::optional<c10::string_view> side,
    const std::optional<Tensor>& sorter,
    std::optional<int64_t> sorter_bdim) {
  auto buckets_logical_rank = rankWithoutBatchDim(sorted_sequence, sorted_sequence_bdim);
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);

  // Preprocess sorter and sorted_sequence.
  // If they both exist, and only one has a bdim, then we need to make sure both do.
  // After this step, we can forget about sorter for a bit.
  auto buckets = moveBatchDimToFront(sorted_sequence, sorted_sequence_bdim);
  std::optional<int64_t> buckets_bdim;
  if (sorted_sequence_bdim.has_value()) {
    buckets_bdim = 0;
  }

  std::optional<Tensor> sorter_;
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
      return std::make_tuple(std::move(result), 0);
    }
    // B<...>D, <...>V -> B<...>D, B<...>V
    if (buckets_bdim.has_value() && !self_bdim.has_value()) {
      auto self_ = moveBatchDimToFront(self, self_bdim);
      self_ = ensure_has_bdim(self_, self_bdim.has_value(), buckets.size(0));
      auto result = at::searchsorted(buckets, self_, out_int32, right, side, sorter_);
      return std::make_tuple(std::move(result), 0);
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
    auto self_view_ = self_logical_rank == 0 ? self_.unsqueeze(-1) : self_.flatten(1);
    auto result = at::searchsorted(buckets, self_view_, out_int32, right, side, sorter_);
    result = self_logical_rank == 0 ? result.squeeze(-1) : result.view(self_.sizes());
    return std::make_tuple(std::move(result), 0);
  }
  // BD, * -> BD, flat(*) -> BD, B flat(*)
  if (buckets_bdim.has_value() && !self_bdim.has_value()) {
    auto bdim_size = buckets.size(*buckets_bdim);
    auto self_ = ensure_has_bdim(self, false, bdim_size);
    auto self_view_ = self_logical_rank == 0 ? self_.unsqueeze(-1) : self_.flatten(1);
    auto result = at::searchsorted(buckets, self_view_, out_int32, right, side, sorter_);
    result = self_logical_rank == 0 ? result.squeeze(-1) : result.view(self_.sizes());
    return std::make_tuple(std::move(result), 0);
  }
  // D, B* -> no change
  if (!buckets_bdim.has_value() && self_bdim.has_value()) {
    auto result = at::searchsorted(buckets, self, out_int32, right, side, sorter_);
    return std::make_tuple(std::move(result), self_bdim);
  }
  TORCH_INTERNAL_ASSERT(false);
}

static Tensor bucketize_decomp_Tensor(
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  // checking logical rank
  TORCH_CHECK(boundaries.dim() == 1, "bucketize: boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  return at::searchsorted(boundaries, self, out_int32, right, std::nullopt, std::nullopt);
}

static Tensor bucketize_decomp_Scalar(
    const Scalar& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  // checking logical rank
  TORCH_CHECK(boundaries.dim() == 1, "bucketize: boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  return at::searchsorted(boundaries, self, out_int32, right, std::nullopt, std::nullopt);
}

// Use when the other macros don't work out.
// - dim_pos: index of the dim argument
// - keepdim_case: either True, False, or Variable.
//   See NOTE: [keepdim cases] for more details.
// - maybe_keepdim_pos. The index of the keepdim argument,
//   if exists. Otherwise, the value is ignored.
#define REDUCTION_BOXED_ARGS(op, dim_pos, keepdim_case, maybe_keepdim_pos) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction< \
      SINGLE_ARG(boxed_reduction_batch_rule<dim_pos, keepdim_case, maybe_keepdim_pos>)>());

// Provided for your convenience; most operators that have a keepdim arg
// will work with this macro.
// Assumes the dim arg is at position 1 and the keepdim arg is at pos 2.
#define REDUCTION_WITH_KEEPDIM_ARG(op) \
  REDUCTION_BOXED_ARGS(op, 1, KEEPDIM_CASE_VARIABLE, 2)

// Provided for your convenience; most operators that do not have a keepdim
// arg will work with this macro.
// Assumes the dim arg is at position 1 and the operation always returns
// a tensor of the same rank (instead of a smaller rank).
#define REDUCTION_NO_KEEPDIM_ARG(op) \
  REDUCTION_BOXED_ARGS(op, 1, KEEPDIM_CASE_TRUE, -1)

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT2(searchsorted, Tensor, searchsorted_batch_rule);
  REDUCTION_NO_KEEPDIM_ARG(_fft_r2c);
  REDUCTION_NO_KEEPDIM_ARG(_fft_c2r);
  REDUCTION_NO_KEEPDIM_ARG(_fft_c2c);
  REDUCTION_WITH_KEEPDIM_ARG(amax);
  REDUCTION_WITH_KEEPDIM_ARG(amin);
  REDUCTION_WITH_KEEPDIM_ARG(aminmax);
  m.impl("all", all_decomp);
  REDUCTION_WITH_KEEPDIM_ARG(all.dim);
  REDUCTION_WITH_KEEPDIM_ARG(all.dims);
  m.impl("any", any_decomp);
  REDUCTION_WITH_KEEPDIM_ARG(any.dim);
  REDUCTION_WITH_KEEPDIM_ARG(any.dims);
  REDUCTION_WITH_KEEPDIM_ARG(argmax);
  REDUCTION_WITH_KEEPDIM_ARG(argmin);
  m.impl("bucketize.Tensor", bucketize_decomp_Tensor);
  m.impl("bucketize.Scalar", bucketize_decomp_Scalar);
  REDUCTION_BOXED_ARGS(count_nonzero.dim_IntList, 1, KEEPDIM_CASE_FALSE, -1);
  REDUCTION_NO_KEEPDIM_ARG(cummax);
  REDUCTION_NO_KEEPDIM_ARG(cummin);
  REDUCTION_NO_KEEPDIM_ARG(cumprod);
  REDUCTION_NO_KEEPDIM_ARG(cumsum);
  m.impl("dist", dist_decomp);
  REDUCTION_BOXED_ARGS(kthvalue, 2, KEEPDIM_CASE_VARIABLE, 3);
  REDUCTION_BOXED_ARGS(linalg_vector_norm, 2, KEEPDIM_CASE_VARIABLE, 3);
  REDUCTION_NO_KEEPDIM_ARG(logcumsumexp);
  REDUCTION_WITH_KEEPDIM_ARG(logsumexp);
  m.impl("max", max_decomp);
  REDUCTION_WITH_KEEPDIM_ARG(max.dim);
  m.impl("mean", mean_decomp);
  REDUCTION_WITH_KEEPDIM_ARG(mean.dim);
  m.impl("median", median_decomp);
  REDUCTION_WITH_KEEPDIM_ARG(median.dim);
  m.impl("min", min_decomp);
  REDUCTION_WITH_KEEPDIM_ARG(min.dim);
  REDUCTION_WITH_KEEPDIM_ARG(mode);
  m.impl("nanmedian", nanmedian_decomp);
  REDUCTION_WITH_KEEPDIM_ARG(nanmedian.dim);
  REDUCTION_WITH_KEEPDIM_ARG(nansum);
  m.impl("norm.Scalar", norm_scalar_decomp);
  REDUCTION_BOXED_ARGS(norm.ScalarOpt_dim, 2, KEEPDIM_CASE_VARIABLE, 3);
  m.impl("prod", prod_decomp);
  REDUCTION_WITH_KEEPDIM_ARG(prod.dim_int);
  REDUCTION_BOXED_ARGS(std.correction, 1, KEEPDIM_CASE_VARIABLE, 3);
  REDUCTION_NO_KEEPDIM_ARG(_softmax);
  REDUCTION_NO_KEEPDIM_ARG(_safe_softmax);
  REDUCTION_NO_KEEPDIM_ARG(sort);
  REDUCTION_BOXED_ARGS(sort.stable, 2, KEEPDIM_CASE_TRUE, -1);
  REDUCTION_BOXED_ARGS(std_mean.correction, 1, KEEPDIM_CASE_VARIABLE, 3);
  m.impl("sum", sum_decomp);
  REDUCTION_WITH_KEEPDIM_ARG(sum.dim_IntList);
  REDUCTION_BOXED_ARGS(topk, 2, KEEPDIM_CASE_TRUE, -1);
  REDUCTION_BOXED_ARGS(var.correction, 1, KEEPDIM_CASE_VARIABLE, 3);
  REDUCTION_BOXED_ARGS(var_mean.correction, 1, KEEPDIM_CASE_VARIABLE, 3);
  REDUCTION_NO_KEEPDIM_ARG(_log_softmax);
  REDUCTION_BOXED_ARGS(rot90, 2, KEEPDIM_CASE_TRUE, -1);
  VMAP_SUPPORT(_log_softmax_backward_data, _log_softmax_backward_batch_rule);
  VMAP_SUPPORT(_softmax_backward_data, _softmax_backward_batch_rule);
  VMAP_SUPPORT(_is_all_true, _is_all_true_batch_rule);
  VMAP_SUPPORT(_is_any_true, _is_any_true_batch_rule);
}

} // namespace at::functorch
