// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <utility>

#include <ATen/Operators.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/SmallBuffer.h>
#include <ATen/InferSize.h>

namespace at::functorch {

// Note [Adding vmap support for an operator]
// Hey there! So you have an operator and you want to get it to work with vmap.
// For example, let's say you just invented the `sum.int` operator and want to make
// it so that the following works.
// >>> tensor = torch.randn(B, 3)
// >>> vmap(torch.sum, (0, None))(tensor, 0)` works
// There are three main ways to do so.
//
// Note [Writing batch rule for out-of-place operators]
// If your operator is out-of-place, you can write a batch rule for it.
// The batch rule defines how to perform the operator on inputs where each
// Tensor input may have an additional dimension that is being vmapped over.
// We refer to this dimension as the *batch dimension* or bdim for short.
//
// For example, let's consider writing a batch rule for
// `Tensor sum(const Tensor& self, int64_t dim)`. The signature of the
// batch rule has an additional std::optional<int64_t> argument after each
// Tensor argument and return. So, in this case, the batch rule has signature
//   tuple<Tensor, std::optional<int64_t>> sum_batch_rule(
//       const Tensor& self, std::optional<int64_t> self_bdim, int64_t dim);
//
// The vmap call above invokes the batch rule with `self = tensor`,
// `self_bdim = 0`, and `dim = 0`. Note that there are **no BatchedTensors**
// involved in this case; there exists some plumbing that automatically unwraps
// BatchedTensors before calling the batch rule.
//
// To write the logic of the batch rule: think about the semantics of the
// `sum` operation if `self` had an additional dimension (indicated by self_bdim):
// - If `self_bdim` is null, then we just do `result = self.sum(dim)` as usual
// - If `self_bdim` is not-null, then we need to modify `dim`. `dim` is equal
//   to whatever the user passed in (0 in this case), but we should actually
//   perform the reduction over dimension 1 and do `result = self.sum(1)`
//   because dim 0 is being vmapped over.
// Finally, we return the result as well as a new bdim
// - If `self_bdim` is null, then there's no batch dim in the result.
// - If `self_bdim` is not-null, then we return where the bdim is.
//   Since we invoked `result = self.sum(1)`, the bdim is still at dim 0.
//
// Now that we have written `sum_batch_rule`, we have to register it inside a
// TORCH_LIBRARY_IMPL block:
//   TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
//     ...
//     VMAP_SUPPORT2(sum, int, sum_batch_rule);
//     ...
//   }
//
// Note [Reusing batch rules to add vmap support for a complicated operator]
// Can't figure out how to write a batch rule for a big operation? If the
// operation can be expressed as a composition of other operations that do have
// batch rules, then that is another way to add vmap support. For example,
// consider the following schema
//   func: addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1)
// and assume we already have batching rules for basic arithmetic operators.
//
// To add vmap support, define a decomposition using the same signature:
//   Tensor addcmul_decomp(const Tensor& self, const Tensor& tensor1,
//                         const Tensor& tensor2, const Scalar& value) {
//     auto product = torch.mul(tensor1, tensor2);
//     return torch.add(self, product, value);
//   }
// And register it inside a TORCH_LIBRARY_IMPL block:
//   TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
//     ...
//     m.impl("addcmul", addcmul_decomp);
//     ...
//   }
//
// Note [Writing batch rule for in-place operators]
// TODO: This is kinda complicated. Saving this for a future date.

namespace{

std::tuple<Tensor, std::optional<int64_t>> unsqueeze_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    int64_t dim) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto rank = rankWithoutBatchDim(self, self_bdim);
  dim = maybe_wrap_dim(dim, rank + 1) + 1;
  return std::make_tuple(self_.unsqueeze(dim), 0);
}

// NB: repeat is not actually a view, but it is in this file
std::tuple<Tensor, std::optional<int64_t>> repeat_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    c10::SymIntArrayRef sizes) {

  SymDimVector sizes_with_bdim = { sizes.begin(), sizes.end() };
  sizes_with_bdim.insert(sizes_with_bdim.begin(), 1);
  auto self_ = moveBatchDimToFront(self, self_bdim);
  while (self_.dim() < (int64_t)sizes_with_bdim.size()) {
    self_ = self_.unsqueeze(1);
  }
  return std::make_tuple(self_.repeat_symint(sizes_with_bdim), 0);
}


std::tuple<Tensor, std::optional<int64_t>> _unsafe_view_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    c10::SymIntArrayRef size) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  SymDimVector view_size(size);
  view_size.insert(view_size.begin(), self_.sym_size(0));

  // See if the view is valid. If it's not, then we copy.
  // It's OK to copy, because _unsafe_view(x) guarantees that x isn't used
  // anymore.
  const at::SymDimVector inferred_size = at::infer_size_dv(view_size, self_.sym_numel());
  const auto stride = at::detail::computeStride(self_.sym_sizes(),
                                                self_.sym_strides(),
                                                inferred_size);
  if (!stride.has_value()) {
    self_ = self_.contiguous();
  }
  return std::make_tuple(at::_unsafe_view_symint(self_, view_size), 0);
}

std::tuple<Tensor, std::optional<int64_t>> flip_batch_rule(const Tensor& self, std::optional<int64_t> self_bdim, IntArrayRef dims) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  VmapDimVector new_dims;
  for (auto i: dims) {
    new_dims.push_back(getPhysicalDim(self_, true, i));
  }
  return std::make_tuple(at::flip(self_, new_dims), 0);
}

const Tensor& resize__plumbing(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value() ||
      optional_memory_format == c10::MemoryFormat::Contiguous,
      "resize_: batching rule only supports None or Contiguous MemoryFormat");
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "resize__plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    c10::impl::ExcludeDispatchKeyGuard guard2(DispatchKey::FuncTorchBatched);
    return self.resize_(size, optional_memory_format);
  }

  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  TORCH_INTERNAL_ASSERT(self_bdim.has_value());

  // TODO: The following algorithm only works for batch dim == 0.
  // To get it to work for something else we need the ability to modify
  // the BatchDims attribute of BatchedTensorImpl
  TORCH_INTERNAL_ASSERT(self_bdim == 0, "NYI: resize_ batch rule for batch dim != 0");

  // Resize the wrapped tensor
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  self_value = moveBatchDimToFront(self_value, self_bdim);
  VmapDimVector new_size(size);
  new_size.insert(new_size.begin(), self_value.size(*self_bdim));
  self_value.resize_(new_size);

  // Update the sizes and strides of the wrapper
  auto* batched = maybeGetBatchedImpl(self);
  TORCH_INTERNAL_ASSERT(batched);
  batched->refreshTensorMetadata();

  return self;
}

std::tuple<Tensor, std::optional<int64_t>> squeeze_batch_rule(const Tensor& self, std::optional<int64_t> bdim) {
  TORCH_INTERNAL_ASSERT(bdim.has_value());
  // Special case for scalar arrays to replicate PyTorch behavior.
  if (self.dim() == 1) {
    return std::make_tuple(self.alias(), bdim);
  }

  // Manually calculate the output shape by eliding all dimensions of
  // size 1 keeping track of where the batch index started and where it
  // ended up moving to. We also ensure we do not drop the batch index.
  auto shape = self.sym_sizes();
  SymDimVector squeezed_sizes;
  bool before_batch_idx = true;
  int64_t new_batch_idx = 0;
  int64_t original_idx = 0;

  for (const auto& it : shape) {
    // Keep only dimensions != 1 and the batch dimension (irrespective of size).
    if (it != 1 || original_idx == bdim) {
      squeezed_sizes.push_back(it);
      if (original_idx == bdim) {
        before_batch_idx = false;
      }
      // Only increment for the dimensions that will be kept in the output.
      if (before_batch_idx) {
        ++new_batch_idx;
      }
    }
    ++original_idx;
  }

  auto result = self.view_symint(squeezed_sizes);
  return std::make_tuple(std::move(result), std::optional<int64_t>(new_batch_idx));
}

std::tuple<Tensor, std::optional<int64_t>> squeeze_dims_batch_rule(
    const Tensor& self, std::optional<int64_t> bdim, IntArrayRef dims) {
  TORCH_INTERNAL_ASSERT(bdim.has_value());
  // Special case for scalar arrays to replicate PyTorch behavior.
  auto ndim = self.dim();
  if (ndim == 1) {
    TORCH_CHECK(
        dims.empty() || (dims.size() == 1 && dims[0] == 0),
        "Dimension is out of range (expected to be in range of [-1, 0], but got ", dims);
    return std::make_tuple(self.alias(), bdim);
  }

  // Adjust any dimensions higher than the batch dimension
  DimVector adjusted_dims(dims.begin(), dims.end());
  int64_t updated_batch_idx = *bdim;
  for (auto &d : adjusted_dims) {
    auto actual_dim = c10::maybe_wrap_dim(d, ndim - 1);
    if (actual_dim < *bdim) {
      d = actual_dim;
      if (self.sym_size(actual_dim) == 1) {
        // A column before batch dimension will be dropped so adjust accordingly.
        --updated_batch_idx;
      }
    } else {
      // Since dimension to be squeezed is after the batch dimension adjust by one to account
      // for the original batch dimension. In this case batch dimension won't move.
      d = actual_dim + 1;
    }
  }
  return std::make_tuple(self.squeeze(adjusted_dims), std::optional<int64_t>(updated_batch_idx));
}

std::tuple<Tensor, std::optional<int64_t>> squeeze_dim_batch_rule(
    const Tensor& self, std::optional<int64_t> bdim, int64_t dim) {
  return squeeze_dims_batch_rule(self, bdim, {dim});
}

std::tuple<Tensor, std::optional<int64_t>> select_batching_rule(const Tensor& self, std::optional<int64_t> bdim, int64_t dim, c10::SymInt index) {
  if (!bdim) {
    return std::make_tuple(self.select_symint(dim, std::move(index)), std::nullopt);
  }

  auto _self = moveBatchDimToFront(self, bdim);
  auto dim_physical = getPhysicalDim(_self, true, dim);
  auto result = _self.select_symint(dim_physical, std::move(index));
  return std::make_tuple(std::move(result), 0);
}

std::tuple<Tensor, std::optional<int64_t>> _reshape_alias_batch_rule(const Tensor& self, std::optional<int64_t> bdim, const c10::SymIntArrayRef shape, const c10::SymIntArrayRef strides) {
  (void) strides;
  TORCH_INTERNAL_ASSERT(bdim.has_value());

  auto self_ = moveBatchDimToFront(self, bdim);
  c10::SymDimVector new_shape(shape.size() + 1);
  new_shape[0] = self_.sym_size(0);
  std::copy(shape.begin(), shape.end(), new_shape.begin() + 1);
  return std::make_tuple(at::reshape_symint(self_, new_shape), 0);
}

std::tuple<Tensor, optional<int64_t>> _lazy_clone_alias_batch_rule(
    const Tensor &self, optional<int64_t> bdim) {
  TORCH_INTERNAL_ASSERT(bdim.has_value());
  auto self_ = moveBatchDimToFront(self, bdim);
  return std::make_tuple(at::_lazy_clone_alias(self), 0);
}

std::tuple<Tensor, std::optional<int64_t>> roll_batch_rule(const Tensor& self, std::optional<int64_t> bdim, SymIntArrayRef shifts, IntArrayRef dims) {
  TORCH_INTERNAL_ASSERT(bdim.has_value());

  auto self_ = moveBatchDimToFront(self, bdim);
  VmapDimVector new_dims;
  if (!dims.empty()) {
    for (auto i: dims) {
      new_dims.push_back(getPhysicalDim(self, true, i));
    }
    return std::make_tuple(at::roll_symint(self_, shifts, new_dims), 0);
  }
  // We will do something like: t.reshape(a, -1).roll(1, dims=[1, ]).reshape(old_shape)
  auto old_shape = self_.sym_sizes();
  new_dims.push_back(1);
  auto logical_rank = rankWithoutBatchDim(self, bdim);
  if (logical_rank == 0) {
    self_ = self_.unsqueeze(0);
  }

  auto output = at::roll_symint(self_.flatten(1), shifts, new_dims);
  // NOTE: For scalar tensor, we don't need to unsqueeze as reshape
  // with `old_shape` takes care of it.
  output = output.reshape_symint(old_shape);
  return std::make_tuple(output, 0);
}

std::tuple<Tensor, std::optional<int64_t>> diagonal_batching_rule(
    const Tensor &self, std::optional<int64_t> self_bdim,
    int64_t offset, int64_t dim1, int64_t dim2)
{
  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto dim1_ = maybe_wrap_dim(dim1, logical_rank) + 1;
  auto dim2_ = maybe_wrap_dim(dim2, logical_rank) + 1;
  auto result = at::diagonal(self_, offset, dim1_, dim2_);
  return std::make_tuple(std::move(result), 0);
}

std::tuple<Tensor, std::optional<int64_t>> diagonal_backward_batch_rule(
    const Tensor& grad_input, std::optional<int64_t> grad_input_bdim,
    c10::SymIntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  auto logical_rank = rankWithoutBatchDim(grad_input, grad_input_bdim);
  auto grad_input_ = moveBatchDimToFront(grad_input, grad_input_bdim);
  dim1 = maybe_wrap_dim(dim1, logical_rank + 1) + 1;
  dim2 = maybe_wrap_dim(dim2, logical_rank + 1) + 1;
  c10::SymDimVector input_sizes_(input_sizes.size() + 1);
  input_sizes_[0] = grad_input_.size(0);
  std::copy(input_sizes.begin(), input_sizes.end(), input_sizes_.begin() + 1);
  auto result = at::diagonal_backward_symint(grad_input_, input_sizes_, offset, dim1, dim2);
  return std::make_tuple(std::move(result), 0);
}

std::tuple<Tensor, std::optional<int64_t>> slice_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    int64_t dim,
    std::optional<c10::SymInt> start,
    std::optional<c10::SymInt> end,
    c10::SymInt step) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  dim = getPhysicalDim(self, self_bdim.has_value(), dim);

  auto result = self_.slice_symint(dim, std::move(start), std::move(end), std::move(step));
  return std::make_tuple(std::move(result), 0);
}

static bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}

std::tuple<Tensor, std::optional<int64_t>>
transpose_int_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    int64_t dim0,
    int64_t dim1) {
  // PyTorch has a special case where scalar_tensor.transpose(dim0, dim1) works
  // for dim0, dim1 in {0, -1} and returns the scalar tensor. If the following happens:
  // >>> x = torch.randn(B0)  # the per-examples are all scalars
  // >>> vmap(lambda x: x.transpose(0, -1), x)
  // then we replicate this behavior.
  if (/*physical*/self.dim() == 1 && is_allowed_dim_on_scalar_tensor(dim0) &&
      is_allowed_dim_on_scalar_tensor(dim1)) {
    return std::make_tuple(self, self_bdim);
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  dim0 = getPhysicalDim(self, self_bdim.has_value(), dim0);
  dim1 = getPhysicalDim(self, self_bdim.has_value(), dim1);
  auto result = self_.transpose(dim0, dim1);
  return std::make_tuple(std::move(result), 0);
}

std::tuple<Tensor, std::optional<int64_t>> permute_batching_rule(
    const Tensor &self, std::optional<int64_t> self_bdim, IntArrayRef dims)
{
  if (!self_bdim.has_value()) {
    return std::make_tuple(self.permute(dims), self_bdim);
  }

  auto self_ = moveBatchDimToFront(self, self_bdim);
  VmapDimVector dims_;
  dims_.reserve(dims.size() + 1);
  dims_.emplace_back(0);
  for (auto dim : dims) {
    dims_.emplace_back(getPhysicalDim(self_, self_bdim.has_value(), dim));
  }

  return std::make_tuple(self_.permute(dims_), 0);
}

std::tuple<Tensor, std::optional<int64_t>> select_backward_batch_rule(
    const Tensor& grad_input, std::optional<int64_t> grad_input_bdim,
    c10::SymIntArrayRef input_sizes, int64_t dim, c10::SymInt index) {
  auto logical_rank = rankWithoutBatchDim(grad_input, grad_input_bdim);
  auto grad_input_ = moveBatchDimToFront(grad_input, grad_input_bdim);
  dim = maybe_wrap_dim(dim, logical_rank + 1) + 1;
  c10::SymDimVector input_sizes_(input_sizes.size() + 1);
  input_sizes_[0] = grad_input_.sym_size(0);
  std::copy(input_sizes.begin(), input_sizes.end(), input_sizes_.begin() + 1);
  auto result = at::select_backward_symint(grad_input_, input_sizes_, dim, std::move(index));
  return std::make_tuple(std::move(result), 0);
}

std::tuple<Tensor, std::optional<int64_t>> slice_backward_batch_rule(
    const Tensor& grad_input, std::optional<int64_t> grad_input_bdim,
    SymIntArrayRef input_sizes, int64_t dim, c10::SymInt start, c10::SymInt end, c10::SymInt step) {
  auto logical_rank = rankWithoutBatchDim(grad_input, grad_input_bdim);
  auto grad_input_ = moveBatchDimToFront(grad_input, grad_input_bdim);
  dim = maybe_wrap_dim(dim, logical_rank) + 1;
  c10::SymDimVector input_sizes_(input_sizes.size() + 1);
  input_sizes_[0] = grad_input_.size(0);
  std::copy(input_sizes.begin(), input_sizes.end(), input_sizes_.begin() + 1);
  auto result = at::slice_backward_symint(grad_input_, input_sizes_, dim, std::move(start), std::move(end), std::move(step));
  return std::make_tuple(std::move(result), 0);
}

std::tuple<Tensor, std::optional<int64_t>> view_batching_rule(
    const Tensor &self, std::optional<int64_t> self_bdim, SymIntArrayRef sym_size)
{
  TORCH_INTERNAL_ASSERT(self_bdim.has_value());
  auto self_ = moveBatchDimToFront(self, self_bdim);
  c10::SmallVector<c10::SymInt> size_(sym_size.size() + 1);
  // copy batch size
  size_[0] = self_.sym_size(0);
  std::copy(sym_size.cbegin(), sym_size.cend(), size_.begin() + 1);
  return std::make_tuple(self_.view_symint(size_), 0);
}

std::tuple<Tensor, std::optional<int64_t>> view_copy_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    c10::SymIntArrayRef size) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  SymDimVector view_size(size.size() + 1);
  view_size[0] = self_.size(0);
  std::copy(size.cbegin(), size.cend(), view_size.begin() + 1);

  return std::make_tuple(at::view_copy_symint(self_, view_size), 0);
}

template <typename F, F Func>
std::tuple<Tensor, std::optional<int64_t>> expand_batch_rule(
    const Tensor &self, std::optional<int64_t> self_bdim, SymIntArrayRef size, bool implicit)
{
  auto self_dim = self.dim();
  TORCH_CHECK(static_cast<uint64_t>(self_dim - 1) <= size.size(),
              "expand: the number of sizes provided (", size.size(), ") ",
              "must be greater or equal to the number of dimensions in the tensor (", static_cast<uint64_t>(self_dim - 1), ")");

  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto self_sizes = self_.sym_sizes();
  const auto& batch_size = self_sizes[0];

  c10::SmallVector<c10::SymInt> size_(size.size() + 1);
  size_[0] = batch_size;
  std::copy(size.cbegin(), size.cend(), size_.begin() + 1);

  // Here, we know we are expanding a (logical) tensor to a larger number
  // of dimensions. We have to be careful because we can't call expand directly
  // due to the presence of batch dimensions.
  //
  // As an example, let B0 be a batch dimension and consider expand(Tensor[B0, 3], [2, 3]).
  // The result should be a tensor of size [B0, 2, 3].
  // A physical view of size [B0, 3] can't directly be expanded to size [B0, 2, 3]
  // so the strategy here is to view it first as a tensor of size [B0, 1, 3] and
  // then expand.
  auto extra_dims = size.size() - (self_dim - 1);
  c10::SmallVector<c10::SymInt> view_shape(size_.size(), /*init_value*/1);
  view_shape[0] = batch_size;
  std::copy(self_sizes.cbegin() + 1, self_sizes.cend(),
            view_shape.begin() + 1 + extra_dims);

  return std::make_tuple(Func(self_.view_symint(view_shape), size_, implicit), 0);
}

std::tuple<Tensor, std::optional<int64_t>> unfold_batch_rule(
    const Tensor &self, std::optional<int64_t> self_bdim, int64_t dim, int64_t size, int64_t step)
{
  TORCH_INTERNAL_ASSERT(self_bdim.has_value());
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  dim = maybe_wrap_dim(dim, logical_rank) + 1;
  if (logical_rank==0) {
    self_ = self_.unsqueeze(-1);
  }
  auto result = self_.unfold(dim, size, step);
  if (logical_rank==0) {
    result = result.squeeze(-1);
  }
  return std::make_tuple(std::move(result), 0);
}

std::tuple<Tensor, std::optional<int64_t>> narrow_copy_batch_rule(
    const Tensor &self, std::optional<int64_t> self_bdim, int64_t dim, c10::SymInt start, c10::SymInt length)
{
  TORCH_INTERNAL_ASSERT(self_bdim.has_value());
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  dim = maybe_wrap_dim(dim, logical_rank) + 1;
  auto result = self_.narrow_copy_symint(dim, std::move(start), std::move(length));

  return std::make_tuple(std::move(result), 0);
}

std::tuple<std::vector<Tensor>, std::optional<int64_t>> unsafe_split_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    c10::SymInt split_size,
    int64_t dim) {
  TORCH_INTERNAL_ASSERT(self_bdim.has_value());
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  dim = maybe_wrap_dim(dim, logical_rank) + 1;
  auto result = self_.unsafe_split_symint(std::move(split_size), dim);
  return std::make_tuple(std::move(result), 0);
}

std::tuple<Tensor, std::optional<int64_t>> diag_embed_batch_rule(const Tensor& self, std::optional<int64_t> self_bdim, int64_t offset, int64_t dim1, int64_t dim2) {
  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim);
  dim1 = maybe_wrap_dim(dim1, logical_rank + 1) + 1;
  dim2 = maybe_wrap_dim(dim2, logical_rank + 1) + 1;
  return std::make_tuple(at::diag_embed(self_, offset, dim1, dim2), 0);
}

Tensor trace_decomp(const Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 2, "trace: expected a matrix, but got tensor with dim ", tensor.dim());
  return tensor.diagonal().sum();
}

std::tuple<Tensor, std::optional<int64_t>> tril_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    int64_t diagonal = 0) {
  TORCH_CHECK(self.dim() >= 2, "tril: The input tensor must have at least 2 dimensions.");
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto result = at::tril(self_, diagonal);
  return std::make_tuple(std::move(result), 0);
}

std::tuple<Tensor, std::optional<int64_t>> triu_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    int64_t diagonal = 0) {
  TORCH_CHECK(self.dim() >= 2, "triu: The input tensor must have at least 2 dimensions.");
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto result = at::triu(self_, diagonal);
  return std::make_tuple(std::move(result), 0);
}

}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT(flip, flip_batch_rule);
  m.impl("trace", trace_decomp);
  VMAP_SUPPORT(tril, tril_batch_rule);
  VMAP_SUPPORT(triu, triu_batch_rule);
  VMAP_SUPPORT(repeat, repeat_batch_rule);
  VMAP_SUPPORT(_unsafe_view, _unsafe_view_batch_rule);
  VMAP_SUPPORT(unsqueeze, unsqueeze_batch_rule);
  m.impl("resize_", resize__plumbing);
  VMAP_SUPPORT2(select, int, select_batching_rule);
  VMAP_SUPPORT(squeeze, squeeze_batch_rule);
  VMAP_SUPPORT2(squeeze, dim, squeeze_dim_batch_rule);
  VMAP_SUPPORT2(squeeze, dims, squeeze_dims_batch_rule);
  VMAP_SUPPORT(_reshape_alias, _reshape_alias_batch_rule);
  VMAP_SUPPORT(_lazy_clone_alias, _lazy_clone_alias_batch_rule);
  VMAP_SUPPORT(roll, roll_batch_rule);
  VMAP_SUPPORT(permute, permute_batching_rule);
  VMAP_SUPPORT(diagonal, diagonal_batching_rule);
  VMAP_SUPPORT(diagonal_backward, diagonal_backward_batch_rule);
  VMAP_SUPPORT(select_backward, select_backward_batch_rule);
  VMAP_SUPPORT(slice_backward, slice_backward_batch_rule);
  VMAP_SUPPORT(view, view_batching_rule);
  VMAP_SUPPORT(view_copy, view_copy_batch_rule);
  VMAP_SUPPORT(expand, SINGLE_ARG(expand_batch_rule<decltype(&ATEN_FN(expand)), &ATEN_FN(expand)>));
  VMAP_SUPPORT(expand_copy, SINGLE_ARG(expand_batch_rule<decltype(&ATEN_FN(expand_copy)), &ATEN_FN(expand_copy)>));
  VMAP_SUPPORT(unfold, unfold_batch_rule);
  VMAP_SUPPORT2(slice, Tensor, slice_batch_rule);
  VMAP_SUPPORT2(transpose, int, transpose_int_batch_rule);
  m.impl("t", native::t);  // CompositeExplicitAutograd, should not go in BatchRulesDecompositions.cpp
  m.impl("t_", native::t_);  // CompositeExplicitAutograd, should not go in BatchRulesDecompositions.cpp
  VMAP_SUPPORT(diag_embed, diag_embed_batch_rule);
  VMAP_SUPPORT(narrow_copy, narrow_copy_batch_rule);
  VMAP_SUPPORT2(unsafe_split, Tensor, unsafe_split_batch_rule);
}

} // namespace at::functorch
