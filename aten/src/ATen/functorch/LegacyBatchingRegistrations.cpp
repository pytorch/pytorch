// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/library.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/ATen.h>
#include <ATen/native/TensorShape.h>

#include <ATen/NestedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/BatchingMetaprogramming.h>
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/BatchRulesHelper.h>

#include <utility>

namespace at {
namespace functorch {


// NOTE: [What is a batching rule?]
//
// NB: the following description only applies to this file and is about
// the legacy (deprecated) batching rule API. Please see writing_batch_rules.md
// for how to write new-style batching rules.
//
// This files contains batching rules written with the legacy (now-deprecated)
// batching rule API.
// Please try to use the new-style batching rule API (see writing_batch_rules.md)
//
// A *batching rule* implements the logic of how to call an operator on inputs
// that have zero or more additional batch dimensions. When one does a vmap, the
// dimension(s) being vmap'ed over get recorded as batch dimensions.
//
// For example, vmap(torch.add)(x, y)
// 1. wraps `x` into batched_x = BatchedTensor(x, bdims=[(lvl=1, dim=0)];
// 2. wraps `y` into batched_y = BatchedTensor(y, bdims=[(lvl=1, dim=0)];
// 3. and then runs `torch.add(batched_x, batched_y)`.

// NOTE: [When should I add a batching rule?]
// When you are adding a new operator, you'll need to add a batching rule so
// that vmap can work efficiently with said operator. If you do not, we'll attempt
// to generate a slow fallback for the batching rule.

// NOTE: [How to write batching rules?]
// The signature of a batching rule should look like exactly like the C++ signature
// of its operator.
//
// First, see NOTE: [Logical vs physical args] in VmapTransforms.h for terminology.
//
// At a high level, what a batching rule does is the following:
// 1. Converts (logical) BatchedTensors to views on physical tensors.
// 2. Converts logical arguments (e.g. dimension indexes, shapes) to physical
//    arguments that correspond to the physical tensors.
// 3. Calls at:: operations on the physical tensors and arguments to produce
//    some physical results.
// 4. Converts physical results back to BatchedTensors.
//
// Steps 1, 2, and 4 differ for operators with different batching behaviors. When
// writing a new batching rule, please select a VmapTransform that matches the
// batching behavior of your operation. The VmapTransform provides helper functions
// to do steps (1), (2), and (4).
// (see NOTE: [What is an VmapTransform?] in VmapTransforms.h)

namespace{
// PyTorch allows operations to specify dim 0 and dim -1 on a scalar tensor.
static bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}

static int64_t get_current_level() {
  auto maybe_level = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_level.has_value());
  return maybe_level->layerId();
}

// This check should probably go into the dispatcher...
static bool participatesInCurrentLevel(const Tensor& self) {
  auto current_level = get_current_level();
  auto* maybe_batched_impl = maybeGetBatchedImpl(self);
  if (!maybe_batched_impl) {
    return false;
  }
  auto self_level = maybe_batched_impl->level();
  TORCH_INTERNAL_ASSERT(self_level <= current_level);
  return self_level == current_level;
}

static bool participatesInCurrentLevel(ITensorListRef self) {
  for (const Tensor& tensor : self) {
    if (participatesInCurrentLevel(tensor)) {
      return true;
    }
  }
  return false;
}

std::vector<Tensor> tensor_split_sections_batching_rule(const Tensor& self, int64_t sections, int64_t dim) {
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::tensor_split(self, sections, dim);
  }
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::tensor_split(self_physical.tensor(), sections, dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

std::vector<Tensor> tensor_split_indices_batching_rule(const Tensor& self, IntArrayRef indices, int64_t dim) {
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::tensor_split(self, indices, dim);
  }
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::tensor_split(self_physical.tensor(), indices, dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

Tensor& squeeze_dims__batching_rule(Tensor& self, IntArrayRef dims) {
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return self.squeeze_(dims);
  }
  auto* batched = maybeGetBatchedImpl(self);
  const auto bdim = batched->bdim();
  auto logical_dim = self.dim();

  if (logical_dim == 0) {
    TORCH_CHECK(
        dims.empty() || (dims.size() == 1 && dims[0] == 0),
        "Dimension is out of range (expected to be in range of [-1, 0], but got ", dims);
    return self;
  }

  // Adjust any dimensions higher than the batch dimension
  DimVector adjusted_dims(dims.begin(), dims.end());
  int64_t updated_batch_idx = bdim;
  for (auto &d : adjusted_dims) {
    auto actual_dim = c10::maybe_wrap_dim(d, logical_dim);
    if (actual_dim < bdim) {
      d = actual_dim;
      if (batched->value().sym_size(actual_dim) == 1) {
        // A column before batch dimension will be dropped so adjust accordingly.
        --updated_batch_idx;
      }
    } else {
      // Since dimension to be squeezed is after the batch dimension adjust by one to account
      // for the original batch dimension. In this case batch dimension won't move.
      d = actual_dim + 1;
    }
  }

  batched->value().squeeze_(adjusted_dims);
  if (updated_batch_idx != bdim) {
    batched->unsafe_set_bdim(updated_batch_idx);
  }
  batched->refreshTensorMetadata();
  return self;
}

Tensor& squeeze_dim__batching_rule(Tensor& self, int64_t dim) {
  return squeeze_dims__batching_rule(self, {dim});
}

Tensor& squeeze__batching_rule(Tensor& self) {
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return self.squeeze_();
  }
  auto* batched = maybeGetBatchedImpl(self);

  // Need to find out how many dimensions of size 1 are before the bdim
  const auto bdim = batched->bdim();
  const auto physical_shape = batched->value().sizes();
  auto how_many_dims_of_size_1_before_bdim = 0;
  for (const auto i : c10::irange(0, physical_shape.size())) {
    if ((int64_t)i == bdim) {
      break;
    }
    if (physical_shape[i] == 1) {
      how_many_dims_of_size_1_before_bdim++;
    }
  }

  int64_t new_bdim = bdim - how_many_dims_of_size_1_before_bdim;
  if (physical_shape[bdim] != 1) {
    // if bdim is not 1, can just call squeeze_()
    batched->value().squeeze_();
  } else {
    // otherwise, squeeze_() is going to get rid of the bdim too.
    // We "fix it up" by calling unsqueeze_.
    batched->value().squeeze_();
    batched->value().unsqueeze(new_bdim);
  }

  // Refresh metadata
  batched->unsafe_set_bdim(new_bdim);
  batched->refreshTensorMetadata();
  return self;
}

Tensor& unsqueeze__batching_rule(Tensor& self, int64_t dim) {
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return self.unsqueeze_(dim);
  }
  auto* batched = maybeGetBatchedImpl(self);
  auto logical_dim = self.dim();
  int64_t dim_physical = maybe_wrap_dim(dim, logical_dim + 1);
  if (dim_physical >= batched->bdim()) {
    dim_physical = 1 + dim_physical;
  } else {
    batched->unsafe_set_bdim(batched->bdim() + 1);
  }
  batched->value().unsqueeze_(dim_physical);

  // Also need to change some metadata...
  batched->refreshTensorMetadata();
  return self;
}

Tensor& transpose__batching_rule(Tensor& self, int64_t dim0, int64_t dim1) {
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return self.transpose_(dim0, dim1);
  }
  auto* batched = maybeGetBatchedImpl(self);
  auto logical_dim = self.dim();

  // PyTorch has a special case where scalar_tensor.transpose(dim0, dim1) works
  // for dim0, dim1 in {0, -1} and returns the scalar tensor. If the following happens:
  // >>> x = torch.randn(B0)  # the per-examples are all scalars
  // >>> vmap(lambda x: x.transpose_(0, -1), x)
  // then we replicate this behavior.
  if (logical_dim == 0 &&
      is_allowed_dim_on_scalar_tensor(dim0) &&
      is_allowed_dim_on_scalar_tensor(dim1)) {
    // No transposing happened :P
    return self;
  }

  dim0 = maybe_wrap_dim(dim0, logical_dim);
  dim1 = maybe_wrap_dim(dim1, logical_dim);

  dim0 = dim0 >= batched->bdim() ? dim0 + 1 : dim0;
  dim1 = dim1 >= batched->bdim() ? dim1 + 1 : dim1;
  batched->value().transpose_(dim0, dim1);

  // Also need to change some metadata...
  batched->refreshTensorMetadata();
  return self;
}

std::vector<Tensor> split_batching_rule(const Tensor& self, int64_t split_size, int64_t dim) {
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::split(self, split_size, dim);
  }
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::split(self_physical.tensor(), split_size, dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

std::vector<Tensor> split_with_sizes_batching_rule(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return split_with_sizes(self, split_sizes, dim);
  }
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = split_with_sizes(self_physical.tensor(), split_sizes, dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

std::vector<Tensor> unbind_batching_rule(const Tensor& self, int64_t dim) {
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::unbind(self, dim);
  }
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::unbind(self_physical.tensor(), dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

// given (sizes, strides, storage_offset) returns the maximum location that
// can be indexed (or nullopt if such a location doesn't exist, e.g., tensors
// with zero-size dims).
static optional<c10::SymInt> maximum_indexable_location(
    c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides, c10::SymInt storage_offset) {
  auto result = native::storage_size_for(sizes, strides);
  if (result == 0) {
    return nullopt;
  }
  return result + storage_offset;
}

// Let x be the "first slice" of physical_tensor.
// This checks that the range of possible memory locations accessible by
// x.as_strided(sizes, strides, maybe_storage_offset)
// are within the bounds of possible memory locations accessible by x.
static void checkBasicAsStridedValidForSlice(
    const Tensor& physical_tensor,
    int64_t num_batch_dims,
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    optional<c10::SymInt> maybe_storage_offset) {
  auto slice_sizes = physical_tensor.sym_sizes().slice(num_batch_dims);
  auto slice_strides = physical_tensor.sym_strides().slice(num_batch_dims);
  auto base_offset = physical_tensor.sym_storage_offset();

  auto storage_offset = maybe_storage_offset.value_or(base_offset);

  auto max_as_strided_loc = maximum_indexable_location(sizes, strides, storage_offset);
  auto max_slice_loc = maximum_indexable_location(slice_sizes, slice_strides, base_offset);

  if (!max_as_strided_loc.has_value()) {
    return;
  }
  if (!max_slice_loc.has_value()) {
    TORCH_CHECK(false,
        "result = tensor.as_strided(", sizes, ", ",  strides, ", ", storage_offset, ") ",
        "can access memory outside of `tensor`. `tensor` has no storage but the ",
        "passed-in (size, stride, storage_offset) imply a result with some storage. ",
        "This is not supported inside of vmap, please try to rewrite the ",
        "`as_strided` call as a sequence of PyTorch view operations");
  }

  TORCH_CHECK(
      *max_as_strided_loc <= *max_slice_loc && base_offset <= storage_offset,
      "result = tensor.as_strided(", sizes, ", ",  strides, ", ", storage_offset, ") ",
      "can access memory outside of `tensor`. `result` can access some ",
      "memory in range [", storage_offset, ", ", *max_as_strided_loc, "], but ",
      "`tensor` can only access some memory in range [", base_offset, ", ",
      *max_slice_loc, "]. This is not supported inside of vmap, please try to ",
      "rewrite the `as_strided` call as a sequence of PyTorch view operations");
}

// What are the semantics of as_strided inside of vmap?
// y = vmap(lambda x: x.as_strided(sizes, strides, offset))(xs)
// This returns a view on `x`, `y`, such that each y[i] has:
// - sizes: `sizes`
// - strides: `strides`
// - storage_offset: offset + i * x.stride(batch_dim)
//
// In other words, it is as if we had treated each x[i] as having storage
// offset equal to xs.offset() and called as_strided(sizes, sizes, offset).
// (that is equivalent to x[i].as_strided(
//    sizes, sizes, offset + x[i].storage_offset() - xs.offset()) for all i)
//
// Note that this *may* be different from actually running as_strided
// in a for-loop. This is due to how as_strided takes in `offset` to be
// an *absolute* offset. As an example, consider:
// >>> x = torch.tensor([0., 1., 2., 3., 4.]).as_strided([4], [1], 1)
// >>> z = [x[i].as_strided([1], [1], 1) for i in range(4)]
// Each z[i] is actually the same view on x (z[i] == torch.tensor([1.]))!
// However, we consider the above for-loop comprehension to be a user error:
// a user should have written the following if they wanted to use as_strided
// in a per-sample way:
// >>> z = [x[i].as_strided([1], [1], 1 + x[i].storage_offset() - 1) for i in range(4)]
Tensor as_strided_batching_rule(
    const Tensor& tensor,
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    optional<c10::SymInt> storage_offset) {
  if (!participatesInCurrentLevel(tensor)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::as_strided_symint(tensor, sizes, strides, std::move(storage_offset));
  }
  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(tensor);
  auto num_batch_dims = physical_view.numBatchDims();
  auto physical_sizes = physical_view.getPhysicalShape(sizes);
  const auto& physical_tensor = physical_view.tensor();

  // We can't rely on the physical as_strided call to do this for us because
  // we do some sanity checks on the size/strides before calling into as_strided.
  TORCH_CHECK(sizes.size() == strides.size(),
      "Tensor.as_strided(size, stride, ...): size and stride must have the ",
      "same length! Got size ", sizes, " and stride ", strides);

  // Sanity checks:
  // 1. as_strided(sizes, strides, storage_offset + tensor[i].offset() - tensor.offset())
  // is valid for a slice of the input tensor.
  // See Note: [When will the as_strided batching rule fail?] for details.
  checkBasicAsStridedValidForSlice(
      physical_tensor, num_batch_dims, sizes, strides, storage_offset);

  // physical_strides = physical tensor's batch strides + (logical) strides
  auto batch_strides = physical_tensor.strides().slice(0, num_batch_dims);
  SymDimVector physical_strides;
  physical_strides.reserve(num_batch_dims + strides.size());
  physical_strides.insert(
      physical_strides.end(), batch_strides.begin(), batch_strides.end());
  physical_strides.insert(
      physical_strides.end(), strides.begin(), strides.end());

  // If zi = xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
  // is valid for all i, then it turns out that
  // xs.as_strided(physical_sizes, physical_strides, offset) always succeeds
  // and creates a tensor y such that each y[i] references the same memory
  // locations as zi. See NOTE: [When will the as_strided batching rule fail?]
  auto result = physical_view.tensor().as_strided_symint(
      physical_sizes, physical_strides, std::move(storage_offset));
  return physical_view.getPhysicalToLogicalMap().apply(result);
}

// NOTE: [When will the as_strided batching rule fail?]
// If zi = xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
// is valid for all i, then it turns out that
// xs.as_strided(physical_sizes, physical_strides, offset) always succeeds and
// creates a tensor y such that each y[i] refers to the same memory as zi.
//
// Let's say we have xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset()).
// Furthermore, let's say that as a part of being "valid" this as_strided call
// does not return a result that can index memory not indexable by xs[i].
//
// WLOG, assume that there's only one batch dim and it is at the front of the
// `xs` tensor. Let B be the batch size and S be the stride of the batch dim.
// - If the batch dim isn't at the front of the tensor, then we can just move it
// to the front with movedim/permute. This is always valid because it just swaps
// some strides around.
// - This proof also works for tensors with multiple batch dims. We just have to
// do a little accounting:
//   - instead of [B], we'd have [B0, B1, ..., Bk].
//   - instead of [S], we'd have [S0, S1, ..., Sk].
//   - instead of i, we'd have a list of indices [I0, I1, ..., Ik]
//   - instead of S * I, we'd have \sum_{i=0}^k S_i * I_i
//
// [Equation 1]
// xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset()) has:
// - sizes: sizes
// - strides: strides
// - offset: offset + S * i
//
// x.as_strided itself checks that:
// - (sizes, strides, offset) are in bounds for `x`'s storage.
// - strides are positive
// - offset is positive
//
// Claim 1: if xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
// is valid, then
// ([B] + sizes, [S] + strides, offset + xs.offset()) are in bounds for `xs`'s storage.
//
// If we have the claim, then xs.as_strided([B] + sizes, [S] + strides, offset)
// won't error out. So all we need to check is that the memory locations are
// what we expected. See [Hand-wavy proof of Claim 1] for proof (it's not very important)
//
// xs.as_strided(physical_sizes, physical_strides, offset) is equivalent to
// xs.as_strided([B] + sizes, [S] + strides, offset)
//
// xs.as_strided([B] + sizes, [S] + strides, offset) has:
// - sizes: [B] + sizes
// - strides: [S] + strides
// - offset: offset
//
// xs.as_strided([B] + sizes, [S] + strides, offset)[i] has:
// - sizes: sizes
// - strides: strides
// - offset: offset + S * i
// These memory locations are exactly the same as what we got for [Equation 1],
// so the xs.as_strided([B] + sizes, [S] + strides, offset) is valid.
//
// [Hand-wavy proof of Claim 1]
// Part of our definition of being valid is that xs[i].as_strided(...)
// must return a tensor that only uses memory indexable by xs[i].
// This means that (sizes, strides, offset + xs[i].offset() - xs.offset()) satisfies:
//    offset + xs[i].offset() - xs.offset() + 1 + \sum_j (sizes[j] - 1) * strides[j]
//    <= xs[i].offset() + 1 + \sum_j (xs[i].size(j) - 1) * xs[i].stride(j)
// (the largest-index memory location of xs[i].as_strided(...) must be \leq
// the largest-index memory location of xs[i])
//
// Fiddling that inequality gives us:
//    offset - xs.offset() + 1 + \sum_j (sizes[j] - 1) * strides[j]
//    <= 1 + \sum_j (xs[i].size(j) - 1) * xs[i].stride(j)
//
//    offset - xs.offset() + 1 + (B-1)*S + \sum_j (sizes[j] - 1) * strides[j]
//    <= 1 + (B-1)*S + \sum_j (xs[i].size(j) - 1) * xs[i].stride(j)
//
//    offset - xs.offset() + 1 + (B-1)*S + \sum_j (sizes[j] - 1) * strides[j]
//    <= 1 + \sum_j (xs.size(j) - 1) * xs.stride(j)
//
//    offset + 1 + (B-1)*S + \sum_j (sizes[j] - 1) * strides[j]
//    <= xs.offset() + 1 + \sum_j (xs.size(j) - 1) * xs.stride(j)
// (the largest-index memory location of xs.as_strided(size, stride, offset)
// is \leq than the largest-index memory location of xs)
// Under the assumptions we've made, the lower bound (lowest indexed memory)
// is trivially within the storage.
//
// Therefore ([B] + sizes, [S] + strides, offset) are in bounds for
// `xs`'s storage.

template <typename F, F Func, typename... ExtraArgs>
Tensor unwrap_and_call(const Tensor& input, ExtraArgs... args) {
  if (!participatesInCurrentLevel(input)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return Func(input, args...);
  }
  // guard against the user passing in a batch of scalar tensors with batch
  auto* input_batched = unsafeGetBatchedImpl(input);
  auto output_physical = Func(input_batched->value(), args...);
  return makeBatched(output_physical, input_batched->bdim(), input_batched->level());
}

template <typename F, F Func, typename... ExtraArgs>
Tensor unwrap_and_call_method(const Tensor& input, ExtraArgs... extra_args) {
  if (!participatesInCurrentLevel(input)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return (input.*Func)(extra_args...);
  }
  auto* input_batched = unsafeGetBatchedImpl(input);
  auto output_physical = (input_batched->value().*Func)(extra_args...);
  return makeBatched(output_physical, input_batched->bdim(), input_batched->level());
}

Tensor cat_batching_rule(const ITensorListRef& tensors, int64_t dim) {
  if (!participatesInCurrentLevel(tensors)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::cat(tensors, dim);
  }

  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);

  // NB: Probably bad for perf that we're allocating std::vectors for each level, but
  // what can you do.
  auto materialized = tensors.materialize();
  dim = at::legacy_cat_wrap_dim(dim, materialized);

  // Strategy:
  // we're going to unwrap tensors, move their batch dims to the front,
  // and put them into `tensors_to_cat`. Tensors that don't have a batch dim
  // will get one forced onto them.
  //
  // Then, we'll do at::cat(tensors_to_cat, ...).
  //
  // There's a special case where at::cat ignores tensors that have logical shape
  // [0]. If we see a Tensor that has logical shape [0] (but physical shape [B, 0]),
  // we'll just slice the tensor to get a Tensor of shape [0] to pass to at::cat.
  std::vector<Tensor> tensors_to_cat;
  tensors_to_cat.reserve(tensors.size());
  c10::optional<int64_t> bdim_size = c10::nullopt;

  // find the bdim size. Might not exist if all BatchedTensors should be skipped
  // by cat's special case.
  for (const auto& tensor : tensors) {
    if (!participatesInCurrentLevel(tensor)) {
      continue;
    }
    if (at::native::cat_should_skip_tensor(tensor)) {
      continue;
    }
    const auto* batched = unsafeGetBatchedImpl(tensor);
    bdim_size = batched->value().size(batched->bdim());
    break;
  }

  // unwrap batchedtensors; expand out bdims
  for (const auto& tensor : tensors) {
    if (!participatesInCurrentLevel(tensor)) {
      if (at::native::cat_should_skip_tensor(tensor) || !bdim_size.has_value()) {
        tensors_to_cat.emplace_back(tensor);
        continue;
      }
      tensors_to_cat.emplace_back(ensure_has_bdim(tensor, /*has_bdim*/false, *bdim_size));
      continue;
    }
    const auto* batched = unsafeGetBatchedImpl(tensor);
    if (at::native::cat_should_skip_tensor(tensor)) {
      // Special case: slice the tensor to get something of shape [0] to pass to cat
      // We slice instead of allocate a new tensor to propagate requires_gradness...
      tensors_to_cat.emplace_back(batched->value().select(/*dim=*/batched->bdim(), /*index=*/0));
      continue;
    }
    tensors_to_cat.emplace_back(moveBatchDimToFront(batched->value(), batched->bdim()));
  }

  auto new_dim = bdim_size.has_value() ? dim + 1 : dim;
  c10::optional<int64_t> new_bdim = bdim_size.has_value() ? c10::make_optional((int64_t)0) : nullopt;
  auto result = at::cat(tensors_to_cat, new_dim);
  return makeBatched(result, new_bdim, get_current_level());
}

Tensor block_diag_batching_rule(TensorList tensors) {
  if (!participatesInCurrentLevel(tensors)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::block_diag(tensors);
  }
  auto physical_views = MultiBatchVmapTransform::logicalToPhysical(tensors);
  auto physical_tensors = fmap(
      physical_views, [](const VmapPhysicalView& view) -> Tensor { return view.tensor(); });
  TORCH_INTERNAL_ASSERT(
      !tensors.empty(), "The dispatcher should not have dispatched here otherwise.");
  // Implementing this as a dummy for loop for now, since I'm not sure how to do it any better.
  // I'm probably not accounting for potentially multiple batched dimensions?
  auto bdim = physical_tensors[0].size(0);
  std::vector<Tensor> batched_outputs;
  batched_outputs.reserve(bdim);
  for (const auto& i : c10::irange(bdim)) {
    std::vector<Tensor> inputs_for_batch;
    inputs_for_batch.reserve(physical_tensors.size());
    for (const auto& t : physical_tensors) {
      inputs_for_batch.push_back(t[i]);
    }
    auto out_for_batch = at::block_diag(inputs_for_batch);
    batched_outputs.push_back(out_for_batch.unsqueeze(0));
  }
  auto result = at::cat(batched_outputs);
  return physical_views[0].getPhysicalToLogicalMap().apply(result);
}

Tensor stack_batching_rule(TensorList tensors, int64_t dim) {
  if (!participatesInCurrentLevel(tensors)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::stack(tensors, dim);
  }
  auto physical_views = MultiBatchVmapTransform::logicalToPhysical(tensors);
  auto physical_tensors = fmap(
      physical_views, [](const VmapPhysicalView& view) -> Tensor { return view.tensor(); });
  TORCH_INTERNAL_ASSERT(
      !tensors.empty(), "The dispatcher should not have dispatched here otherwise.");
  // NB: stack wraps the dimensionality to (logical dim + 1), so we have to
  // manually handle that here.
  auto dim_physical =
      physical_views[0].numBatchDims() + maybe_wrap_dim(dim, /*logical*/tensors[0].dim() + 1);
  auto result = at::stack(physical_tensors, dim_physical);
  return physical_views[0].getPhysicalToLogicalMap().apply(result);
}

Tensor new_empty_strided_batching_rule(
    const Tensor& self,
    SymIntArrayRef sym_size,
    SymIntArrayRef sym_stride,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory) {

  auto size = C10_AS_INTARRAYREF_SLOW(sym_size);
  auto stride = C10_AS_INTARRAYREF_SLOW(sym_stride);
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return self.new_empty_strided(
        size, stride, dtype, layout, device, pin_memory);
  }

  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
  auto physical_size = physical_view.getPhysicalShape(size);

  // Let [B0, B1, B2] be the shape of the batch dims. We're going to create
  // the batch dimensions at the front of the tensor (in memory layout),
  // irrespective of whether or not they are actually at the front (in memory layout)
  // in the original `self` tensor. This is because when a user calls
  // `new_empty_strided` in general, the `strides` they provide are for a new
  // tensor and have no relation to the strides of the original tensor.
  //
  // So, the physical shape of the result should be ([B0, B1, B2] + size),
  // but what about the physical strides?
  //
  // We're actually free to pick whatever stride we want:
  // e.g., for size=[5, 3], stride=[0, 1], we could decide to
  // use
  // - physical size: [B0, B1, B2, 5, 3]
  // - physical stride: [9999*B1*B2, 9999*B2, 9999, 0, 1]
  //
  // Let's select some reasonable strides such that:
  // - The batch dims are "contiguous" with respect to each other
  // - if empty_strided(size, stride) would have created a contiguous Tensor,
  // then this new physical Tensor (with batch dims) is also contiguous
  //
  // Let S be the size of the storage if one were to construct a tensor
  // with `size` and `stride` via empty_strided(size, stride).
  // Then the physical sizes/strides should be:
  // - physical size: [B0, B1, B2, 5, 3]
  // - physical stride: [B1 * B2 * S, B2 * S, S, 0, 1]
  auto batch_shape = IntArrayRef(
      physical_view.tensor().sizes().begin(), physical_view.numBatchDims());

  // physical_strides = [B1 * B2 * S, B2 * S, S]
  auto physical_strides = at::detail::defaultStrides(batch_shape);
  TORCH_CHECK(size.size() == stride.size(),
        "new_empty_strided(sizes, strides): dimensionality of sizes (",
        size.size(), ") must match dimensionality of strides (",
        stride.size(), ")");
  auto storage_size = native::storage_size_for(size, stride);
  for (auto& physical_stride : physical_strides) {
    physical_stride *= storage_size;
  }

  // physical_strides = [B1 * B2 * S, B2 * S, S] + strides
  physical_strides.insert(physical_strides.end(), stride.begin(), stride.end());

  auto result = physical_view.tensor().new_empty_strided(
      physical_size, physical_strides, dtype, layout, device, pin_memory);
  return physical_view.getPhysicalToLogicalMap().apply(result);
}

Tensor nested_cat_batching_rule(const ITensorListRef& tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "cat() not supported on empty tensor list");

  std::vector<std::vector<Tensor>> unbound;
  for (auto tensor_iter = tensors.begin(); tensor_iter != tensors.end(); ++tensor_iter) {
    auto* maybe_batched_impl = maybeGetBatchedImpl(*tensor_iter);
    TORCH_CHECK(maybe_batched_impl, "Tried to run batching rule for cat() on a non-batched tensor");
    auto nt = maybe_batched_impl->value();
    TORCH_CHECK(nt.is_nested(), "Tried to run batching rule for cat() on a non-nested tensor");
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::BatchedNestedTensor);
    auto this_unbound = nt.unbind();
    if (unbound.size() > 0) {
      TORCH_INTERNAL_ASSERT(unbound.front().size() == this_unbound.size(),
          "cat() not supported for differently-sized nested arguments");
    }
    unbound.push_back(this_unbound);
  }

  // Do a cat for each set of zipped unbound components
  const auto num_components = unbound.front().size();
  std::vector<Tensor> outputs;
  for (auto i : c10::irange(num_components)) {
    std::vector<Tensor> arg_list;
    for (auto j : c10::irange(unbound.size())) {
      arg_list.push_back(unbound[j][i]);
    }
    outputs.push_back(at::cat(arg_list, dim));
  }

  // NB: NTs only support batching over dim 0
  auto out_nt = at::_nested_tensor_from_tensor_list(outputs);
  return makeBatched(out_nt, 0, get_current_level());
}

}

TORCH_LIBRARY_IMPL(_, FuncTorchBatched, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchedTensorForLoopFallback>());
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // still legacy b/c teturns multiple tensors
  m.impl("tensor_split.sections", tensor_split_sections_batching_rule);
  m.impl("tensor_split.indices", tensor_split_indices_batching_rule);
  m.impl("split.Tensor", split_batching_rule);
  m.impl("split_with_sizes", split_with_sizes_batching_rule);
  m.impl("unbind.int", unbind_batching_rule);
  m.impl("cat", cat_batching_rule);
  m.impl("block_diag", block_diag_batching_rule);
  m.impl("stack", stack_batching_rule);

  // still legacy b/c needs special inplace rules
  m.impl("squeeze_", squeeze__batching_rule);
  m.impl("squeeze_.dim", squeeze_dim__batching_rule);
  m.impl("squeeze_.dims", squeeze_dims__batching_rule);
  m.impl("unsqueeze_", unsqueeze__batching_rule);
  m.impl("transpose_", transpose__batching_rule);

  // still legacy because these are ridiculously complicated
  m.impl("as_strided", as_strided_batching_rule);
  m.impl("new_empty_strided", new_empty_strided_batching_rule);

}

TORCH_LIBRARY_IMPL(_, BatchedNestedTensor, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchedNestedTensorForLoopFallback>());
}

// TODO: Move this somewhere better?
TORCH_LIBRARY_IMPL(aten, BatchedNestedTensor, m) {
  m.impl("cat", nested_cat_batching_rule);
}
} // namespace functorch
} // namespace at
