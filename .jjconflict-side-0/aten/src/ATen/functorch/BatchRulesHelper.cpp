// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/WrapDimUtils.h>

namespace at::functorch {

Tensor moveBatchDimToFront(Tensor tensor, std::optional<int64_t> maybe_batch_dim) {
  if (!maybe_batch_dim.has_value()) {
    return tensor;
  }
  if (maybe_batch_dim.value() == 0) {
    return tensor;
  }
  return tensor.movedim(maybe_batch_dim.value(), 0);
}

int64_t rankWithoutBatchDim(const Tensor& tensor, std::optional<int64_t> maybe_batch_dim) {
  int64_t result = tensor.dim();
  if (maybe_batch_dim.has_value()) {
    result -= 1;
  }
  return result;
}

int64_t numelWithoutBatchDim(const Tensor& tensor, std::optional<int64_t> maybe_batch_dim) {
  if (!maybe_batch_dim) {
    return tensor.numel();
  }
  return tensor.numel() / tensor.size(*maybe_batch_dim);
}

std::optional<int64_t> valIfNonempty(std::optional<int64_t> maybe_empty, int64_t new_val) {
  if (maybe_empty.has_value()) {
    return new_val;
  }
  return std::nullopt;
}

int64_t getPhysicalDim(const Tensor& tensor, bool has_batch_dim, int64_t logical_dim) {
  // NB: assumes the batch dim is at the front of the tensor
  std::optional<int64_t> bdim = has_batch_dim ? std::optional<int64_t>(0) : std::nullopt;
  auto rank = rankWithoutBatchDim(tensor, bdim);
  auto wrapped_dim = maybe_wrap_dim(logical_dim, rank);
  if (has_batch_dim) {
    return wrapped_dim + 1;
  }
  return wrapped_dim;
}

VmapDimVector getPhysicalDims(const Tensor& tensor, bool has_batch_dim, IntArrayRef logical_dims) {
  // NB: assumes the batch dim is at the front of the tensor
  std::optional<int64_t> bdim = has_batch_dim ? std::optional<int64_t>(0) : std::nullopt;
  auto rank = rankWithoutBatchDim(tensor, bdim);
  VmapDimVector result;
  result.reserve(logical_dims.size());
  for (auto d : logical_dims){
    if (has_batch_dim) {
      result.push_back(maybe_wrap_dim(d, rank)+1);
    } else {
      result.push_back(maybe_wrap_dim(d, rank));
    }
  }
  return result;
}

Tensor maybePadToLogicalRank(const Tensor& tensor, std::optional<int64_t> has_bdim, int64_t logical_rank) {
  if (!has_bdim) {
    return tensor;
  }
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, has_bdim);
  if (tensor_logical_rank >= logical_rank) {
    return tensor;
  }
  VmapSymDimVector new_sizes(tensor.sym_sizes().begin(), tensor.sym_sizes().end());
  for (int64_t i = 0; i < logical_rank - tensor_logical_rank; i++) {
    new_sizes.insert(new_sizes.begin() + 1, 1);
  }
  return tensor.view_symint(SymIntArrayRef{new_sizes.begin(), new_sizes.end()});
}

void check_randomness(RandomnessType randomness, bool any_tensor_batched) {
  TORCH_CHECK(
    randomness != RandomnessType::Error,
    "vmap: called random operation while in randomness error mode. Please either use the "
    "'same' or 'different' randomness flags on vmap or perform the randomness operation out of vmap"
  );

  TORCH_CHECK(
    !(randomness == RandomnessType::Same && any_tensor_batched),
    "Vmap does not currently support same randomness with a batched tensor input. ",
    "Please file an issue with functorch"
  )
}

void check_randomness(RandomnessType randomness) {
  check_randomness(randomness, false); // for ops that don't take in any tensors, don't hit same error
}

Tensor reshape_dim_into(int64_t src, int64_t dst, const Tensor& x) {
  auto x_dim = x.dim();
  src = maybe_wrap_dim(src, x_dim);
  dst = maybe_wrap_dim(dst, x_dim - 1); // Returned Tensor has one fewer dim
  VmapDimVector new_shape(x.sizes().begin(), x.sizes().end());
  new_shape.erase(new_shape.begin() + src);
  new_shape[dst] *= x.sizes()[src];
  return at::reshape(x.movedim(src, dst), new_shape);
}

Tensor reshape_dim_outof(int64_t src, int64_t size1, const Tensor& x) {
  src = maybe_wrap_dim(src, x.dim());
  VmapDimVector shape(x.sizes().begin(), x.sizes().end());
  if (shape[src] != 0) {
    // NOTE: 0 % 0 leads to FPE
    TORCH_INTERNAL_ASSERT(shape[src] % size1 == 0);
  }
  // split any size out of `0`-sized dim
  int64_t size2 = 0;
  if (shape[src] != 0) {
    size2 = shape[src] / size1;
  }
  shape[src] = size1;
  shape.insert(shape.begin() + src + 1, size2);
  return at::reshape(x, shape);
}

Tensor reshape_dim_outof_symint(int64_t src, const c10::SymInt& size1, const Tensor& x) {
  src = maybe_wrap_dim(src, x.dim());
  c10::SymDimVector shape(x.sym_sizes().begin(), x.sym_sizes().end());
  if (shape[src] != 0) {
    // NOTE: 0 % 0 leads to FPE
    TORCH_INTERNAL_ASSERT(shape[src] % size1 == 0);
  }
  c10::SymInt size2;
  // split any size out of `0`-sized dim
  if (shape[src] == 0) {
    size2 = 0;
  } else {
    size2 = shape[src] / size1;
  }
  shape[src] = size1;
  shape.insert(shape.begin() + src + 1, size2);
  return at::reshape_symint(x, shape);
}

void vmapIncompatibleInplaceError(const char* schema_name) {
  TORCH_CHECK(false,
    "vmap: ", schema_name, "(self, *extra_args) is not possible because ",
    "there exists a Tensor `other` in extra_args that has more elements ",
    "than `self`. This happened due to `other` being vmapped over but ",
    "`self` not being vmapped over in a vmap. ",
    "Please try to use out-of-place operators instead of ", schema_name, ". ",
    "If said operator is being called inside the PyTorch framework, ",
    "please file a bug report instead.");
}

static void handleScalarTypePromotion(Tensor& logical_scalar_tensor, Tensor& second) {
  auto result_type = at::native::result_type(logical_scalar_tensor[0], second);
  if (logical_scalar_tensor.scalar_type() != result_type) {
    logical_scalar_tensor = logical_scalar_tensor.to(result_type);
  }
  if (second.scalar_type() != result_type) {
    second = second.to(result_type);
  }
}

std::tuple<Tensor, Tensor> _binary_pointwise_helper(
    const Tensor& tensor, std::optional<int64_t> tensor_batch_dim,
    const Tensor& other, std::optional<int64_t> other_batch_dim,
    bool do_type_promotion) {
  // compute max logical rank
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, tensor_batch_dim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_batch_dim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  auto tensor_ = moveBatchDimToFront(tensor, tensor_batch_dim);
  auto other_ = moveBatchDimToFront(other, other_batch_dim);

  // In the (0D, ND) case, type promotion semantics are different :/
  if (do_type_promotion) {
    auto tensor_is_logical_scalar = (tensor_logical_rank == 0 && tensor_batch_dim.has_value());
    auto other_is_logical_scalar = (other_logical_rank == 0 && other_batch_dim.has_value());
    if (tensor_is_logical_scalar && !other_is_logical_scalar) {
      handleScalarTypePromotion(tensor_, other_);
    }
    if (other_is_logical_scalar && !tensor_is_logical_scalar) {
      handleScalarTypePromotion(other_, tensor_);
    }
  }

  // If the dimensions aren't aligned, we need to line them up.
  // Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // Note that only tensors that have a batch dim need to be modified.
  // Tensor[B, 2, 3, 5] + Tensor[5] -> no changes needed
  tensor_ = maybePadToLogicalRank(tensor_, tensor_batch_dim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_batch_dim, max_logical_rank);

  return std::make_tuple(std::move(tensor_), std::move(other_));
}

} // namespace at::functorch
