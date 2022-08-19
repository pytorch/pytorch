// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <functorch/csrc/BatchedTensorImpl.h>

#include <ATen/WrapDimUtils.h>
#include <c10/util/Exception.h>

#include <functorch/csrc/Constants.h>
#include <c10/util/irange.h>

namespace at {
namespace functorch {

BatchedTensorImpl::BatchedTensorImpl(Tensor value, int64_t bdim, int64_t level)
  : TensorImpl(
      c10::DispatchKeySet(kBatchedKey),
      value.dtype(),
      value.device()
    )
  , value_(std::move(value))
  , level_(level)
  , bdim_(bdim)
{
  // TODO: I don't think this ctor gets used.
  TORCH_INTERNAL_ASSERT(false);
  TORCH_INTERNAL_ASSERT(value_.defined());
  set_storage_access_should_throw();
  set_sizes_strides_policy(SizesStridesPolicy::CustomStrides);
  checkInvariants();

  const auto public_dims = value_.dim() - 1;
  const auto value_sizes = value_.sizes();
  const auto value_strides = value_.strides();
  sizes_and_strides_.resize(public_dims);
  for (const auto dim : c10::irange(0, public_dims)) {
    auto actual_dim = actualDim(dim, /*wrap_dim=*/false);
    sizes_and_strides_.size_at_unchecked(dim) = value_sizes.at(actual_dim);
    sizes_and_strides_.stride_at_unchecked(dim) = value_strides.at(actual_dim);
  }
  storage_offset_= value_.storage_offset();
  refresh_numel();
  refresh_contiguous();
}

BatchedTensorImpl::BatchedTensorImpl(DispatchKeySet key_set, Tensor value, int64_t bdim, int64_t level)
  : TensorImpl(
      key_set.add(kBatchedKey),
      value.dtype(),
      value.device()
    )
  , value_(std::move(value))
  , level_(level)
  , bdim_(bdim)
{
  TORCH_INTERNAL_ASSERT(value_.defined());
  set_storage_access_should_throw();
  set_sizes_strides_policy(SizesStridesPolicy::CustomStrides);
  checkInvariants();
  refreshTensorMetadata();
}

void BatchedTensorImpl::refreshTensorMetadata() {
  const auto public_dims = value_.dim() - 1;
  const auto value_sizes = value_.sizes();
  const auto value_strides = value_.strides();
  sizes_and_strides_.resize(public_dims);
  for (const auto dim : c10::irange(0, public_dims)) {
    auto actual_dim = actualDim(dim, /*wrap_dim=*/false);
    sizes_and_strides_.size_at_unchecked(dim) = value_sizes.at(actual_dim);
    sizes_and_strides_.stride_at_unchecked(dim) = value_strides.at(actual_dim);
  }
  storage_offset_= value_.storage_offset();
  refresh_numel();
  refresh_contiguous();
}

int64_t BatchedTensorImpl::actualDim(int64_t dim, bool wrap_dim) const {
  if (wrap_dim) {
    const auto ndim = sizes_and_strides_.size();
    dim = maybe_wrap_dim(dim, ndim);
  }
  auto is_bdim = createBatchDimBitset(bdim_);

  // TODO(vfdev): As BatchedTensorImpl is refactored and has only one dim.
  // Below code may be simplified.

  // Example: assume dim = 3, and is_bdim = 10010011000...
  // The 1's are batch dims and 0's are normal dims of the underlying value_ Tensor.
  // actualDim gives us the index of `dim` in the `value_` Tensor, which is equivalent
  // to asking "where does the 3rd (0-indexed) zero occur in the bitset?".
  // The answer to that is index 5.
  //
  // TODO(rzou): the PDEP instruction does exactly this
  // (https://stackoverflow.com/questions/7669057/find-nth-set-bit-in-an-int)
  // but it might require newer (>= ~2015) CPUs. We should clean this up
  // if/when we have dropped support for older CPUs.
  int64_t non_bdim_count = 0;
  for (int64_t actual_dim = 0; actual_dim < kVmapMaxTensorDims; actual_dim++) {
    if (is_bdim[actual_dim]) {
      continue;
    }
    if (non_bdim_count == dim) {
      return actual_dim;
    }
    non_bdim_count++;
  }
  // If we hit this assert, then that means
  // `non_bdim_count` + #num_bdims > kVmapMaxTensorDims. We restrict the number
  // of dims a BatchedTensorImpl can have to kVmapMaxTensorDims so this should
  // never be hit.
  TORCH_INTERNAL_ASSERT(false);
}

void BatchedTensorImpl::checkInvariants() const {
  TORCH_INTERNAL_ASSERT(level_ > -1);
}

// The following are publically exposed as methods of Tensor

IntArrayRef BatchedTensorImpl::strides_custom() const {
  return strides_default();
}

// TODO: implement proper contiguity on batched tensor, then put
// sizes_strides_policy back to Default
bool BatchedTensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  TORCH_CHECK(memory_format == MemoryFormat::Contiguous,
      "NYI: querying is_contiguous inside of vmap for memory_format ",
      "other than torch.contiguous_format");
  return is_contiguous_;
}

// The following are some internal inherited methods that we do not support.
// They should never get called.
void BatchedTensorImpl::set_size(int64_t dim, int64_t new_size) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_size for BatchedTensorImpl");
}
void BatchedTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_stride for BatchedTensorImpl");
}
void BatchedTensorImpl::set_storage_offset(int64_t storage_offset) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_storage_offset for BatchedTensorImpl");
}
#ifdef DEBUG
bool BatchedTensorImpl::has_storage() const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "BatchedTensorImpl assumes that storage_ is never set");
  return false;
}
#endif

const char* BatchedTensorImpl::tensorimpl_type_name() const {
  return "BatchedTensorImpl";
}

Tensor makeBatched(const Tensor& tensor, int64_t bdim, int64_t level) {
  DispatchKeySet key_set = getKeysToPropagateToWrapper(tensor);
  auto* batched = maybeGetBatchedImpl(tensor);
  if (batched) {
    auto batched_level = batched->level();
    TORCH_INTERNAL_ASSERT(level > batched_level, " batched_level: ", batched_level, " level: ", level);
  }
  return at::detail::make_tensor<BatchedTensorImpl>(key_set, tensor, bdim, level);
}

Tensor addBatchDim(const Tensor& tensor, int64_t dim, int64_t level) {
  return makeBatched(tensor, dim, level);
}

}
} // namespace at
