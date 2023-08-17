// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <ATen/functorch/BatchedTensorImpl.h>

#include <ATen/WrapDimUtils.h>
#include <c10/util/Exception.h>

#include <c10/util/irange.h>

namespace at {
namespace functorch {

BatchedTensorImpl::BatchedTensorImpl(DispatchKeySet key_set, Tensor value, int64_t bdim, int64_t level)
  : TensorImpl(
      key_set.add(
          value.is_nested() ? DispatchKeySet({
              DispatchKey::NestedTensor,
              DispatchKey::BatchedNestedTensor
          }) : DispatchKeySet({DispatchKey::FuncTorchBatched})),
      value.dtype(),
      value.device()
    )
  , value_(std::move(value))
  , level_(level)
  , bdim_(bdim)
{
  TORCH_INTERNAL_ASSERT(value_.defined());
  set_storage_access_should_throw();
  set_custom_sizes_strides(
      value_.is_nested() ? SizesStridesPolicy::CustomSizes : SizesStridesPolicy::CustomStrides);
  checkInvariants();
  refreshTensorMetadata();
}

void BatchedTensorImpl::refreshTensorMetadata() {
  const auto public_dims = value_.dim() - 1;
  sizes_and_strides_.resize(public_dims);

  if (!value_.is_nested()) {
    const auto value_sizes = value_.sizes();
    const auto value_strides = value_.strides();
    for (const auto dim : c10::irange(0, public_dims)) {
      auto actual_dim = actualDim(dim, /*wrap_dim=*/false);
      sizes_and_strides_.size_at_unchecked(dim) = value_sizes.at(actual_dim);
      sizes_and_strides_.stride_at_unchecked(dim) = value_strides.at(actual_dim);
    }
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
  if (bdim_ <= dim) {
    return dim + 1;
  } else {
    return dim;
  }
}

void BatchedTensorImpl::checkInvariants() const {
  TORCH_INTERNAL_ASSERT(level_ > -1);
}

int64_t BatchedTensorImpl::size_custom(int64_t d) const {
  if (!value_.is_nested()) {
    return size(d);
  }
  // TODO: Error messages will mention the actualDim, which could be confusing; fix this
  auto actual_dim = actualDim(d, /*wrap_dim=*/ true);
  return value_.size(actual_dim);
}

c10::SymInt BatchedTensorImpl::sym_size_custom(int64_t d) const {
  if (!value_.is_nested()) {
    return sym_size(d);
  }
  // TODO: Error messages will mention the actualDim, which could be confusing; fix this
  auto actual_dim = actualDim(d, /*wrap_dim=*/ true);
  return value_.sym_size(actual_dim);
}

IntArrayRef BatchedTensorImpl::sizes_custom() const {
  TORCH_CHECK(!value_.is_nested(), "sizes() is not supported for batched nested tensors");
  return sizes_default();
}

SymIntArrayRef BatchedTensorImpl::sym_sizes_custom() const {
  TORCH_CHECK(!value_.is_nested(), "sizes() is not supported for batched nested tensors");
  return sym_sizes_default();
}

// The following are publically exposed as methods of Tensor

IntArrayRef BatchedTensorImpl::strides_custom() const {
  return strides_default();
}

SymIntArrayRef BatchedTensorImpl::sym_strides_custom() const {
  return sym_strides_default();
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

c10::intrusive_ptr<TensorImpl> BatchedTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  TORCH_CHECK(false, "accessing `data` under vmap transform is not allowed");
  return nullptr;
}

c10::intrusive_ptr<TensorImpl> BatchedTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  TORCH_CHECK(false, "accessing `data` under vmap transform is not allowed");
  return nullptr;
}

void BatchedTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_CHECK(false, "mutating directly with `.data` under vmap transform is not allowed.");
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
