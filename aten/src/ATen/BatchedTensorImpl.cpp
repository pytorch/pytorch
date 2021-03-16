#include <ATen/BatchedTensorImpl.h>

#include <ATen/WrapDimUtils.h>
#include <c10/util/Exception.h>

namespace at {

/**
 * Return a TensorImpl that is a shallow-copy of this TensorImpl.
 *
 * For usage of `version_counter` and `allow_tensor_metadata_change`,
 * see NOTE [ TensorImpl Shallow-Copying ].
 */
c10::intrusive_ptr<TensorImpl> BatchedTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  Tensor v = value();
  BatchDims b(bdims().begin(), bdims().end());
  auto impl = c10::make_intrusive<BatchedTensorImpl>(v, b);
  copy_tensor_metadata(
    /*src_impl=*/this,
    /*dest_impl=*/impl.get(),
    /*version_counter=*/version_counter,
    /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->refresh_numel();
  return impl;
}

/**
 * Return a TensorImpl that is a shallow-copy of this TensorImpl.
 *
 * For usage of `version_counter` and `allow_tensor_metadata_change`,
 * see NOTE [ TensorImpl Shallow-Copying ].
 */
c10::intrusive_ptr<TensorImpl> BatchedTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  Tensor v = value();
  BatchDims b(bdims().begin(), bdims().end());
  auto impl = c10::make_intrusive<BatchedTensorImpl>(v, b);
  copy_tensor_metadata(
    /*src_impl=*/this,
    /*dest_impl=*/impl.get(),
    /*version_counter=*/std::move(version_counter),
    /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->refresh_numel();
  return impl;
}

/**
 * Shallow-copies data from another TensorImpl into this TensorImpl.
 *
 * For why this function doesn't check this TensorImpl's `allow_tensor_metadata_change_`,
 * see NOTE [ TensorImpl Shallow-Copying ].
 */
void BatchedTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_INTERNAL_ASSERT(false);
}

BatchedTensorImpl::BatchedTensorImpl(Tensor value, BatchDims bdims)
  : TensorImpl(
      c10::DispatchKeySet(DispatchKey::Batched),
      value.dtype(),
      value.device()
    )
  , value_(std::move(value))
  , bdims_(std::move(bdims))
{
  TORCH_INTERNAL_ASSERT(value_.defined());
  checkInvariants();

  TORCH_INTERNAL_ASSERT(bdims_.size() == 1);

  const auto public_dims = value_.dim() - bdims_.size();
  const auto value_sizes = value_.sizes();
  const auto value_strides = value_.strides();
  sizes_and_strides_.resize(public_dims);
  for (int64_t dim = 0; dim < public_dims; dim++) {
    auto actual_dim = actualDim(dim, /*wrap_dim=*/false);
    sizes_and_strides_.size_at_unchecked(dim) = value_sizes.at(actual_dim);
    sizes_and_strides_.stride_at_unchecked(dim) = value_strides.at(actual_dim);
  }
  refresh_numel();
  refresh_contiguous();
}

BatchedTensorImpl::BatchedTensorImpl(DispatchKeySet key_set, Tensor value, BatchDims bdims)
  : TensorImpl(
      key_set.add(DispatchKey::Batched),
      value.dtype(),
      value.device()
    )
  , value_(std::move(value))
  , bdims_(std::move(bdims))
{
  TORCH_INTERNAL_ASSERT(value_.defined());
  checkInvariants();

  TORCH_INTERNAL_ASSERT(bdims_.size() == 1);

  const auto public_dims = value_.dim() - bdims_.size();
  const auto value_sizes = value_.sizes();
  const auto value_strides = value_.strides();
  sizes_and_strides_.resize(public_dims);
  for (int64_t dim = 0; dim < public_dims; dim++) {
    auto actual_dim = actualDim(dim, /*wrap_dim=*/false);
    sizes_and_strides_.size_at_unchecked(dim) = value_sizes.at(actual_dim);
    sizes_and_strides_.stride_at_unchecked(dim) = value_strides.at(actual_dim);
  }
  refresh_numel();
  refresh_contiguous();
}

int64_t BatchedTensorImpl::actualDim(int64_t dim, bool wrap_dim) const {
  if (wrap_dim) {
    const auto ndim = sizes_and_strides_.size();
    dim = maybe_wrap_dim(dim, ndim);
  }
  auto is_bdim = createBatchDimBitset(bdims_);

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
  int64_t prev_level = -1;
  for (const auto& bdim : bdims_) {
    TORCH_INTERNAL_ASSERT(bdim.level() > prev_level);
    prev_level = bdim.level();
  }
}

// The following are publically exposed as methods of Tensor
bool BatchedTensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
  TORCH_CHECK(memory_format == MemoryFormat::Contiguous,
      "NYI: querying is_contiguous inside of vmap for memory_format ",
      "other than torch.contiguous_format");
  return is_contiguous_;
}

const Storage& BatchedTensorImpl::storage() const {
  TORCH_CHECK(false, "Due to limitations, we cannot access the storage() of a tensor from inside of vmap.");
}
// int64_t BatchedTensorImpl::storage_offset() const {
//   TORCH_CHECK(false, "Due to limitations, we cannot access the storage_offset() of a tensor from inside of vmap.");
// }
// 
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
bool BatchedTensorImpl::has_storage() const {
  // TODO: what are the implications of this?
  return false;
  // TORCH_INTERNAL_ASSERT(false, "Can't query has_storage for BatchedTensorImpl");
}

Tensor makeBatched(const Tensor& tensor, BatchDims bdims) {
  TORCH_INTERNAL_ASSERT(bdims.size() == 1);

  // NB: temporarily disable 
  // TORCH_INTERNAL_ASSERT(!isBatchedTensor(tensor));
  // auto tensor_dim = tensor.dim();
  // TORCH_CHECK(
  //     tensor_dim <= kVmapMaxTensorDims,
  //     "vmap only supports tensors of dimensionality up to ", kVmapMaxTensorDims,
  //     "; got a tensor with dim ", tensor_dim);
  // TORCH_INTERNAL_ASSERT(
  //     std::all_of(bdims.begin(), bdims.end(),
  //         [](const BatchDim& bdim) { return bdim.level() < kVmapNumLevels; }),
  //     "We only support up to ", kVmapNumLevels, " nested vmaps");
  // propagate the CUDA key for the is_cuda check.
  // TODO: probably need to propagate other backend keys as well...
  DispatchKeySet key_set;
  if (tensor.is_cuda()) {
    key_set = key_set.add(DispatchKey::CUDA);
  }
  return at::detail::make_tensor<BatchedTensorImpl>(key_set, tensor, std::move(bdims));
}

Tensor addBatchDim(const Tensor& tensor, int64_t level, int64_t dim) {
  BatchDims new_bdims = { { level, dim } };
  TORCH_INTERNAL_ASSERT(new_bdims.size() == 1);
  return makeBatched(tensor, std::move(new_bdims));

  // const auto* batched = maybeGetBatchedImpl(tensor);
  // if (!batched) {
  //   BatchDims bdims;
  //   bdims.emplace_back(level, dim);
  //   return at::detail::make_tensor<BatchedTensorImpl>(tensor, std::move(bdims));
  // }
  // BatchDims new_bdims(batched->bdims().begin(), batched->bdims().end());
  // auto actual_bdim = batched->actualDim(dim, /*wrap_dim=*/true);
  // new_bdims.emplace_back(level, actual_bdim);
  // return makeBatched(batched->value(), std::move(new_bdims));
}

bool inplaceIsVmapCompatible(const Tensor& self, const Tensor& other) {
  const auto* other_batched = maybeGetBatchedImpl(other);
  if (!other_batched) {
    return true;
  }
  const auto* self_batched = maybeGetBatchedImpl(self);
  if (!self_batched) {
    // self is not batched but other is batched
    return false;
  }
  auto self_levels = createVmapLevelsBitset(self_batched->bdims());
  auto other_levels = createVmapLevelsBitset(other_batched->bdims());
  return self_levels == (self_levels | other_levels);
}

} // namespace at
