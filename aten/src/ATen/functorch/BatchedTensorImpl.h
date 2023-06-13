// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <bitset>
#include <utility>

#include <ATen/ArrayRef.h>
#include <ATen/SmallVector.h>
#include <ATen/Tensor.h>

namespace at {
namespace functorch {

using Tensor = at::Tensor;

// We assume this in a few other places in the codebase,
// but there isn't a centralized definition.
constexpr int64_t kVmapMaxTensorDims = 64;

// The valid vmap levels range from [0, 64). This effectively means that we
// support a maximum of 64 nested vmaps.
constexpr int64_t kVmapNumLevels = 64;

// Store this number of elements of BatchDims on the stack. Most people will
// probably use <= 5 nested vmaps, but adjust this number as necessary.
constexpr int64_t kBatchDimsStackSize = 5;

// A BatchedTensorImpl holds an underlying Tensor and a single batch dim
// NB: We use the term "BatchedTensor" to mean a Tensor that is backed with a
// BatchedTensorImpl.
//
// The batch dimensions are treated as being "private"; they are not user-visible.
// For example, in the following Tensor,
//    bt = BatchedTensorImpl(ones(2, 3, 5, 7), lvl=1, dim=0)
// dimension 0 is batch dimension.
//
// bt.sizes() returns (5, 7); bt.sum(0) performs a reduction over the (public)
// dim 0, which is equivalent to dim 3 in the underlying ones(2, 3, 5, 7) tensor.
struct TORCH_API BatchedTensorImpl : public c10::TensorImpl {
  explicit BatchedTensorImpl(at::DispatchKeySet key_set, Tensor value, int64_t dim, int64_t level);

  // Returns batch dimension of this tensor
  int64_t bdim() const { return bdim_; }

  // Returns batch dimension of this tensor
  int64_t level() const { return level_; }

  // BatchedTensorImpl wraps a Tensor
  const Tensor& value() const { return value_; }

  // Given a public dimension index, return the dimension index in the underlying
  // value() tensor.
  // For example, if we have
  //    bt = BatchedTensorImpl(ones(2, 3, 5, 7), lvl=1, dim=0)
  // bt.actualDim(0) -> 1
  // bt.actualDim(1) -> 2
  // bt.actualDim(2) -> 3
  // bt.actualDim(3) -> Error
  int64_t actualDim(int64_t dim, bool wrap_dim = true) const;

  // We have to override this because we opted into CustomStrides
  IntArrayRef strides_custom() const override;
  SymIntArrayRef sym_strides_custom() const override;
  // Override a bunch of methods inherited from TensorImpl to return error messages.
  bool is_contiguous_custom(at::MemoryFormat memory_format=at::MemoryFormat::Contiguous) const override;
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const override;
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;
#ifdef DEBUG
  bool has_storage() const override;
#endif

  void refreshTensorMetadata();

  // Used in torchdim. torchdim uses non-lexical BatchedTensor; the way it
  // accomplishes this is a hack where it is able to modify the levels of
  // BatchedTensor to match the level of the current vmap transform.
  void _unsafe_set_level(int64_t level) {
    level_ = level;
  }

  // Used in batching rule for in-place view operations that can change
  // the index of the bdim (think squeeze_, unsqueeze_)
  void unsafe_set_bdim(int64_t bdim) {
    // NB: you MUST call refreshTensorMetadata after doing this.
    bdim_ = bdim;
  }
 private:
  // see NOTE: [BatchedTensorImpl levels invariant]
  void checkInvariants() const;
  const char* tensorimpl_type_name() const override;

  Tensor value_;

  int64_t level_;
  int64_t bdim_;
};

// NB: We use the term "BatchedTensor" to mean a Tensor that is backed with a
// BatchedTensorImpl.
inline bool isBatchedTensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(DispatchKey::FuncTorchBatched);
}

// It is unsafe to call this on a Tensor that is not backed by a
// BatchedTensorImpl. Please use `maybeGetBatchedImpl` whenever possible.
inline BatchedTensorImpl* unsafeGetBatchedImpl(Tensor tensor) {
  return static_cast<BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

inline BatchedTensorImpl* maybeGetBatchedImpl(Tensor tensor) {
  if (!isBatchedTensor(tensor)) {
    return nullptr;
  }
  return unsafeGetBatchedImpl(std::move(tensor));
}

// Returns a bitset. If bit i is set, then that means dim i is a batchdim.
inline std::bitset<kVmapMaxTensorDims> createBatchDimBitset(int64_t dim) {
  std::bitset<kVmapMaxTensorDims> is_bdim;
  is_bdim.set(dim);
  return is_bdim;
}

// Creates a bitset for the given level
inline std::bitset<kVmapNumLevels> createVmapLevelsBitset(int64_t level) {
  std::bitset<kVmapNumLevels> result;
  result.set(level);
  return result;
}

// Use this to construct a BatchedTensor from a regular Tensor
TORCH_API Tensor makeBatched(const Tensor& tensor, int64_t dim, int64_t level);

// Adds a batch dim to `tensor`, returning a BatchedTensor
TORCH_API Tensor addBatchDim(const Tensor& tensor, int64_t dim, int64_t level);

// Certain dispatch keys must be propagated to the BatchedTensor (or, in general,
// any wrapper Tensor subclasses). This is because there are methods on Tensor
// that skip dispatch and check for the presence of a dispatch key (e.g. is_cpu()).
// TODO: should probably contain more (or all?) backend keys
constexpr DispatchKeySet kKeysToPropagateToWrapper({
  DispatchKey::Negative,
  DispatchKey::Conjugate,
  DispatchKey::XLA,
  DispatchKey::CUDA,
  DispatchKey::CPU,
});

inline DispatchKeySet getKeysToPropagateToWrapper(const Tensor& tensor, DispatchKeySet to_propagate=kKeysToPropagateToWrapper) {
  auto key_set = tensor.unsafeGetTensorImpl()->key_set();
  return key_set & kKeysToPropagateToWrapper;
}

}
}
