#pragma once

#include <bitset>

#include <ATen/ArrayRef.h>
#include <ATen/SmallVector.h>
#include <ATen/Tensor.h>

namespace at {

// We assume this in a few other places in the codebase,
// but there isn't a centralized definition.
constexpr int64_t kVmapMaxTensorDims = 64;

// Store this number of elements of BatchDims on the stack. Most people will
// probably use <= 5 nested vmaps, but adjust this number as necessary.
constexpr int64_t kBatchDimsStackSize = 5;

struct BatchDim {
  BatchDim(int64_t level, int64_t dim) : dim_(dim), level_(level) {}
  int64_t dim() const {
    return dim_;
  }
  int64_t level() const {
    return level_;
  }
 private:
  int64_t dim_;
  int64_t level_;
};

using BatchDims = SmallVector<BatchDim, kBatchDimsStackSize>;
using BatchDimsRef = ArrayRef<BatchDim>;

struct TORCH_API BatchedTensorImpl : public c10::TensorImpl {
  explicit BatchedTensorImpl(Tensor value, BatchDims bdims);

  BatchDimsRef bdims() const { return bdims_; }
  const Tensor& value() const { return value_; };

  int64_t actualDim(int64_t dim, bool wrap_dim = true) const;

  // Override a bunch of methods inherited from TensorImpl.
  bool is_contiguous(at::MemoryFormat memory_format=at::MemoryFormat::Contiguous) const override;
  IntArrayRef strides() const override;
  int64_t stride(int64_t d) const override;
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;
  bool has_storage() const override;
  const Storage& storage() const override;
  int64_t storage_offset() const override;

 private:
  Tensor value_;
  BatchDims bdims_;
};

inline bool isBatched(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(DispatchKey::Vmap);
}

inline BatchedTensorImpl* getBatched(Tensor tensor) {
  return static_cast<BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

// Returns a bitset. If bit i is set, then that means dim i is a batchdim.
inline std::bitset<kVmapMaxTensorDims> createBatchDimBitset(BatchDimsRef bdims) {
  std::bitset<kVmapMaxTensorDims> is_bdim;
  for (const auto& bdim : bdims) {
    is_bdim.set(bdim.dim());
  }
  return is_bdim;
}

inline std::ostream& operator<<(std::ostream& out, const BatchDim& bdim) {
  out << "(lvl=" << bdim.level() << ", dim=" << bdim.dim() << ")";
  return out;
}

inline Tensor makeBatched(const Tensor& tensor, BatchDims bdims) {
  TORCH_INTERNAL_ASSERT(!isBatched(tensor));
  auto tensor_dim = tensor.dim();
  TORCH_CHECK(
      tensor_dim < kVmapMaxTensorDims,
      "vmap only supports tensors of dimensionality up to ", kVmapMaxTensorDims,
      "; got a tensor with dim ", tensor_dim);
  return at::detail::make_tensor<BatchedTensorImpl>(tensor, std::move(bdims));
}

// Adds a batch dim to `tensor`, returning a Tensor backed by a BatchedTensorImpl.
TORCH_API Tensor addBatchDim(const Tensor& tensor, int64_t level, int64_t dim);


}
