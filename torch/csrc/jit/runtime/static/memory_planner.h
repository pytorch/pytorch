#pragma once

#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

/// There are three types of ops in a processed graph in Static Runtime:
///   1. op with _out variant
///   2. view producing op
///   3. tensor producing op (could be replaced with type 1 by adding the _out
///      variant to Static Runtime)
/// In Static Runtime, type 2 ops are replaced with their corespoinding copy
/// versions when enable_out_variant is enabled and become type 1 ops.The memory
/// planner only manages tensors that are outputs of type 1 ops. For type 3, the
/// output tensors are allocated inside the operator and can't be directly
/// managed by memory planner.
///
/// Memory planner tries to minimize the number of memory allocations by
/// tracking the output tensors of ops with _out variants with unique DataPtr
/// (part of StorageImpl). It tries to do this in several steps:
///   1. record the max memory usage for each Tensor with unique DataPtr at the
///      end of each iteration
///   2. in the next iteration, allocate the buffer for the max total usage and
///      compute the offset of each allocation with regard to the single memory
///      buffer, optionally reusing memory. In the first iteration, we rely on
///      the default allocator for memory allocation.
///   3. free the buffer at the end of each iteration
/// Steps 1 and 3 are handled by `deallocate()`, and step 2 by `allocate()`.
/// Only models with simple output types are supported, i.e. None, Tensor or
/// List/Tuple/Dict of Tensors. Complex output types such as List of Lists are
/// not supported.

class MemoryPlanner {
 public:
  explicit MemoryPlanner(
      StaticRuntime* runtime,
      const FastMap<const Value*, std::vector<const Value*>>&,
      const ValueGroup& value_group,
      bool enable_out_variant,
      bool manage_output_tensors);
  // disable copying and moving
  MemoryPlanner(const MemoryPlanner&) = delete;
  MemoryPlanner& operator=(const MemoryPlanner&) = delete;
  MemoryPlanner(MemoryPlanner&&) = delete;
  MemoryPlanner& operator=(MemoryPlanner&&) = delete;

  void allocate();
  void deallocate();
  void deallocateOutputTensors();

  size_t total_num_managed_tensors() const {
    return num_managed_tensors_;
  }

  size_t total_num_managed_output_tensors() const {
    return managed_output_tensors_.size();
  }

  size_t total_num_unmanaged() const {
    return unmanaged_ivalues_.size();
  }

  size_t total_managed() const {
    return managed_bytes_;
  }

  size_t total_reused_tensors() const {
    return reused_tensors_;
  }

  size_t numOutputBufferBytes() const {
    return output_buffer_bytes_;
  }

  bool isManagedOutputTensorValue(const Value* value) const {
    return managed_output_tensor_values_.find(value) !=
        managed_output_tensor_values_.end();
  }

  // Check if `ivalue` is contained as a managed tensor. Only used in DCHECK().
  bool isManagedOutputTensor(const IValue& ivalue) const {
    if (!output_buffer_ || // output buffer got already deallocated.
        output_buffer_bytes_ == 0 || // memory planning is not yet initialized.
        !ivalue.isTensor() // a non-tensor is never managed
    ) {
      return false;
    }
    const auto& tensor = ivalue.toTensor();
    if (!tensor.has_storage() || !tensor.storage().data_ptr()) {
      return false;
    }
    // TODO: Improve this once D31357486 is landed.
    uint8_t* tensor_ptr =
        static_cast<uint8_t*>(tensor.storage().data_ptr().get());
    uint8_t* buffer_start = static_cast<uint8_t*>(output_buffer_.get());
    uint8_t* buffer_end = buffer_start + output_buffer_bytes_;
    return buffer_start <= tensor_ptr && tensor_ptr < buffer_end;
  }

  bool isManagedStorageImpl(const at::StorageImpl* impl) const {
    if (managed_tensor_storage_impls_.empty()) {
      return false;
    }
    // Comparing pointers that aren't within the same array is
    // UB. We're doing fancy memory allocation stuff, so we cast to an
    // integer type and carry on.
    const auto impl_p = reinterpret_cast<uintptr_t>(impl);
    const auto start =
        reinterpret_cast<uintptr_t>(&managed_tensor_storage_impls_[0]);
    const auto end = reinterpret_cast<uintptr_t>(
        &managed_tensor_storage_impls_[managed_tensor_storage_impls_.size()]);
    return impl_p >= start && impl_p < end;
  }

 private:
  // ivalues created in one run but not managed by MemoryPlanner
  std::vector<IValue*> unmanaged_ivalues_;

  // each pair contains the size (in bytes) of data to be allocated
  // and a vector of Tensors' storages that should be backed by that
  // same data. Thus, if memonger is disabled, all vectors are of
  // size 1.

  // We allocate StorageImpls ourselves so that 1) we don't have to do
  // an extra two loads per Tensor (which will likely miss in the CPU
  // data cache) first reading the Storage (i.e., StorageImpl pointer)
  // from the TensorImpl object and then second dereferencing it and
  // 2) our memory access pattern during allocate() has high locality.
  std::vector<std::pair<size_t, at::StorageImpl>>
      managed_tensor_storage_impls_{};
  // We don't have any guarantee that the model doesn't change the
  // Storage for managed tensors out from under us during execution,
  // so we have to check the StorageImpls each time we deallocate.
  std::vector<std::pair<size_t, std::vector<at::Tensor*>>> managed_tensors_;
  at::DataPtr buffer_; // allocated each time we call Run()
  size_t num_managed_tensors_{0};
  size_t managed_bytes_{0};
  size_t reused_tensors_{0};

  // Since output tensors are alive after one inference, their storage
  // is managed differently (e.g., deallocation happens on the client side).
  FastSet<const Value*> managed_output_tensor_values_{};
  std::vector<std::pair<size_t, at::Tensor*>> managed_output_tensors_{};
  at::DataPtr output_buffer_;
  size_t output_buffer_bytes_{0};

  void allocateManagedTensors();
  void allocateOutputTensors();

  static size_t compute_aligned_tensor_size(size_t nbytes);
  static at::DataPtr allocate_buffer(size_t size);
};

} // namespace jit
} // namespace torch
