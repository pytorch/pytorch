#pragma once

#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch::jit {

// A StorageGroup represents a collection of tensors that share backing storage.
class StorageGroup {
 public:
  // Every storage group must contain at least one tensor.
  explicit StorageGroup(at::Tensor* tensor) : group_{tensor} {}

  void addTensor(at::Tensor* tensor) {
    group_.push_back(tensor);
  }

  const std::vector<at::Tensor*>& group() const {
    return group_;
  }

  size_t maxTensorSize() const {
    return max_tensor_size_;
  }

  void setMaxTensorSize(size_t new_size) {
    max_tensor_size_ = new_size;
  }

  size_t numManagedTensors() const {
    return group_.size();
  }

 private:
  // The size attribute represents the amount of memory that will be
  // allocated for all tensors in this storage group. Initially it
  // is zero, eventually it gets updated by the MemoryPlanner.
  size_t max_tensor_size_ = 0;
  std::vector<at::Tensor*> group_{};
};

// A contiguous buffer of `StorageImpl`s
class ManagedStorages {
 public:
  ManagedStorages();

  ~ManagedStorages();

  void allocate(size_t capacity);

  void deallocate();

  bool is_allocated() const {
    return storages_ != nullptr;
  }

  // Append a new StorageImpl to the buffer. The new StorageImpl is given the
  // same size and allocator as `storageImpl` argument
  void append(at::StorageImpl& storageImpl);

  at::StorageImpl& operator[](size_t idx) {
    TORCH_INTERNAL_ASSERT(storages_ != nullptr);
    return storages_[idx];
  }

  const at::StorageImpl& operator[](size_t idx) const {
    TORCH_INTERNAL_ASSERT(storages_ != nullptr);
    return storages_[idx];
  }

  size_t size() const {
    return size_;
  }

  bool empty() const {
    return size_ == 0;
  }

  size_t capacity() const {
    return capacity_;
  }

 private:
  // We will use placement-new to add new storages to this buffer
  at::StorageImpl* storages_;

  // Current number of storages that have been placed into the storage buffer
  size_t size_;

  // Total allocated capacity of the storage buffer
  size_t capacity_;
};

TORCH_API std::vector<StorageGroup> assignStorageToManagedTensors(
    graph_node_list nodes,
    const ManagedTensorRanges& ranges,
    const c10::FastMap<const Value*, at::Tensor*>& tensor_value_to_tensor);

// There are three types of ops in a processed graph in Static Runtime:
//   1. op with _out variant
//   2. view-producing op
//   3. tensor-producing op (could be replaced with type 1 by adding the _out
//      variant to Static Runtime)
// In Static Runtime, type 2 ops are replaced with their corresponding copy
// versions when enable_out_variant is enabled and become type 1 ops.The memory
// planner only manages tensors that are outputs of type 1 ops. For type 3, the
// output tensors are allocated inside the operator and can't be directly
// managed by memory planner.
//
// Memory planner tries to minimize the number of memory allocations by
// tracking the output tensors of ops with _out variants with unique DataPtr
// (part of StorageImpl). It tries to do this in several steps:
//   1. record the max memory usage for each Tensor with unique DataPtr at the
//      end of each iteration
//   2. in the next iteration, allocate the buffer for the max total usage and
//      compute the offset of each allocation with regard to the single memory
//      buffer, optionally reusing memory. In the first iteration, we rely on
//      the default allocator for memory allocation.
//   3. free the buffer at the end of each iteration
// Steps 1 and 3 are handled by `deallocate()`, and step 2 by `allocate()`.
// Only models with simple output types are supported, i.e. None, Tensor or
// List/Tuple/Dict of Tensors. Complex output types such as List of Lists are
// not supported.
//
// Additional Optimizations:
//
// [Borrowed IValue Outputs]
// A few native ops (notably, `static_runtime::dict_unpack` and
// `static_runtime::VarTupleUnpack`) simply unpack IValues to a bunch of
// outputs without modification. For example, `dict_unpack` does the following:
// for each key in inputs:
//     output[i] = dict_input[key]
// To avoid refcount bumps, the outputs of these ops are non-owning references.
// This requires special logic in the memory planner - when adding an op that
// borrows outputs, be sure that the memory planner is updated accordingly!
//
// [Managed Output Tensors]
// The memory planner is able to manage output tensors if the appropriate
// `StaticModuleOptions` are set. However, the memory planner handles output
// tensors separately from regular intermediate tensors:
// 1. They don't participate in memory reuse.
// 2. The memory planner cannot reclaim their backing storage until they have
//    been explicitly freed by the client.

class MemoryPlanner {
 public:
  MemoryPlanner(
      BlockRunner* block_runner,
      const BlockInfo& block_info,
      bool enable_out_variant,
      bool manage_output_tensors);

  // disable copying and moving
  MemoryPlanner(const MemoryPlanner&) = delete;
  MemoryPlanner& operator=(const MemoryPlanner&) = delete;
  MemoryPlanner(MemoryPlanner&&) = delete;
  MemoryPlanner& operator=(MemoryPlanner&&) = delete;
  virtual ~MemoryPlanner() = default;

  void allocate();
  void deallocate();
  void deallocateOutputTensors();

  size_t total_num_managed_tensors() const {
    return num_managed_tensors_;
  }

  size_t total_reused_tensors() const {
    return reused_tensors_;
  }

  size_t total_num_managed_output_tensors() const {
    return managed_output_tensors_.size();
  }

  [[nodiscard]] size_t total_num_unmanaged() const {
    return num_unmanaged_non_scalars() + num_unmanaged_scalars();
  }

  [[nodiscard]] size_t num_unmanaged_non_scalars() const {
    return unmanaged_ivalues_.size() + unmanaged_borrowed_ivalues_.size();
  }

  [[nodiscard]] size_t num_unmanaged_scalars() const {
    return num_unmanaged_scalar_ivalues_;
  }

  size_t total_managed() const {
    return managed_bytes_;
  }

  size_t numOutputBufferBytes() const {
    return output_buffer_bytes_;
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
    if (storages_.empty()) {
      return false;
    }
    // Comparing pointers that aren't within the same array is
    // UB. We're doing fancy memory allocation stuff, so we cast to an
    // integer type and carry on.
    const auto impl_p = reinterpret_cast<uintptr_t>(impl);
    const auto start = reinterpret_cast<uintptr_t>(&storages_[0]);
    const auto end =
        reinterpret_cast<uintptr_t>(&storages_[0] + storages_.size());
    return impl_p >= start && impl_p < end;
  }

  bool overlapWithInternalBuffer(void* data_ptr) {
    return buffer_start_ <= data_ptr && data_ptr < buffer_end_;
  }

 protected:
  uint8_t* allocateBuffer(size_t num_bytes);

  size_t managed_bytes_{0};
  size_t reused_tensors_{0};

  // We allocate StorageImpls ourselves so that 1) we don't have to do
  // an extra two loads per Tensor (which will likely miss in the CPU
  // data cache) first reading the Storage (i.e., StorageImpl pointer)
  // from the TensorImpl object and then second dereferencing it and
  // 2) our memory access pattern during allocate() has high locality.
  // We don't have any guarantee that the model doesn't change the
  // Storage for managed tensors out from under us during execution,
  // so we have to check the StorageImpls each time we deallocate.
  ManagedStorages storages_;

  // Contains the size (in bytes) of the data to be allocated for each storage
  std::vector<size_t> storages_nbytes_;

 private:
  // ivalues created in one run but not managed by MemoryPlanner
  std::vector<IValue*> unmanaged_ivalues_;

  // Special class of unmanaged values: some native ops create IValues
  // in a "borrowed" state that can and must be cleaned up without a
  // reference count decrement.
  std::vector<IValue*> unmanaged_borrowed_ivalues_;

  // Even more special class of unmanaged values: if select_tensor
  // outputs are outputs of the graph, then they need to be restored
  // to an ordinary "strong reference" state.
  std::vector<IValue*> borrowed_ivalues_needing_incref_;

  std::vector<std::pair<size_t, at::Tensor*>> managed_output_tensors_{};
  at::DataPtr buffer_; // allocated each time we call Run()
  uint8_t* buffer_start_{nullptr};
  uint8_t* buffer_end_{nullptr};
  size_t num_managed_tensors_{0};
  size_t num_unmanaged_scalar_ivalues_{0};

  at::DataPtr output_buffer_;
  size_t output_buffer_bytes_{0};

  virtual void allocateManagedTensors() = 0;
  virtual void deallocateManagedTensors() = 0;

  void allocateOutputTensors();
};

class StandardMemoryPlanner : public MemoryPlanner {
 public:
  StandardMemoryPlanner(
      BlockRunner* block_runner,
      const BlockInfo& block_info,
      bool enable_out_variant,
      bool manage_output_tensors,
      bool optimize_memory);

 protected:
  void allocateManagedTensors() override;
  void deallocateManagedTensors() override;

  std::vector<StorageGroup> managed_tensors_{};
};

} // namespace torch::jit
