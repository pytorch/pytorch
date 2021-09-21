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

class TORCH_API MemoryPlanner {
 public:
  explicit MemoryPlanner(
      StaticRuntime* runtime,
      const FastMap<const Value*, std::vector<const Value*>>&,
      const FastSet<const Value*>& external_values,
      bool enable_out_variant,
      bool manage_graph_output_memory);
  // disable copying and moving
  MemoryPlanner(const MemoryPlanner&) = delete;
  MemoryPlanner& operator=(const MemoryPlanner&) = delete;
  MemoryPlanner(MemoryPlanner&&) = delete;
  MemoryPlanner& operator=(MemoryPlanner&&) = delete;

  void allocate();
  void deallocate();

  size_t total_num_managed_tensors() const {
    return num_managed_tensors_;
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

  static size_t compute_aligned_tensor_size(size_t nbytes);

 private:
  // ivalues created in one run but not managed by MemoryPlanner
  std::vector<IValue*> unmanaged_ivalues_;

  // each pair contains the size (in bytes) of data to be allocated
  // and a vector of Tensors that should be backed by that same data.
  // Thus, if memonger is disabled, all vectors are of size 1.
  std::vector<std::pair<size_t, std::vector<at::Tensor*>>> managed_tensors_;
  at::DataPtr buffer_; // allocated each time we call Run()
  size_t num_managed_tensors_{0};
  size_t managed_bytes_{0};
  size_t reused_tensors_{0};

  // since output tensors are alive after one inference, their storage
  // is managed differently (e.g., deallocation happens at client side)
  // std::vector<std::pair<size_t, std::vector<at::Tensor*>>>
  //     managed_output_storage_;
  // size_t managed_output_bytes_{0};
  // size_t reused_output_tensors_{0};
  // at::DataPtr output_buffer_; // allocated each time we call Run()

  static at::DataPtr allocate_buffer(size_t size);
};

} // namespace jit
} // namespace torch
