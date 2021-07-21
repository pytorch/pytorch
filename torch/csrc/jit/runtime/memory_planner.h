#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>

namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_API ProcessedNode {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ProcessedNode() = default;
  ProcessedNode(
      Node* n,
      std::vector<const IValue*>&& inputs,
      bool enable_out_variant);

  void run();

  Node* node() const {
    return node_;
  }

  // Input is readonly
  const IValue& Input(size_t i) const {
    DCHECK(i < inputs_.size());
    return *inputs_[i];
  }

  // Output is readwrite
  IValue& Output(size_t i) {
    DCHECK(i < outputs_.size());
    return outputs_[i];
  }

  void set_input(size_t index, const IValue* ival) {
    inputs_[index] = ival;
  }

  const std::vector<IValue>& outputs() const {
    return outputs_;
  }

  const std::vector<const IValue*>& inputs() const {
    return inputs_;
  }

  bool has_out_variant() const {
    return static_cast<bool>(fn_);
  }

  bool verify_outputs_not_overlapping_with_immutable_inputs() const;

 private:
  Node* node_;
  c10::optional<Operation> op_;
  std::function<void(ProcessedNode*)> fn_;
  std::function<void(ProcessedNode*)> native_fn_;
  std::vector<const IValue*> inputs_; // unowned
  std::vector<IValue> outputs_;
};

/// There are three types of ops in a processed graph in Static Runtime:
///   1. op with _out variant
///   2. view producing op
///   3. tensor producing op (could be replaced with type 1 by adding the _out
///      variant to Static Runtime)
/// In Static Runtime, type 2 ops are replaced with their corresponding copy
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
      std::vector<ProcessedNode>& pnodes,
      const Graph& graph,
      std::vector<IValue*> outputs,
      const std::unordered_map<const Value*, std::vector<const Value*>>&
          value_to_same_storage_values,
      const std::unordered_set<const Value*>& external_values,
      bool enable_out_variant);

  // disable copying and moving
  MemoryPlanner(const MemoryPlanner&) = delete;

  MemoryPlanner& operator=(const MemoryPlanner&) = delete;

  MemoryPlanner(MemoryPlanner&&) = delete;

  MemoryPlanner& operator=(MemoryPlanner&&) = delete;

  void allocate();

  void deallocate();

  size_t total_managed() const {
    return managed_bytes_;
  }

  size_t total_reused_tensors() const {
    return reused_tensors_;
  }

 private:
  // ivalues created in one run but not managed by MemoryPlanner
  std::vector<IValue*> unmanaged_ivalues_;

  // each pair contains the size (in bytes) of data to be allocated
  // and a vector of Tensors that should be backed by that same data.
  // Thus, if memonger is disabled, all vectors are of size 1.
  std::vector<std::pair<size_t, std::vector<at::Tensor*>>> managed_tensors_;
  at::DataPtr buffer_; // allocated each time we call Run()
  size_t managed_bytes_{0};
  size_t reused_tensors_{0};

  static size_t compute_aligned_tensor_size(size_t nbytes);

  static at::DataPtr allocate_buffer(size_t size);
};
} // namespace jit
} // namespace torch
