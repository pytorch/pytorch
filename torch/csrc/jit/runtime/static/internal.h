#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/variant.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/static/ProcessedNodeInputs.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

TORCH_API std::string dumpValueSet(
    const FastSet<const Value*>& value_set,
    const char* set_name = "");

TORCH_API inline bool doesNotHeapAllocateWhenStoredInIValue(const Type& type) {
  switch (type.kind()) {
    // NOTE: NumberType may allocate because it includes complex.
    case TypeKind::NoneType:
    case TypeKind::IntType:
    case TypeKind::FloatType:
    case TypeKind::BoolType:
    case TypeKind::DeviceObjType:
    case TypeKind::StreamObjType:
      return true;
    default:
      return false;
  }
}

bool mayContainAlias(AliasDb& db, const Value* a, const Value* b);

bool mayContainAlias(
    AliasDb& db,
    const FastSet<const Value*>& a,
    const FastSet<const Value*>& b);

// Group values used by `graph` into three categories:
//
// - output_aliases:
//     values that are either outputs or contain aliases of outputs
// - external_aliases:
//     values that are inputs, constants, or their aliases.
//     The output aliases that end up here are as a result of aliasDb failing to
//     recognize them as outputs due to collection object (e.g., Tuple) aliasing
//     inputs.
// Values that dont't show up in output_aliases or external_aliases are created
// and consumed within the graph.
class ValueGroup {
 public:
  explicit ValueGroup(
      const std::shared_ptr<torch::jit::Graph>& graph,
      AliasDb& db);

  bool isExternalAlias(const Value* value) const {
    return external_aliases_.find(value) != external_aliases_.end();
  }

  bool isOutputAlias(const Value* value) const {
    return output_aliases_.find(value) != output_aliases_.end();
  }

  bool isAlwaysAlive(const Value* value) const {
    return isExternalAlias(value) || isOutputAlias(value);
  }

  std::string toString() const {
    return c10::str(
        dumpValueSet(output_aliases_, "ValueGroup::output_aliases_"),
        "\n",
        dumpValueSet(external_aliases_, "ValueGroup::external_aliases_"));
  }

 private:
  FastSet<const Value*> output_aliases_;
  FastSet<const Value*> external_aliases_;
};

class TORCH_API ManagedTensorRanges {
 public:
  ManagedTensorRanges() = default;
  ManagedTensorRanges(
      const std::shared_ptr<Graph>& graph,
      const FastSet<const Value*>& managed_tensor_values);

  // If true, then this node is the last use of at least one
  // managed tensor. availableTensorsAfterNode(node) will return a vector
  // of the managed tensors that are available for re-use
  // in the nodes following this one.
  bool nodeFreesManagedTensors(Node* node) const;
  const std::vector<const Value*>& availableTensorsAfterNode(Node* node) const;

  // True if the value has a tracked lifetime and lifetime.start ==
  // lifetime.end. "Unused" does not imply "unmanaged" -
  // managed tensors can be unused if they're not passed to any ops!
  bool isUnusedValue(const Value* value) const;

  // For testing. True if v1 and v2 are both mutable types and have lifetimes
  // that overlap.
  bool lifetimesOverlap(const Value* v1, const Value* v2) const;

 private:
  struct Lifetime {
    Lifetime(size_t start_, size_t end_) : start(start_), end(end_) {}
    size_t start;
    size_t end;
  };

  // Returns nullptr if we are not tracking the lifetime of value
  Lifetime* getLifetime(const Value* value);
  const Lifetime* getLifetime(const Value* value) const;
  // Collect all values in the input that have tracked lifetimes.
  // A value's lifetime may not be tracked if it is a graph input
  // or immutable type (containers with at least one mutable
  // type are mutable)
  std::vector<const Value*> collectValuesWithTrackedLifetimes(
      at::ArrayRef<const Value*> values);

  // Maps Node* to the set of managed tensors that are now available
  // for re-use after this node.
  FastMap<Node*, std::vector<const Value*>> node_to_newly_free_tensors_{};
  // Maps each Value* to its lifetime (start node index, end node index)
  FastMap<const Value*, Lifetime> value_lifetimes_{};
};

class TORCH_API ProcessedFunction {
 public:
  ProcessedFunction(
      Node* node,
      bool enable_out_variant,
      bool check_memory_overlap);

  enum class Kind : uint8_t {
    kOutVariant,
    kNativeFunction,
    kInterpreterFallback,
  };

  const std::function<void(ProcessedNode*)>& f() const {
    return f_;
  }

  Kind kind() const {
    return kind_;
  }

  bool checkMemoryOverlap() const {
    return check_memory_overlap_;
  }

 private:
  std::function<void(ProcessedNode*)> f_;
  Kind kind_{ProcessedFunction::Kind::kOutVariant};
  bool check_memory_overlap_{false};
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_API ProcessedNode {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ProcessedNode() = default;
  // ProcessedNodes are created within StaticModule and then
  // associated with a shared values array using set_values() when
  // they are copied into a StaticRuntime.
  ProcessedNode(
      Node* n,
      ProcessedFunction* fn,
      ProcessedNodeInputs inputs,
      uint16_t outputs_offset);

  ProcessedNode(const ProcessedNode&) = default;
  ProcessedNode& operator=(const ProcessedNode&) = default;

  // These should be noexcept, but some Android build is failing
  // saying the noexcept specification doesn't match the calculated
  // one. Maybe c10::variant is throwing it off?
  ProcessedNode(ProcessedNode&&) = default;
  ProcessedNode& operator=(ProcessedNode&&) = default;

  void run();

  Node* node() const {
    return node_;
  }

  // Input is readonly
  C10_NODISCARD const IValue& Input(uint32_t i) const {
    return values_[inputs_[i]];
  }

  // Output is readwrite
  IValue& Output(uint32_t i) {
    DCHECK(i < num_outputs_);
    return values_[outputs_offset_ + i];
  }

  C10_NODISCARD const IValue& Output(uint32_t i) const {
    DCHECK(i < num_outputs_);
    return values_[outputs_offset_ + i];
  }

  C10_NODISCARD c10::ArrayRef<const IValue> outputs() const {
    return c10::ArrayRef<const IValue>(values_ + outputs_offset_, num_outputs_);
  }

  C10_NODISCARD auto num_outputs() const {
    return num_outputs_;
  }

  C10_NODISCARD uint16_t num_inputs() const {
    return inputs_.size();
  }

  std::vector<IValue> inputs_ivalue_vec() const;

  bool has_out_variant() const {
    return fn_->kind() == ProcessedFunction::Kind::kOutVariant;
  }

  bool has_native() const {
    return fn_->kind() == ProcessedFunction::Kind::kNativeFunction;
  }

#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  const char* get_op_name() const {
    return op_name_;
  }
#endif

  bool check_outputs_for_memory_overlap() const {
    return fn_->checkMemoryOverlap();
  }

  void set_outputs_memory_overlap_detected() {
    overlap_detected_ = true;
  }

  bool outputs_memory_overlap_detected() {
    return overlap_detected_;
  }

  void verify_and_correct_memory_overlap();

  void set_values(IValue* values) {
    DCHECK(values_ == nullptr);
    values_ = values;
  }

  C10_NODISCARD uint16_t output_ivalue_index(uint16_t i) const {
    return outputs_offset_ + i;
  }
  // used in debug mode
  bool verify_no_memory_overlap(bool force_check = false) const;

 private:
  C10_NODISCARD bool verify_outputs_dont_overlap_each_other() const;

  C10_NODISCARD bool verify_inputs_dont_overlap_outputs(bool force_check) const;

  Node* node_;
  const ProcessedFunction* fn_;
  bool overlap_detected_{false};
  ProcessedNodeInputs inputs_;
  uint16_t outputs_offset_;
  uint16_t num_outputs_;
  IValue* values_ = nullptr; // unowned
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  const char* op_name_;
#endif
};

} // namespace jit
} // namespace torch
