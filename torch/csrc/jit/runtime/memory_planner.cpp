#include <ATen/MemoryOverlap.h>
#include <jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/memory_planner.h>

namespace torch {
namespace jit {

static void assign_storage_to_managed_tensors(
    std::vector<ProcessedNode>& pnodes,
    const std::unordered_set<const Value*>& managed_tensor_values,
    const std::unordered_map<const Value*, std::vector<const Value*>>&
        value_to_same_storage_values,
    std::vector<std::pair<size_t, std::vector<at::Tensor*>>>& managed_tensors) {
  // map Value to index to managed_storage, where multiple values can
  // map to the same index (i.e., sharing the same storage)
  std::unordered_map<const Value*, size_t> value_to_storage_idx;

  // Snapshot of the current memory state
  for (auto& pnode : pnodes) {
    for (const auto i : c10::irange(pnode.outputs().size())) {
      auto& ival = pnode.Output(i);
      const auto* val = pnode.node()->outputs()[i];
      if (managed_tensor_values.count(val)) {
        TORCH_CHECK(ival.isTensor())
        at::Tensor* tensor = &ival.toTensor();

        if (value_to_storage_idx.count(val)) {
          managed_tensors[value_to_storage_idx[val]].second.emplace_back(
              tensor);
        } else {
          auto p =
              std::make_pair<size_t, std::vector<at::Tensor*>>(0, {tensor});
          managed_tensors.emplace_back(std::move(p));
          // first of a group, update the value_to_storage_idx map with the
          // index
          if (value_to_same_storage_values.count(val)) {
            auto storage_idx = managed_tensors.size() - 1;
            for (const auto* v : value_to_same_storage_values.at(val)) {
              value_to_storage_idx[v] = storage_idx;
            }
          }
        }
      }
    }
  }
}

ProcessedNode::ProcessedNode(
    Node* node,
    std::vector<const IValue*>&& inputs,
    bool enable_out_variant)
    : node_(node), inputs_(std::move(inputs)) {
  // TODO leverage type information
  outputs_.resize(node->outputs().size());

  if (enable_out_variant && (fn_ = getOutOfPlaceOperation(node))) {
    VLOG(1) << "Switch to out variant for node: " << PrintNode(node);
    return;
  }
  if (!fn_ && (native_fn_ = getNativeOperation(node))) {
    VLOG(1) << "Switch to native impl for node: " << PrintNode(node);
    return;
  }
  {
    const Operator& op = node->getOperator();
    op_ = op.getOperation(node);
    VLOG(1) << "Fallback interpreter for node: " << PrintNode(node);
  }
}

void ProcessedNode::run() {
  DCHECK(verify_outputs_not_overlapping_with_immutable_inputs());
  if (fn_) {
    fn_(this);
  } else if (native_fn_) {
    native_fn_(this);
  } else {
    std::vector<IValue> stack;
    const size_t size = node_->inputs().size();
    stack.reserve(size);
    for (const auto i : c10::irange(size)) {
      stack.emplace_back(Input(i));
    }

    DCHECK(op_);
    op_->operator()(&stack);

    DCHECK_EQ(stack.size(), node_->outputs().size());
    for (const auto i : c10::irange(node_->outputs().size())) {
      Output(i) = std::move(stack[i]);
    }
  }
}

bool ProcessedNode::verify_outputs_not_overlapping_with_immutable_inputs()
    const {
  auto schema = node()->maybeSchema();
  if (!schema || schema->is_mutable()) {
    return true;
  }
  for (const IValue* in : inputs_) {
    if (!in->isTensor()) {
      continue;
    }
    const auto& in_t = in->toTensor();
    for (const IValue& out : outputs_) {
      if (!out.isTensor()) {
        continue;
      }
      const auto& out_t = out.toTensor();
      at::MemOverlapStatus status = at::get_overlap_status(in_t, out_t);
      if (status != at::MemOverlapStatus::NO) {
        return false;
      }
    }
  }
  return true;
}

MemoryPlanner::MemoryPlanner(
    std::vector<ProcessedNode>& pnodes,
    const Graph& graph,
    const std::vector<IValue*> outputs,
    const std::unordered_map<const Value*, std::vector<const Value*>>&
        value_to_same_storage_values,
    const std::unordered_set<const Value*>& external_values,
    bool enable_out_variant) {
  // collect register indices of outputs of ops with out variant
  std::unordered_set<const Value*> managed_tensor_values;
  std::unordered_set<const Value*> leaked_values;
  if (enable_out_variant) {
    for (const ProcessedNode& pnode : pnodes) {
      if (pnode.has_out_variant()) {
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        for (const auto i : c10::irange(pnode.outputs().size())) {
          const Value* out_v = pnode.node()->outputs()[i];
          if (external_values.count(out_v)) {
            continue;
          }
          // Types are stored in the underlying TorchScript IR
          const auto& type = out_v->type();
          if (type->cast<TensorType>()) {
            managed_tensor_values.insert(out_v);
          } else if (isOptimizableContainerType(pnode.node())) {
            // We "leak" certain container types because their allocations take
            // a long time
            leaked_values.insert(out_v);
          }
        }
      }
    }
  }

  // collect unmanaged output ivalues
  std::unordered_set<IValue*> unmanaged_ivalues;
  for (ProcessedNode& pnode : pnodes) {
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (const auto i : c10::irange(pnode.outputs().size())) {
      // Types are stored in the underlying TorchScript IR
      const Value* out_v = pnode.node()->outputs()[i];
      if (managed_tensor_values.count(out_v) || leaked_values.count(out_v)) {
        continue;
      }
      IValue& out = pnode.Output(i);
      unmanaged_ivalues.insert(&out);
    }
  }
  // since runtime->outputs() escape from run(), remove them from
  // managed_tensor_values and from unmanaged_ivalues
  for (const Value* output : graph.outputs()) {
    managed_tensor_values.erase(output);
  }
  for (IValue* output : outputs) {
    unmanaged_ivalues.erase(output);
  }

  // copy to unmanaged_ivalues_
  for (IValue* out : unmanaged_ivalues) {
    unmanaged_ivalues_.emplace_back(out);
  }

  if (enable_out_variant) {
    ::torch::jit::assign_storage_to_managed_tensors(
        pnodes,
        managed_tensor_values,
        value_to_same_storage_values,
        managed_tensors_);
  }
}

// Don't change the size if it is already aligned, otherwise increase the size
// to make it aligned.
size_t MemoryPlanner::compute_aligned_tensor_size(size_t nbytes) {
  // Note: everything below is size_t
  return (nbytes + c10::gAlignment - 1) & (~(c10::gAlignment - 1));
}

at::DataPtr MemoryPlanner::allocate_buffer(size_t size) {
  at::Allocator* allocator = c10::GetCPUCachingAllocator();
  return allocator->allocate(size);
}

void MemoryPlanner::allocate() {
  if (managed_bytes_ == 0) {
    return;
  }
  buffer_ = allocate_buffer(managed_bytes_);

  size_t offset = 0;
  uint8_t* start = static_cast<uint8_t*>(buffer_.get());

  reused_tensors_ = 0;
  for (const auto& ms : managed_tensors_) {
    auto tensor_size = ms.first;
    if (tensor_size == 0) {
      continue;
    }
    const auto& tensors = ms.second;
    DCHECK_LE(offset + tensor_size, managed_bytes_);
    void* src = static_cast<void*>(start + offset);

    for (auto* tensor : tensors) {
      tensor->storage().set_data_ptr_noswap(
          at::DataPtr(src, src, nullptr, tensor->device()));
      tensor->storage().set_nbytes(tensor_size);
      reused_tensors_++;
    }
    reused_tensors_--;

    offset += tensor_size;
  }
  DCHECK_EQ(offset, managed_bytes_);
}

void MemoryPlanner::deallocate() {
  managed_bytes_ = 0;

  // free memory used by outputs of ops in out variants
  // but keep the TensorImpl and StorageImpl around
  for (auto& ms : managed_tensors_) {
    const auto& tensors = ms.second;
    size_t max = ms.first;
    for (auto& tensor : tensors) {
      size_t current_size =
          compute_aligned_tensor_size(tensor->storage().nbytes());
      tensor->storage().unsafeGetStorageImpl()->reset();
      max = std::max(max, current_size);
    }
    // Static runtime does not know the size of tensors statically, so we use
    // the tensor size from the previous run to allocate tensors for the next
    // run (following C2 tradition), exploiting the fact that tensor storage
    // size does not have to match that of real tensor size. The following logic
    // records the tensor storage size for the next run.
    ms.first = max;
    managed_bytes_ += max;
  }

  // for unmanaged ivalues (either tensor or non-tensor), we reset the *iv so
  // that the objects pointed to by *iv may be reclaimed by reference counting
  for (auto& iv : unmanaged_ivalues_) {
    *iv = IValue();
  }
  buffer_ = {};
}

} // namespace jit
} // namespace torch
