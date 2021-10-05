#include <torch/csrc/jit/runtime/static/memory_planner.h>

#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

static void assign_storage_to_managed_tensors(
    StaticRuntime* runtime,
    const FastSet<const Value*>& managed_tensor_values,
    const FastMap<const Value*, std::vector<const Value*>>&
        value_to_same_storage_values,
    std::vector<std::pair<size_t, std::vector<at::Tensor*>>>& managed_tensors) {
  // map Value to index to managed_storage, where multiple values can
  // map to the same index (i.e., sharing the same storage)
  FastMap<const Value*, size_t> value_to_storage_idx;

  // Snapshot of the current memory state
  for (auto& pnode : runtime->nodes()) {
    for (const auto i : c10::irange(pnode.outputs().size())) {
      auto& ival = pnode.Output(i);
      const auto* val = pnode.node()->outputs()[i];
      if (managed_tensor_values.count(val)) {
        TORCH_CHECK(ival.isTensor());
        at::Tensor* tensor = &ival.toTensor();
        auto f = value_to_storage_idx.find(val);
        if (f != value_to_storage_idx.end()) {
          auto storage_idx = f->second;
          managed_tensors[storage_idx].second.emplace_back(tensor);
        } else {
          auto p =
              std::make_pair<size_t, std::vector<at::Tensor*>>(0, {tensor});
          managed_tensors.emplace_back(std::move(p));
          // first of a group, update the value_to_storage_idx map with the
          // index
          auto f = value_to_same_storage_values.find(val);
          if (f != value_to_same_storage_values.end()) {
            auto storage_idx = managed_tensors.size() - 1;
            const auto& same_storage_values = f->second;
            for (const auto* v : same_storage_values) {
              value_to_storage_idx[v] = storage_idx;
            }
          }
        }
      }
    }
  }
}

MemoryPlanner::MemoryPlanner(
    StaticRuntime* runtime,
    const FastMap<const Value*, std::vector<const Value*>>&
        value_to_same_storage_values,
    const FastSet<const Value*>& external_values,
    bool enable_out_variant,
    bool manage_graph_output_memory) {
  // collect register indices of outputs of ops with out variant
  FastSet<const Value*> managed_tensor_values;
  FastSet<const Value*> leaked_values;
  if (enable_out_variant) {
    for (ProcessedNode& pnode : runtime->nodes()) {
      if (pnode.has_out_variant()) {
        for (const auto i : c10::irange(pnode.outputs().size())) {
          const Value* out_v = pnode.node()->outputs()[i];
          if (external_values.count(out_v)) {
            continue;
          }
          // Types are stored in the underlying TorchScript IR
          const auto& type = out_v->type();
          if (type->castRaw<TensorType>()) {
            managed_tensor_values.insert(out_v);
          } else if (runtime->is_optimizable_container_type(pnode.node())) {
            // We "leak" certain container types because their allocations
            // take a long time
            leaked_values.insert(out_v);
          }
        }
      }
    }
  }

  // collect unmanaged output ivalues
  FastSet<IValue*> unmanaged_ivalues;
  for (ProcessedNode& pnode : runtime->nodes()) {
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
  for (const Value* output : runtime->graph().outputs()) {
    managed_tensor_values.erase(output);
  }
  for (IValue* output : runtime->outputs()) {
    unmanaged_ivalues.erase(output);
  }

  // copy to unmanaged_ivalues_
  unmanaged_ivalues_.reserve(unmanaged_ivalues.size());
  unmanaged_ivalues_.insert(
      unmanaged_ivalues_.begin(),
      unmanaged_ivalues.begin(),
      unmanaged_ivalues.end());

  if (enable_out_variant) {
    ::torch::jit::assign_storage_to_managed_tensors(
        runtime,
        managed_tensor_values,
        value_to_same_storage_values,
        managed_tensors_);
  }

  num_managed_tensors_ = 0;
  for (const auto& ms : managed_tensors_) {
    num_managed_tensors_ += ms.second.size();
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

  const bool have_managed_tensor_storages = !managed_tensor_storages_.empty();
  reused_tensors_ = 0;
  auto storageIdx = 0;
  for (const auto& ms : managed_tensors_) {
    auto tensor_size = ms.first;
    if (tensor_size == 0) {
      continue;
    }
    const auto& tensors = ms.second;
    DCHECK_LE(offset + tensor_size, managed_bytes_);
    void* src = static_cast<void*>(start + offset);

#define INNER_TENSOR_LOOP_BODY(tensor, storage)                             \
  do {                                                                      \
    DCHECK_EQ(tensor->device().type(), c10::DeviceType::CPU);               \
    storage.set_data_ptr_noswap(                                            \
        at::DataPtr(src, src, nullptr, c10::Device(c10::DeviceType::CPU))); \
    storage.set_nbytes(tensor_size);                                        \
    reused_tensors_++;                                                      \
  } while (0)

    if (C10_UNLIKELY(!have_managed_tensor_storages)) {
      for (auto* tensor : tensors) {
        const auto& storage = tensor->storage();
        INNER_TENSOR_LOOP_BODY(tensor, storage);
      }
    } else {
      for (auto* tensor : tensors) {
        const auto& storage = *managed_tensor_storages_[storageIdx++];
        DCHECK_EQ(&storage, &tensor->storage());
        INNER_TENSOR_LOOP_BODY(tensor, storage);
      }
    }
#undef INNER_TENSOR_LOOP_BODY
    reused_tensors_--;

    offset += tensor_size;
  }
  DCHECK_EQ(offset, managed_bytes_);
}

void MemoryPlanner::deallocate() {
  managed_bytes_ = 0;

  // free memory used by outputs of ops in out variants
  // but keep the TensorImpl and StorageImpl around.

  // We don't have any guarantee that the model doesn't change the
  // Storage for managed tensors out from under us during execution,
  // so we have to grab the Storages each time we deallocate.
  managed_tensor_storages_.clear();
  managed_tensor_storages_.reserve(num_managed_tensors_);
  for (auto& ms : managed_tensors_) {
    const auto& tensors = ms.second;
    size_t max = ms.first;
    for (auto& tensor : tensors) {
      const auto& storage = tensor->storage();
      size_t current_size = compute_aligned_tensor_size(storage.nbytes());
      storage.unsafeGetStorageImpl()->reset();
      managed_tensor_storages_.push_back(&storage);
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
