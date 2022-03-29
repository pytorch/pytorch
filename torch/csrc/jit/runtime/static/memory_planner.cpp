#include <torch/csrc/jit/runtime/static/memory_planner.h>

#include <ATen/NativeFunctions.h>
#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <iterator>

namespace torch {
namespace jit {

namespace {

bool isUnmanagedSpecialCase(const ProcessedNode& pnode, size_t output_idx) {
  DCHECK(output_idx < pnode.outputs().size());
  static const auto to_maybe_copy_out_symbol =
      c10::Symbol::fromQualString("static_runtime::to_maybe_copy_out");
  // Heuristic and special case:
  // If to_maybe_copy_out did not actually do anything in the
  // first iteration, assume it will continue to not do anything
  // and avoid managing its output.
  return pnode.node()->kind() == to_maybe_copy_out_symbol &&
      pnode.Output(output_idx).isNone();
}

void mapOverManagedTensors(
    const std::vector<ProcessedNode>& nodes,
    const FastSet<const Value*>& managed_tensor_values,
    const std::function<void(const Value*, at::Tensor*)>& action) {
  for (auto& pnode : nodes) {
    auto* node = pnode.node();
    for (const auto output_idx : c10::irange(node->outputs().size())) {
      auto* output = node->output(output_idx);

      if (managed_tensor_values.find(output) == managed_tensor_values.end()) {
        continue;
      }

      auto& ival = pnode.Output(output_idx);

      // ival is allowed to be None in special cases, e.g. to_maybe_copy_out
      DCHECK(
          ival.isTensor() ||
          (ival.isNone() && isUnmanagedSpecialCase(pnode, output_idx)));

      if (ival.isTensor()) {
        action(
            output,
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            const_cast<at::Tensor*>(&ival.toTensor()));
      }
    }
  }
}

FastMap<const Value*, at::Tensor*> tensorValueToTensor(
    const std::vector<ProcessedNode>& nodes,
    const FastSet<const Value*>& managed_tensor_values) {
  FastMap<const Value*, at::Tensor*> result;
  auto action = [&result](const Value* value, at::Tensor* tensor) mutable {
    result.emplace(value, tensor);
  };
  mapOverManagedTensors(nodes, managed_tensor_values, action);
  return result;
}

std::vector<std::pair<at::Tensor*, const Value*>> collectManagedTensors(
    const std::vector<ProcessedNode>& nodes,
    const FastSet<const Value*>& managed_tensor_values) {
  std::vector<std::pair<at::Tensor*, const Value*>> result;
  auto action = [&result](const Value* value, at::Tensor* tensor) mutable {
    result.emplace_back(tensor, value);
  };
  mapOverManagedTensors(nodes, managed_tensor_values, action);
  return result;
}

// Don't change the size if it is already aligned, otherwise increase the size
// to make it aligned.
size_t compute_aligned_tensor_size(size_t nbytes) {
  // Note: everything below is size_t
  return (nbytes + c10::gAlignment - 1) & (~(c10::gAlignment - 1));
}

at::DataPtr allocate_buffer(size_t size) {
  at::Allocator* allocator = c10::GetCPUCachingAllocator();
  return allocator->allocate(size);
}

} // namespace

size_t assignOffsetsNaive(
    std::vector<PrecomputedOffsetsMemoryPlanner::ManagedTensor>&
        managed_tensors) {
  size_t managed_bytes = 0;
  for (auto& tensor : managed_tensors) {
    tensor.offset = managed_bytes;
    managed_bytes += tensor.size;
  }
  return managed_bytes;
}

// [Precomputed Offsets Memory Planning Algorithm]
// This algorithm is taken from this paper "Efficient Memory Management for Deep
// Neural Net Inference" [arXiv:2001.03288] in the "Greedy by Size for Offset
// Calculation" section. Generally, this algorithm performs better than the one
// implemented in the standard memory planner. However, there are tradeoffs:
//
// 1) The complexity of computing the offsets is roughly O(n^2), which makes the
//    first iteration longer
//
// 2) Since re-computing offsets is slow, this
//    implementation does not change the offsets once they have
//    been set. This means that warming up with tensors that are too small can
//    cause performance to degrade due to dynamic allocations.
//    TODO(mikeiovine): we should fall back to the standard algorithm if we
//    detect too many reallocations.
//
// It is therefore best to use this algorithm in cases where you are able to
// to warm up the model with the largest possible inputs.
size_t assignOffsetsOptimized(
    std::vector<PrecomputedOffsetsMemoryPlanner::ManagedTensor>&
        managed_tensors,
    const ManagedTensorRanges& ranges) {
  size_t managed_bytes = 0;
  // offset -> (tensor size, tensor value) collection, sorted by offset.
  std::map<size_t, std::vector<std::pair<size_t, const Value*>>>
      processed_tensors;

  std::sort(
      managed_tensors.begin(),
      managed_tensors.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.size > rhs.size; });

  for (auto& managed_tensor : managed_tensors) {
    const auto size = managed_tensor.size;
    auto* tensor = managed_tensor.value;

    size_t prev_offset = 0;
    c10::optional<size_t> best_offset = c10::nullopt;
    size_t smallest_gap = std::numeric_limits<size_t>::max();

    for (auto& size_and_tensors : processed_tensors) {
      auto x_offset = size_and_tensors.first;
      auto& tensors = size_and_tensors.second;
      for (auto& tensor_size_and_tensor : tensors) {
        auto x_size = tensor_size_and_tensor.first;
        auto* x = tensor_size_and_tensor.second;

        if (!ranges.lifetimesOverlap(x, tensor)) {
          continue;
        }

        auto new_offset = x_offset + x_size;
        if (x_offset < prev_offset) {
          prev_offset = std::max(prev_offset, new_offset);
          // prev_offset can't get any bigger since the tensors are sorted by
          // size, so there's no need to consider the rest of the tensors for
          // this offset value
          break;
        }

        const auto gap = x_offset - prev_offset;
        if (gap >= size && gap < smallest_gap) {
          smallest_gap = gap;
          best_offset = prev_offset;
        }
        prev_offset = new_offset;
        // Again, prev_offset can't get any bigger since the tensors are sorted
        // by size, so we're free to skip over the rest of the tensors with this
        // offset value.
        break;
      }
    }

    const auto new_offset =
        best_offset.has_value() ? *best_offset : prev_offset;
    managed_tensor.offset = new_offset;
    processed_tensors[new_offset].emplace_back(size, tensor);
    managed_bytes = std::max(managed_bytes, new_offset + size);
  }
  return managed_bytes;
}

std::vector<StorageGroup> assignStorageToManagedTensors(
    graph_node_list nodes,
    const ManagedTensorRanges& ranges,
    const FastMap<const Value*, at::Tensor*>& tensor_value_to_tensor) {
  std::vector<StorageGroup> managed_tensor_groups;
  // This set maps each Value* to its assigned storage group.
  FastMap<const Value*, size_t> storage_group_mapping;
  // On each iteration, this vector stores the set of storage groups that
  // are available for re-use.
  std::vector<size_t> free_storage_groups;

  auto makeNewStorageGroup = [&](const Value* value) {
    const auto storage_group = managed_tensor_groups.size();
    storage_group_mapping.emplace(value, storage_group);
    auto* tensor_ptr = tensor_value_to_tensor.at(value);
    managed_tensor_groups.emplace_back(tensor_ptr);
  };

  auto assignToAvailableStorageGroup = [&](const Value* value) {
    DCHECK(!free_storage_groups.empty());
    const auto storage_group = free_storage_groups.back();
    DCHECK_LT(storage_group, managed_tensor_groups.size());
    storage_group_mapping.emplace(value, storage_group);
    auto* tensor_ptr = tensor_value_to_tensor.at(value);
    managed_tensor_groups[storage_group].addTensor(tensor_ptr);
    free_storage_groups.pop_back();
  };

  auto isManagedTensor = [&](const Value* value) {
    return tensor_value_to_tensor.find(value) != tensor_value_to_tensor.end();
  };

  for (auto* node : nodes) {
    // Assign storage groups to outputs
    for (const auto output_idx : c10::irange(node->outputs().size())) {
      Value* output = node->output(output_idx);
      if (!isManagedTensor(output)) {
        continue;
      }
      if (free_storage_groups.empty()) {
        makeNewStorageGroup(output);
        continue;
      }
      assignToAvailableStorageGroup(output);
    }

    // This node may be the last use of some managed tensors. If so, we
    // can mark the corresponding storage groups as free.
    if (ranges.nodeFreesManagedTensors(node)) {
      const auto& new_free_tensors =
          ranges.availableTensorValuesAfterNode(node);
      for (auto* tensor_value : new_free_tensors) {
        // We need to check this here to handle special cases like
        // to_maybe_copy_out. We don't know if the tensor value is managed until
        // after the first iter, but `ranges` is initialized at load time!
        if (!isManagedTensor(tensor_value)) {
          continue;
        }
        const auto storage_group = storage_group_mapping.at(tensor_value);
        free_storage_groups.push_back(storage_group);
      }
    }
  }
  return managed_tensor_groups;
}

namespace {

bool setIncludes(const FastSet<const Value*>& set, const Value* v) {
  return set.find(v) != set.end();
}

std::vector<std::pair<size_t, at::Tensor*>> assignStorageToOutputTensors(
    BlockRunner* block_runner,
    const FastSet<const Value*>& managed_output_tensor_values) {
  std::vector<std::pair<size_t, at::Tensor*>> managed_output_tensors;
  for (auto& pnode : block_runner->nodes()) {
    for (const auto i : c10::irange(pnode.outputs().size())) {
      auto& ival = pnode.Output(i);
      const auto* val = pnode.node()->outputs()[i];
      if (!setIncludes(managed_output_tensor_values, val) ||
          isUnmanagedSpecialCase(pnode, i)) {
        continue;
      }
      TORCH_CHECK(ival.isTensor());
      at::Tensor* tensor = &ival.toTensor();
      managed_output_tensors.emplace_back(0, tensor);
    }
  }
  return managed_output_tensors;
}

} // namespace

MemoryPlanner::MemoryPlanner(
    BlockRunner* block_runner,
    const BlockInfo& block_info,
    bool enable_out_variant,
    bool manage_output_tensors) {
  const auto& managed_tensor_values = block_info.managed_tensor_values();
  const auto& managed_output_tensor_values =
      block_info.managed_output_tensor_values();
  const auto& leaked_values = block_info.leaked_values();

  // collect unmanaged output ivalues
  FastSet<IValue*> unmanaged_ivalues;
  FastSet<IValue*> unmanaged_borrowed_ivalues;
  for (ProcessedNode& pnode : block_runner->nodes()) {
    const auto borrows_outputs = borrowsOutputs(pnode.node()->kind());
    for (const auto i : c10::irange(pnode.outputs().size())) {
      const Value* out_v = pnode.node()->outputs()[i];
      const bool in_managed_tensors = setIncludes(managed_tensor_values, out_v);
      const bool is_unmanaged_special_case = isUnmanagedSpecialCase(pnode, i);
      if (in_managed_tensors && !is_unmanaged_special_case) {
        ++num_managed_tensors_;
      }
      const bool in_managed_sets = in_managed_tensors ||
          // Manage output tensors might have been turned off, so we have to
          // check the flag here
          (manage_output_tensors &&
           setIncludes(managed_output_tensor_values, out_v)) ||
          setIncludes(leaked_values, out_v);

      if (in_managed_sets && !is_unmanaged_special_case) {
        continue;
      }
      if (doesNotHeapAllocateWhenStoredInIValue(*out_v->type())) {
        // Scalars do not need to be freed after each iteration.
        num_unmanaged_scalar_ivalues_++;
      } else if (borrows_outputs) {
        IValue& out = pnode.Output(i);
        unmanaged_borrowed_ivalues.insert(&out);
      } else {
        IValue& out = pnode.Output(i);
        unmanaged_ivalues.insert(&out);
      }
    }
  }
  for (IValue* output : block_runner->outputs()) {
    auto it = unmanaged_borrowed_ivalues.find(output);
    if (it != unmanaged_borrowed_ivalues.end()) {
      borrowed_ivalues_needing_incref_.push_back(output);
      unmanaged_borrowed_ivalues.erase(it);
    } else {
      unmanaged_ivalues.erase(output);
    }
  }

  // copy to unmanaged_ivalues_
  unmanaged_ivalues_.reserve(unmanaged_ivalues.size());
  unmanaged_ivalues_.insert(
      unmanaged_ivalues_.begin(),
      unmanaged_ivalues.begin(),
      unmanaged_ivalues.end());
  unmanaged_borrowed_ivalues_.reserve(unmanaged_borrowed_ivalues.size());
  unmanaged_borrowed_ivalues_.insert(
      unmanaged_borrowed_ivalues_.begin(),
      unmanaged_borrowed_ivalues.begin(),
      unmanaged_borrowed_ivalues.end());

  if (enable_out_variant && manage_output_tensors) {
    managed_output_tensors_ = assignStorageToOutputTensors(
        block_runner, managed_output_tensor_values);
  }
}

uint8_t* MemoryPlanner::allocateBuffer(size_t num_bytes) {
  buffer_ = allocate_buffer(num_bytes);
  uint8_t* start = static_cast<uint8_t*>(buffer_.get());
  buffer_start_ = start;
  buffer_end_ = start + num_bytes;
  return start;
}

void MemoryPlanner::allocateOutputTensors() {
  if (output_buffer_bytes_ == 0) {
    return;
  }
  TORCH_CHECK(
      !output_buffer_,
      "Previously allocated output_buffer_ was not deallocated properly.");
  output_buffer_ = allocate_buffer(output_buffer_bytes_);

  size_t offset = 0;
  uint8_t* start = static_cast<uint8_t*>(output_buffer_.get());

  for (const auto& ms : managed_output_tensors_) {
    auto tensor_size = ms.first;
    auto* tensor = ms.second;
    if (tensor_size == 0) {
      continue;
    }
    DCHECK_LE(offset + tensor_size, output_buffer_bytes_);
    void* src = static_cast<void*>(start + offset);
    // NOTE: Populating `ctx` enables clients to take the ownership of a
    // tensor managed by Static Runtime. Some clients use "move" semantics to
    // pass a Tensor object to another holding object (e.g., a thrift message)
    // to avoid `memcpy`.
    // `torch::distributed::detail::WireDumpOp::dumpTensorData is a concrete
    // example of doing this (See `torch::distributed::detail::hasDeleter`).
    // Since this output Tensor object is permanently owned by Static Runtime,
    // this ownership passing does *not* have an intended effect of keeping the
    // Tensor alive till the "owner" releases it: A premature call to
    // `StaticRuntime::deallocateOutputTensors` can destruct such a Tensor
    // object that a holding object believes to retain, causing it to read
    // corrupted values from an already destructed Tensor object. Therefore, a
    // client of receiving Static Runtime-managed Tensors needs to be very
    // careful to call `StaticRuntime::deallocateOutputTensors` after these
    // holding objects are gone.
    tensor->storage().set_data_ptr_noswap(
        at::DataPtr(src, /*ctx=*/src, nullptr, tensor->device()));
    tensor->storage().set_nbytes(tensor_size);
    offset += tensor_size;
  }
  DCHECK_EQ(offset, output_buffer_bytes_);
}

void MemoryPlanner::allocate() {
  // TODO: Improve this once D31357486 is landed.
  allocateManagedTensors();
  allocateOutputTensors();
}

void MemoryPlanner::deallocate() {
  deallocateManagedTensors();
  for (auto& iv : borrowed_ivalues_needing_incref_) {
    auto old = std::move(*iv);
    *iv = IValue(old);
    c10::MaybeOwnedTraits<c10::IValue>::destroyBorrow(old);
  }
  // for unmanaged ivalues (either tensor or non-tensor), we reset the *iv so
  // that the objects pointed to by *iv may be reclaimed by reference counting
  for (auto& iv : unmanaged_ivalues_) {
    *iv = IValue();
  }
  for (auto& iv : unmanaged_borrowed_ivalues_) {
    c10::MaybeOwnedTraits<c10::IValue>::destroyBorrow(*iv);
  }
  buffer_ = {};
}

void MemoryPlanner::deallocateOutputTensors() {
  size_t output_buffer_bytes = 0;
  for (auto& ms : managed_output_tensors_) {
    auto* tensor = ms.second;
    size_t current_size =
        compute_aligned_tensor_size(tensor->storage().nbytes());
    tensor->storage().unsafeGetStorageImpl()->reset();
    if (current_size > ms.first) {
      ms.first = current_size;
    }
    output_buffer_bytes += ms.first;
  }
  output_buffer_bytes_ = output_buffer_bytes;
  output_buffer_ = {};
}

StandardMemoryPlanner::StandardMemoryPlanner(
    BlockRunner* block_runner,
    const BlockInfo& block_info,
    bool enable_out_variant,
    bool manage_output_tensors,
    bool optimize_memory)
    : MemoryPlanner(
          block_runner,
          block_info,
          enable_out_variant,
          manage_output_tensors) {
  const auto& managed_tensor_values = block_info.managed_tensor_values();
  if (enable_out_variant) {
    const auto tensor_value_to_tensor =
        tensorValueToTensor(block_runner->nodes(), managed_tensor_values);
    if (optimize_memory) {
      managed_tensors_ = assignStorageToManagedTensors(
          block_info.node_ptrs(),
          block_info.managed_tensor_ranges(),
          tensor_value_to_tensor);
    } else {
      for (auto& tensor : tensor_value_to_tensor) {
        managed_tensors_.emplace_back(tensor.second);
      }
    }
  }
}

void StandardMemoryPlanner::allocateManagedTensors() {
  if (managed_bytes_ == 0) {
    return;
  }
  DCHECK(!managed_tensor_storage_impls_.empty());
  size_t offset = 0;
  auto* start = allocateBuffer(managed_bytes_);

  reused_tensors_ = 0;
  auto group_idx = 0;
  for (auto& ms : managed_tensor_storage_impls_) {
    auto tensor_size = ms.first;
    if (tensor_size == 0) {
      group_idx++;
      continue;
    }
    at::StorageImpl* storageImpl = &ms.second;
    DCHECK_LE(offset + tensor_size, managed_bytes_);
    void* src = static_cast<void*>(start + offset);

#ifndef NDEBUG
    DCHECK_EQ(tensor_size, managed_tensors_[group_idx].maxTensorSize());
    for (auto* tensor : managed_tensors_[group_idx].group()) {
      DCHECK_EQ(storageImpl, tensor->storage().unsafeGetStorageImpl());
    }
#endif
    DCHECK_NE(managed_tensors_[group_idx].numManagedTensors(), 0);
    reused_tensors_ += managed_tensors_[group_idx].numManagedTensors() - 1;
    storageImpl->set_data_ptr_noswap(
        at::DataPtr(src, src, nullptr, c10::Device(c10::DeviceType::CPU)));
    storageImpl->set_nbytes(tensor_size);

    offset += tensor_size;
    group_idx++;
  }
  DCHECK_EQ(offset, managed_bytes_);
}

void StandardMemoryPlanner::deallocateManagedTensors() {
  managed_bytes_ = 0;
  // free memory used by outputs of ops in out variants
  // but keep the TensorImpl and StorageImpl around.

  // We don't have any guarantee that the model doesn't change the
  // Storage for managed tensors out from under us during execution,
  // so we have to check the Storages each time we deallocate.
  auto group_idx = 0;
  const bool first_time = managed_tensor_storage_impls_.empty();
  if (C10_UNLIKELY(first_time)) {
    managed_tensor_storage_impls_.reserve(managed_tensors_.size());
  }
  for (auto& ms : managed_tensors_) {
    const auto& tensors = ms.group();
    size_t max = ms.maxTensorSize();
    auto tensor_idx = 0;
    for (auto& tensor : tensors) {
      const auto& storage = tensor->storage();
      size_t current_size = compute_aligned_tensor_size(storage.nbytes());
      at::StorageImpl* tensorStorageImpl = storage.unsafeGetStorageImpl();
      if (C10_UNLIKELY(first_time)) {
        tensorStorageImpl->reset();

        DCHECK(
            managed_tensor_storage_impls_.size() == group_idx ||
            managed_tensor_storage_impls_.size() == group_idx + 1);
        if (managed_tensor_storage_impls_.size() == group_idx) {
          managed_tensor_storage_impls_.emplace_back(
              0, // will be set at end of outer loop
              std::move(*tensorStorageImpl));
        }
        at::StorageImpl* newImpl = &managed_tensor_storage_impls_.back().second;

        // We want to manage StorageImpls' lifetimes ourselves, but TensorImpl
        // expects to refcount them. unsafe_adapt_non_heap_allocated is our
        // escape hatch: it sets the reference count for the StorageImpl to an
        // impractically high value so that it will never get deallocated by
        // intrusive_ptr, leaving us free to manage its lifetime as we see fit.
        // (Note that allowing it to be deallocated by intrusive_ptr would be
        // UB, because that would entail deleting an object that wasn't
        // allocated with operator new.)
        //
        // For more information, see the doc comment for
        // intrusive_ptr::unsafe_adapt_non_heap_allocated.
        tensor->unsafeGetTensorImpl()->set_storage_keep_dtype(at::Storage(
            c10::intrusive_ptr<at::StorageImpl>::
                unsafe_adapt_non_heap_allocated(newImpl, tensors.size())));
      } else if (C10_UNLIKELY(
                     tensorStorageImpl !=
                     &managed_tensor_storage_impls_[group_idx].second)) {
        tensorStorageImpl->reset();

        // If somehow the tensor got different storage, put it back to
        // the shared impl for this group.
        tensor->unsafeGetTensorImpl()->set_storage_keep_dtype(at::Storage(
            c10::intrusive_ptr<at::StorageImpl>::
                unsafe_adapt_non_heap_allocated(
                    &managed_tensor_storage_impls_[group_idx].second,
                    tensors.size())));
      }
      DCHECK_EQ(
          tensor->storage().unsafeGetStorageImpl(),
          &managed_tensor_storage_impls_[group_idx].second);
      max = std::max(max, current_size);
    }
    // Static runtime does not know the size of tensors statically, so we use
    // the tensor size from the previous run to allocate tensors for the next
    // run (following C2 tradition), exploiting the fact that tensor storage
    // size does not have to match that of real tensor size. The following logic
    // records the tensor storage size for the next run.
    managed_tensor_storage_impls_[group_idx++].first = max;
    ms.setMaxTensorSize(max);
    managed_bytes_ += max;
  }

  DCHECK_EQ(managed_tensor_storage_impls_.size(), managed_tensors_.size());
  VLOG(1) << "managed_bytes: " << managed_bytes_;
}

PrecomputedOffsetsMemoryPlanner::PrecomputedOffsetsMemoryPlanner(
    BlockRunner* block_runner,
    const BlockInfo& block_info,
    bool enable_out_variant,
    bool manage_output_tensors,
    bool optimize_memory)
    : MemoryPlanner(
          block_runner,
          block_info,
          enable_out_variant,
          manage_output_tensors),
      block_runner_(block_runner),
      block_info_(block_info),
      enable_out_variant_(enable_out_variant),
      manage_output_tensors_(manage_output_tensors),
      optimize_memory_(optimize_memory) {}

std::unique_ptr<MemoryPlanner> PrecomputedOffsetsMemoryPlanner::maybe_clone(
    BlockRunner* new_block_runner,
    const FastMap<at::Tensor*, at::Tensor*>& old_tensor_to_new) const {
  auto result = std::make_unique<PrecomputedOffsetsMemoryPlanner>(
      new_block_runner,
      block_info_,
      enable_out_variant_,
      manage_output_tensors_,
      optimize_memory_);
  result->managed_tensor_storage_impls_.reserve(
      managed_tensor_storage_impls_.size());
  result->managed_tensors_.reserve(managed_tensors_.size());
  for (const auto& managed_tensor : managed_tensors_) {
    auto* old_tensor = managed_tensor.tensor;
    auto* new_tensor = old_tensor_to_new.at(old_tensor);
    auto* value = managed_tensor.value;
    const auto& storage = new_tensor->storage();
    size_t current_size = compute_aligned_tensor_size(storage.nbytes());
    auto* storage_impl = storage.unsafeGetStorageImpl();

    storage_impl->reset();
    result->managed_tensor_storage_impls_.emplace_back(
        current_size, std::move(*storage_impl));
    auto* new_impl = &result->managed_tensor_storage_impls_.back().second;
    result->managed_tensors_.emplace_back(
        new_tensor, value, new_impl, current_size, managed_tensor.offset);
    new_tensor->unsafeGetTensorImpl()->set_storage_keep_dtype(at::Storage(
        c10::intrusive_ptr<at::StorageImpl>::unsafe_adapt_non_heap_allocated(
            new_impl, 1)));
  }
  result->managed_bytes_ = managed_bytes_;
  return result;
}

void PrecomputedOffsetsMemoryPlanner::allocateManagedTensors() {
  if (managed_bytes_ == 0) {
    return;
  }
  uint8_t* start = allocateBuffer(managed_bytes_);

  for (auto& managed_tensor : managed_tensors_) {
    auto tensor_size = managed_tensor.size;
    if (tensor_size == 0) {
      continue;
    }
    auto offset = managed_tensor.offset;
    auto* storage_impl = managed_tensor.storage_impl;

    DCHECK_LE(offset + tensor_size, managed_bytes_);
    void* src = static_cast<void*>(start + offset);
    storage_impl->set_data_ptr_noswap(
        at::DataPtr(src, src, nullptr, c10::Device(c10::DeviceType::CPU)));
    storage_impl->set_nbytes(tensor_size);
  }
}

void PrecomputedOffsetsMemoryPlanner::deallocateManagedTensors() {
  const bool first_time = managed_tensor_storage_impls_.empty();
  if (C10_UNLIKELY(first_time)) {
    auto managed_tensors_and_values = collectManagedTensors(
        block_runner_->nodes(), block_info_.managed_tensor_values());
    managed_tensor_storage_impls_.reserve(managed_tensors_and_values.size());
    for (auto& tensor_and_value : managed_tensors_and_values) {
      auto* tensor = tensor_and_value.first;
      auto* value = tensor_and_value.second;
      const auto& storage = tensor->storage();
      size_t current_size = compute_aligned_tensor_size(storage.nbytes());
      auto* storage_impl = storage.unsafeGetStorageImpl();

      storage_impl->reset();
      managed_tensor_storage_impls_.emplace_back(
          current_size, std::move(*storage_impl));
      auto* new_impl = &managed_tensor_storage_impls_.back().second;
      managed_tensors_.emplace_back(
          tensor,
          value,
          new_impl,
          current_size,
          0); // compute the offsets after collecting all tensors

      tensor->unsafeGetTensorImpl()->set_storage_keep_dtype(at::Storage(
          c10::intrusive_ptr<at::StorageImpl>::unsafe_adapt_non_heap_allocated(
              new_impl, 1)));
    }
    if (optimize_memory_) {
      managed_bytes_ = assignOffsetsOptimized(
          managed_tensors_, block_info_.managed_tensor_ranges());
    } else {
      managed_bytes_ = assignOffsetsNaive(managed_tensors_);
    }
    VLOG(1) << "managed_bytes: " << managed_bytes_;
  } else {
    for (auto& managed_tensor : managed_tensors_) {
      auto* tensor = managed_tensor.tensor;
      if (tensor->nbytes() > managed_tensor.size) {
        // When we allocate this tensor on the next run, we'll need to set
        // the StorageImpl's nbytes to managed_tensor.size. We can't let
        // tensor->nbytes() be greater than the size of its storage impl,
        // so shrink it if it got bigger.
        at::native::resize_(*tensor, {0});
      }
      auto* expected_storage_impl = managed_tensor.storage_impl;
      auto* actual_storage_impl = tensor->storage().unsafeGetStorageImpl();
      if (C10_UNLIKELY(actual_storage_impl != expected_storage_impl)) {
        tensor->storage().unsafeGetStorageImpl()->reset();
        tensor->unsafeGetTensorImpl()->set_storage_keep_dtype(at::Storage(
            c10::intrusive_ptr<at::StorageImpl>::
                unsafe_adapt_non_heap_allocated(expected_storage_impl, 1)));
      }
    }
  }
}

} // namespace jit
} // namespace torch
