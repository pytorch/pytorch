#include <torch/csrc/jit/runtime/static/memory_planner.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

static void assign_storage_to_managed_tensors(
    StaticRuntime* runtime,
    const FastSet<const Value*>& managed_tensor_values,
    const FastMap<const Value*, std::vector<const Value*>>&
        value_to_same_storage_values,
    std::vector<StorageGroup>& managed_tensors) {
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
          managed_tensors[storage_idx].addTensor(tensor);
        } else {
          managed_tensors.emplace_back(tensor);
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

static bool setIncludes(const FastSet<const Value*>& set, const Value* v) {
  return set.find(v) != set.end();
}

static void assignStorageToOutputTensors(
    StaticRuntime* runtime,
    const FastSet<const Value*>& managed_output_tensor_values,
    std::vector<std::pair<size_t, at::Tensor*>>* managed_output_tensors) {
  for (auto& pnode : runtime->nodes()) {
    for (const auto i : c10::irange(pnode.outputs().size())) {
      auto& ival = pnode.Output(i);
      const auto* val = pnode.node()->outputs()[i];
      if (!setIncludes(managed_output_tensor_values, val)) {
        continue;
      }
      TORCH_CHECK(ival.isTensor());
      at::Tensor* tensor = &ival.toTensor();
      managed_output_tensors->emplace_back(0, tensor);
    }
  }
}

MemoryPlanner::MemoryPlanner(
    StaticRuntime* runtime,
    const FastMap<const Value*, std::vector<const Value*>>&
        value_to_same_storage_values,
    const ValueGroup& value_group,
    bool enable_out_variant,
    bool manage_output_tensors) {
  // collect register indices of outputs of ops with out variant
  FastSet<const Value*> managed_tensor_values;
  FastSet<const Value*> leaked_values;
  // Never manage graph outputs so that we can do std::move(output_ivalue).
  // This does not affect performance if the graph returns a collection object.
  FastSet<const Value*> graph_output_values(
      runtime->graph().outputs().begin(), runtime->graph().outputs().end());
  if (enable_out_variant) {
    for (ProcessedNode& pnode : runtime->nodes()) {
      if (!pnode.has_out_variant()) {
        continue;
      }
      for (const auto i : c10::irange(pnode.outputs().size())) {
        const Value* out_v = pnode.node()->outputs()[i];
        // Types are stored in the underlying TorchScript IR
        bool is_tensor_type = out_v->type()->castRaw<TensorType>();
        if (manage_output_tensors && is_tensor_type &&
            !setIncludes(graph_output_values, out_v) &&
            value_group.isOutputAlias(out_v)) {
          managed_output_tensor_values_.insert(out_v);
          continue;
        }
        if (value_group.isAlwaysAlive(out_v)) {
          continue;
        }
        if (is_tensor_type) {
          managed_tensor_values.insert(out_v);
        } else if (runtime->is_optimizable_container_type(pnode.node())) {
          // We "leak" certain container types because their allocations
          // take a long time
          leaked_values.insert(out_v);
        }
      }
    }
  }

  // collect unmanaged output ivalues
  FastSet<IValue*> unmanaged_ivalues;
  FastSet<IValue*> unmanaged_borrowed_ivalues;
  for (ProcessedNode& pnode : runtime->nodes()) {
    for (const auto i : c10::irange(pnode.outputs().size())) {
      // Types are stored in the underlying TorchScript IR
      const Value* out_v = pnode.node()->outputs()[i];
      if (setIncludes(managed_tensor_values, out_v) ||
          setIncludes(managed_output_tensor_values_, out_v) ||
          setIncludes(leaked_values, out_v)) {
        continue;
      }
      static const std::array<c10::Symbol, 2> symbols_with_borrowed_outputs = {
          c10::Symbol::fromQualString("static_runtime::dict_unpack"),
          c10::Symbol::fromQualString("static_runtime::VarTupleUnpack"),
      };
      if (doesNotHeapAllocateWhenStoredInIValue(*out_v->type())) {
        // Scalars do not need to be freed after each iteration.
        num_unmanaged_scalar_ivalues_++;
      } else if (
          std::find(
              symbols_with_borrowed_outputs.begin(),
              symbols_with_borrowed_outputs.end(),
              pnode.node()->kind()) != symbols_with_borrowed_outputs.end()) {
        IValue& out = pnode.Output(i);
        unmanaged_borrowed_ivalues.insert(&out);
      } else {
        IValue& out = pnode.Output(i);
        unmanaged_ivalues.insert(&out);
      }
    }
  }
  // since runtime->outputs() escape from run(), remove them from
  // managed_tensor_values and from unmanaged_ivalues
  for (const Value* output : runtime->graph().outputs()) {
    managed_tensor_values.erase(output);
  }
  FastSet<IValue*> borrowed_ivalues_needing_incref;
  for (IValue* output : runtime->outputs()) {
    auto it = unmanaged_borrowed_ivalues.find(output);
    if (it != unmanaged_borrowed_ivalues.end()) {
      borrowed_ivalues_needing_incref_.push_back(output);
      unmanaged_borrowed_ivalues.erase(it);
    } else {
      unmanaged_ivalues.erase(output);
    }
  }

  GRAPH_DEBUG("managed_tensor_values: ", dumpValueSet(managed_tensor_values));
  GRAPH_DEBUG(
      "managed_output_tensor_values_: ",
      dumpValueSet(managed_output_tensor_values_));

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

  if (enable_out_variant) {
    ::torch::jit::assign_storage_to_managed_tensors(
        runtime,
        managed_tensor_values,
        value_to_same_storage_values,
        managed_tensors_);
  }

  if (enable_out_variant && manage_output_tensors) {
    ::torch::jit::assignStorageToOutputTensors(
        runtime, managed_output_tensor_values_, &managed_output_tensors_);
  }

  num_managed_tensors_ = 0;
  for (const auto& ms : managed_tensors_) {
    num_managed_tensors_ += ms.numManagedTensors();
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

void MemoryPlanner::allocateManagedTensors() {
  if (managed_bytes_ == 0) {
    return;
  }
  DCHECK(!managed_tensor_storage_impls_.empty());
  buffer_ = allocate_buffer(managed_bytes_);

  size_t offset = 0;
  uint8_t* start = static_cast<uint8_t*>(buffer_.get());
  buffer_start_ = start;
  buffer_end_ = start + managed_bytes_;

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

} // namespace jit
} // namespace torch
