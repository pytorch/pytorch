#include <torch/nativert/executor/memory/LayoutManager.h>

#include <torch/nativert/executor/ExecutionFrame.h>

#include <c10/core/CPUAllocator.h>
#include <c10/util/Enumerate.h>

namespace torch::nativert {

LayoutManager::LayoutManager(
    LayoutPlanner& planner,
    ExecutionFrame& parent_frame,
    const torch::nativert::LayoutManagerSettings settings)
    : planner_(planner), parent_frame_(parent_frame), settings_(settings) {
  VLOG(1) << "layout manager created for execution frame";
}

void ContiguousLayoutBuffer::allocate(size_t size) {
  VLOG(1) << "allocating " << size << " bytes";
  if (C10_LIKELY(size_ > 0)) {
    if (C10_LIKELY(
            size <= size_) /* NOTE: size will be monotonically increasing */) {
      return clear(size_);
    } else {
      deallocate();
    }
  }
  data_ptr_ = c10::GetCPUCachingAllocator()->allocate(size);
  size_ = size;
}

void LayoutManager::allocate() {
  if (C10_UNLIKELY(state_ == LayoutManagerState::WaitingForValues)) {
    return;
  }

  bool should_allocate_storages =
      state_ == LayoutManagerState::AllocatingStorages;

  ensure_managed_storages(/* allocate= */ should_allocate_storages);

  planner_.with_plan([&](const auto& plan) { allocate_plan(plan); });

  if (should_allocate_storages) {
    state_ = LayoutManagerState::Running;
  }
}

void LayoutManager::allocate_plan(const LayoutPlan& plan) {
  if (C10_UNLIKELY(storage_impl_buffer_.size() == 0 || plan.total_size == 0)) {
    return;
  }

  layout_buffer_.allocate(plan.total_size);
  VLOG(1) << "allocated " << layout_buffer_.size()
          << " bytes for planned layout";

  auto* storage_buf = storage_impl_buffer_.buffer();

  for (const auto i : c10::irange(plan.allocations.size())) {
    auto& planned_allocation = plan.allocations[i];
    auto& local_max_nbytes = planned_tensors_max_nbytes_local_[i];
    local_max_nbytes = std::max(local_max_nbytes, planned_allocation.size);

    void* offset_ptr =
        layout_buffer_.get_ptr_with_offset(planned_allocation.offset);
    auto& storage = storage_buf[i];

    // if the existing data ptr doesn't have an associated deleter then we
    // will set the offset and size directly, as oposed to creating and
    // swapping it with a new one
    //
    // apart from the first allocation when the storage still has the its
    // allocator-created dataptr (https://fburl.com/code/u7dsspjm) whose
    // deleter is non-null (https://fburl.com/code/7hiwo5zo), this should
    // always be true
    if (C10_LIKELY(
            storage._mutable_data_ptr_no_checks().unsafe_reset_data_and_ctx(
                offset_ptr))) {
      storage.unsafe_set_nbytes(planned_allocation.size);
    } else {
      storage.set_data_ptr_noswap(at::DataPtr(
          offset_ptr, offset_ptr, nullptr, c10::Device(c10::DeviceType::CPU)));
      storage.set_nbytes(planned_allocation.size);
    }
  }
}

void LayoutManager::ensure_managed_storages(bool allocate) {
  if (C10_UNLIKELY(planned_tensors_.empty())) {
    return;
  }

  if (C10_UNLIKELY(allocate)) {
    storage_impl_buffer_.allocate(planned_tensors_.size());
    VLOG(1) << "allocated " << planned_tensors_.size() * sizeof(at::StorageImpl)
            << " bytes for contiguous storages";
  }

  auto* storage_buf = storage_impl_buffer_.buffer();

  for (size_t i = 0; i < planned_tensors_.size(); i += 1) {
    auto* tensor = planned_tensors_[i];

    at::StorageImpl& storage = *tensor->storage().unsafeGetStorageImpl();

    if (C10_UNLIKELY(allocate)) {
      // from: https://fburl.com/code/4it00yph
      //
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
          c10::intrusive_ptr<at::StorageImpl>::unsafe_adapt_non_heap_allocated(
              &storage_impl_buffer_.to_managed(storage), 1)));
    } else if (
        C10_UNLIKELY(
            &storage !=
            &storage_buf
                [i]) /* managed storage was replaced for some reason */) {
      storage.reset();
      tensor->unsafeGetTensorImpl()->set_storage_keep_dtype(at::Storage(
          c10::intrusive_ptr<at::StorageImpl>::unsafe_adapt_non_heap_allocated(
              &storage_buf[i], 1)));
    }
  }
}

void LayoutManager::populate_tensor_values() {
  CHECK(planned_tensors_.empty());
  CHECK(unplanned_ivalues_.empty());

  const auto& value_ids = planner_.get_planned_values();
  planned_tensors_.resize(value_ids.size());
  planned_tensors_max_nbytes_local_.resize(value_ids.size());

  for (const auto&& [i, v] : c10::enumerate(value_ids)) {
    planned_tensors_[i] = &parent_frame_.getIValue(v).toTensor();
  }

  const auto& unplanned_value_ids = planner_.get_unplanned_values();
  unplanned_ivalues_.resize(unplanned_value_ids.size());
  for (const auto&& [i, v] : c10::enumerate(unplanned_value_ids)) {
    unplanned_ivalues_[i] = &parent_frame_.getIValue(v);
  }
}

void LayoutManager::try_update_historical_max_nbytes() {
  for (const auto i : c10::irange(planned_tensors_.size())) {
    auto nbytes = get_aligned_nbytes(planned_tensors_[i]->nbytes());
    if (auto& old_max = planned_tensors_max_nbytes_local_[i];
        nbytes > old_max) {
      old_max = nbytes;
      planner_.try_update_max_size_at_index(i, nbytes);
    }
  }
}

void LayoutManager::deallocate_and_plan() {
  const auto uninitialized = state_ == LayoutManagerState::WaitingForValues;

  if (C10_UNLIKELY(uninitialized)) {
    populate_tensor_values();
  }

  try_update_historical_max_nbytes();

  if (C10_UNLIKELY(uninitialized)) {
    planner_.start_worker_if_not_started();
  }

  if (C10_UNLIKELY(uninitialized)) {
    state_ = LayoutManagerState::AllocatingStorages;
  } else if (settings_.deallocateBetweenRequests()) {
    layout_buffer_.deallocate();
  }

  for (auto* ivalue : unplanned_ivalues_) {
    *ivalue = c10::IValue();
  }
}

} // namespace torch::nativert
