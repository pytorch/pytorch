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
    // NOLINTNEXTLINE(bugprone-pointer-arithmetic-on-polymorphic-object)
    auto& storage = storage_buf[i];

    // if the existing data ptr doesn't have an associated deleter then we
    // will set the offset and size directly, as opposed to creating and
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
    at::TensorImpl& tensor_impl = *tensor->unsafeGetTensorImpl();

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
      tensor_impl.set_storage_keep_dtype(at::Storage(
          c10::intrusive_ptr<at::StorageImpl>::unsafe_adapt_non_heap_allocated(
              &storage_impl_buffer_.to_managed(storage), 1)));
    } else if (
        C10_UNLIKELY(
            &storage !=
            // NOLINTNEXTLINE(bugprone-pointer-arithmetic-on-polymorphic-object)
            &storage_buf
                [i]) /* managed storage was replaced for some reason */) {
      storage.reset();
      tensor_impl.set_storage_keep_dtype(at::Storage(
          c10::intrusive_ptr<at::StorageImpl>::unsafe_adapt_non_heap_allocated(
              // NOLINTNEXTLINE(bugprone-pointer-arithmetic-on-polymorphic-object)
              &storage_buf[i],
              1)));
    }

    // resize to zero so that we ensure that we don't access out-of-bounds
    // addr's in the next iteration
    tensor_impl.set_sizes_contiguous({0});
  }
}

void LayoutManager::populate_tensor_values() {
  TORCH_CHECK(planned_tensors_.empty());
  TORCH_CHECK(unplanned_ivalues_.empty());

  const auto& value_ids = planner_.get_planned_values();
  planned_tensors_.resize(value_ids.size());
  planned_tensors_max_nbytes_local_.resize(value_ids.size());

  for (const auto&& [i, v] : c10::enumerate(value_ids)) {
#ifndef NDEBUG
    value_to_vector_idx_map_[v] = i;
#endif
    planned_tensors_[i] = &parent_frame_.getIValue(v).toTensor();
  }

  const auto& unplanned_value_ids = planner_.get_unplanned_values();
  unplanned_ivalues_.resize(unplanned_value_ids.size());
  for (const auto&& [i, v] : c10::enumerate(unplanned_value_ids)) {
    unplanned_ivalues_[i] = &parent_frame_.getIValue(v);
  }
}

#ifndef NDEBUG
void LayoutManager::assert_no_overlapping_storages(
    size_t graph_node_idx) const {
  if (state_ != LayoutManagerState::Running) {
    return;
  }

  /*
    for each value
    (either an input or output)
    ensure that the associated storage
    slice lies within the allocated slice
    if it is managed (or if it is an alias,
    we can use the slice allocated to its source)
    ---
    also ensure that the current index lies
    within the lifetime of this value
  */

  const auto& alias_analyzer = planner_.get_alias_analyzer();
  // get the 'active' values during the execution of nodes[graph_node_idx]
  const auto& alive_values =
      alias_analyzer.alive_values_at_time(graph_node_idx);

  // make sure active memory intervals are non-overlapping
  // by sorting them by start, and ensuring
  // cur.start > prev.end for each
  //
  // by default, the pairs are compared lexicographically.
  // ref: https://cplusplus.com/reference/utility/pair/operators/
  //
  // in our case, this means that leftmost (on the number line) intervals will
  // come first, and if the start point of two intervals is the same, they will
  // be sorted by their relative widths (in increasing order)
  //
  // e.g., the ordering for the following usage intervals
  //
  // |######1######|
  //        |######2######|
  //        |######3#####|
  //
  // would be [1,3,2]

  std::multiset<std::pair<size_t, size_t>> intervals;

  planner_.with_plan([&](const LayoutPlan& plan) {
    // prevent recomputation from occurring
    c10::FastSet<ValueId> checked_values;

    // check that some arbitrary storage (defined by the allocation start and
    // the size in bytes) lies within the slice allocated for value_id during
    // planning.
    //
    // if the checks pass, add the interval [alloc_start, alloc_start +
    // alloc_nbytes) to the set of intervals
    auto check_allocation_bounds =
        [&](ValueId value_id, size_t alloc_start, size_t alloc_end) -> void {
      if (!checked_values.emplace(value_id).second /* already checked */) {
        return;
      }
      auto& alloc = plan.allocations[value_to_vector_idx_map_.at(value_id)];
      TORCH_CHECK(alloc_start >= alloc.offset);
      TORCH_CHECK(alloc_end < alloc.offset + alloc.size);
      intervals.emplace(alloc_start, alloc_end);
    };

    // get the inclusive storage interval for some value (i.e.,
    // [buffer_storage_start_offset, buffer_storage_start_offset +
    // storage_nbytes]) that represents the sub-slice of the runtime-managed
    // buffer allocated to this tensor
    auto try_get_interval =
        [&](ValueId value_id) -> std::optional<std::pair<size_t, size_t>> {
      const auto& iv = parent_frame_.getIValue(value_id);
      if (!iv.isTensor()) {
        return std::nullopt;
      }

      const auto& storage_impl = iv.toTensor().storage().unsafeGetStorageImpl();
      const auto storage_nbytes = storage_impl->nbytes();

      if (const auto start = layout_buffer_.get_offset_from_ptr(
              storage_impl->data_ptr().get());
          start.has_value()) {
        return std::make_pair(*start, *start + storage_nbytes - 1);
      }

      return std::nullopt;
    };

    for (auto v : alive_values) {
      // sanity check lifetimes to ensure this
      // value ~should~ be alive at this point
      const auto& lt = alias_analyzer.lifetime(v);
      TORCH_CHECK(graph_node_idx >= lt.start);
      TORCH_CHECK(graph_node_idx <= lt.end);

      const auto interval = try_get_interval(v->id());
      if (C10_UNLIKELY(!interval.has_value())) {
        continue;
      }

      auto& [v_start, v_end] = *interval;

      // it's possible that v is an alias, in which case
      // we want to try to get the source (i.e., the value)
      // that actually owns the storage
      //
      // NOTE: it's possible the source is ambiguous, hence
      // why get_sources_of_alias returns a set (although it's usually a
      // singleton set)
      if (const auto* srcs_of_v = alias_analyzer.get_sources_of_alias(v);
          srcs_of_v != nullptr /* v is an alias */) {
        // 1. v's interval is a sub-interval of ~a~ source's interval and we
        //    want to add the source's interval to the set of intervals
        // 2. v possibly got re-alloc'd / is not actually aliasing anything
        //    and we want to add v's interval to the set of intervals
        bool found_viable_source = false;

        for (const auto* src_of_v : *srcs_of_v) {
          const auto src_interval = try_get_interval(src_of_v->id());
          if (C10_UNLIKELY(!src_interval.has_value())) {
            continue;
          }

          auto& [src_of_v_start, src_of_v_end] = *src_interval;

          if (v_start >= src_of_v_start && v_end <= src_of_v_end) {
            check_allocation_bounds(
                src_of_v->id(), src_of_v_start, src_of_v_end);
            found_viable_source = true;
            break;
          }
        }

        if (!found_viable_source) {
          check_allocation_bounds(v->id(), v_start, v_end);
        }
      } else /* if v isn't an alias */ {
        check_allocation_bounds(v->id(), v_start, v_end);
      }
    }
  });

  // if we only have less than two active intervals,
  // it isn't possible to have overlap...
  if (intervals.size() < 2) {
    return;
  }

  // ensure that no 'active' buffer intervals are overlapping
  auto it = intervals.begin();
  size_t prev_end = it->second;
  while (++it != intervals.end()) {
    TORCH_CHECK(prev_end < it->first /* cur_start */);
    prev_end = it->second;
  }
}
#endif

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
