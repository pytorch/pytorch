#pragma once

#include <torch/nativert/executor/memory/LayoutPlanner.h>
#include <torch/nativert/executor/memory/LayoutPlannerAlgorithm.h>
#include <torch/nativert/executor/memory/LayoutPlannerSettings.h>

#include <c10/core/alignment.h>
#include <c10/core/impl/alloc_cpu.h>

namespace torch::nativert {

class ExecutionFrame;

struct ContiguousLayoutBuffer {
 public:
  ContiguousLayoutBuffer() = default;
  ~ContiguousLayoutBuffer() {
    deallocate();
  }

  ContiguousLayoutBuffer(ContiguousLayoutBuffer&& other) = delete;
  ContiguousLayoutBuffer(const ContiguousLayoutBuffer& other) = delete;
  ContiguousLayoutBuffer operator=(ContiguousLayoutBuffer&& other) = delete;
  ContiguousLayoutBuffer& operator=(const ContiguousLayoutBuffer& other) =
      delete;

  std::optional<size_t> get_offset_from_ptr(void* offset_ptr) const {
    void* raw_ptr = data_ptr_.get();
    if (!raw_ptr || !offset_ptr) {
      return std::nullopt;
    }

    auto offset = reinterpret_cast<uint8_t*>(offset_ptr) -
        reinterpret_cast<uint8_t*>(raw_ptr);

    return offset < 0 || static_cast<size_t>(offset) >= size_
        ? std::nullopt
        : std::optional(offset);
  }

  void* get_ptr_with_offset(size_t offset) {
    void* raw_ptr = data_ptr_.get();
    TORCH_CHECK(raw_ptr != nullptr);
    TORCH_CHECK(offset <= size_);
    return reinterpret_cast<void*>(
        reinterpret_cast<uint8_t*>(raw_ptr) + offset);
  }

  size_t size() {
    return size_;
  }

  void allocate(size_t size);

  void deallocate() {
    VLOG(1) << "deallocating layout buffer of size " << size_;
    size_ = 0;
    data_ptr_ = {};
  }

  void clear(size_t size) {
    VLOG(1) << "clearing first " << size << "bytes of layout buffer of size "
            << size_;
    TORCH_CHECK(size <= size_);
    std::memset(data_ptr_.get(), 0, size);
  }

 private:
  // the size of the buffer in bytes
  size_t size_{0};

  // the dataptr returned by the allocator
  at::DataPtr data_ptr_;
};

struct ContiguousStorageImplBuffer {
  ContiguousStorageImplBuffer() = default;
  ~ContiguousStorageImplBuffer() {
    deallocate();
  }

  ContiguousStorageImplBuffer(ContiguousStorageImplBuffer&& other) = delete;
  ContiguousStorageImplBuffer(const ContiguousStorageImplBuffer& other) =
      delete;
  ContiguousStorageImplBuffer operator=(ContiguousStorageImplBuffer&& other) =
      delete;
  ContiguousStorageImplBuffer& operator=(
      const ContiguousStorageImplBuffer& other) = delete;

  void deallocate() {
    if (buffer_ == nullptr) {
      return;
    }

    for (const size_t idx : c10::irange(size_)) {
      buffer_[idx].~StorageImpl();
    }

    delete[] reinterpret_cast<unsigned char*>(buffer_);
    buffer_ = nullptr;
    size_ = capacity_ = 0;
  }

  void allocate(size_t capacity) {
    if (size_ > 0) {
      deallocate();
    }

    capacity_ = capacity;

    static_assert(alignof(at::StorageImpl) <= 8);
    buffer_ = reinterpret_cast<at::StorageImpl*>(
        new unsigned char[capacity * sizeof(at::StorageImpl)]);
  }

  size_t capacity() {
    return capacity_;
  }

  size_t size() {
    return size_;
  }

  c10::StorageImpl* buffer() const {
    return buffer_;
  }

  c10::StorageImpl& at(size_t i) {
    TORCH_CHECK(
        i < size_, "requested storage index ", i, " out of bounds ", size_);
    return buffer_[i];
  }

  void reset_all() {
    for (const size_t idx : c10::irange(size_)) {
      buffer_[idx].reset();
    }
  }

  c10::StorageImpl& to_managed(at::StorageImpl& s) {
    TORCH_CHECK(size_ < capacity_);
    return *(new (&buffer_[size_++]) at::StorageImpl(
        at::StorageImpl::use_byte_size_t(),
        static_cast<int64_t>(s.nbytes()),
        s.allocator(),
        s.resizable()));
  }

 private:
  size_t size_{0};
  size_t capacity_{0};
  c10::StorageImpl* buffer_{nullptr};
};

enum class LayoutManagerState { WaitingForValues, AllocatingStorages, Running };

class LayoutManager {
 public:
  LayoutManager(
      LayoutPlanner& planner,
      ExecutionFrame& parent_frame,
      torch::nativert::LayoutManagerSettings settings = {});
  ~LayoutManager() = default;

// this is a debugging function. it will slow thing down SIGNIFICANTLY
// so please ensure this isn't called unless you really need it
//
// it checks a few things in between node executions...
//
// 1. ensures all 'alive' values are within the bounds of their lifetimes
//    - this is the definition of a sanity check since the live-sets are built
//      from the lifetimes lol. if this fails, something is very very wrong
// 2. ensures that all planned values are within the bounds of their
//    allocated storage buffer slices
//      - if the value is an alias, ensure the alias is within the bounds
//        of the source value
// 3. ensures that all planned value data-ptrs are non-overlapping
#ifndef NDEBUG
  void assert_no_overlapping_storages(
      size_t
          graph_node_idx /* the graph node that is currently being computed */)
      const;
#endif

 private:
  friend class LayoutManagerGuard;

  void allocate();
  void deallocate_and_plan();

#ifdef LayoutPlannerTests_TEST_FRIENDS
  LayoutPlannerTests_TEST_FRIENDS;
#endif

  static size_t get_aligned_nbytes(size_t nbytes) {
#if defined(__linux__) && !defined(__ANDROID__)
    auto alignment = c10::c10_compute_alignment(nbytes);
#else
    auto alignment = c10::gAlignment;
#endif
    return (nbytes + alignment - 1) & (~(alignment - 1));
  }

  void allocate_plan(const LayoutPlan& plan);
  void ensure_managed_storages(bool allocate);

  void populate_tensor_values();
  void try_update_historical_max_nbytes();

  LayoutPlanner& planner_;
  ExecutionFrame& parent_frame_;

  std::vector<c10::IValue*> unplanned_ivalues_;

  std::vector<const at::Tensor*> planned_tensors_;
  std::vector<size_t> planned_tensors_max_nbytes_local_;
#ifndef NDEBUG
  c10::FastMap<ValueId, size_t> value_to_vector_idx_map_;
#endif

  ContiguousLayoutBuffer layout_buffer_;
  ContiguousStorageImplBuffer storage_impl_buffer_;

  LayoutManagerState state_{LayoutManagerState::WaitingForValues};
  torch::nativert::LayoutManagerSettings settings_;
};

class LayoutManagerGuard {
 public:
  explicit LayoutManagerGuard(LayoutManager& manager) : manager_(manager) {
    manager_.allocate();
  }
  ~LayoutManagerGuard() {
    manager_.deallocate_and_plan();
  }

  LayoutManagerGuard(LayoutManagerGuard&& other) = delete;
  LayoutManagerGuard(const LayoutManagerGuard& other) = delete;
  LayoutManagerGuard operator=(LayoutManagerGuard&& other) = delete;
  LayoutManagerGuard& operator=(const LayoutManagerGuard& other) = delete;

  LayoutManager& manager_;
};

} // namespace torch::nativert
