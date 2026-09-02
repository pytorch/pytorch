#pragma once

#include <ATen/MapAllocator.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace at::ipc {

struct AtomicCounterOps final {
  static int64_t load(const int64_t* counter_ptr) {
    return __atomic_load_n(counter_ptr, __ATOMIC_ACQUIRE);
  }

  static void store(int64_t* counter_ptr, int64_t value) {
    __atomic_store_n(counter_ptr, value, __ATOMIC_RELEASE);
  }

  static int64_t decrement(int64_t* counter_ptr) {
    return __atomic_fetch_sub(
        counter_ptr, static_cast<int64_t>(1), __ATOMIC_ACQ_REL);
  }
};

template <typename CounterT, typename CounterOps>
class SentDataBase {
 public:
  SentDataBase(std::string handle, uint64_t offset, CounterT* counter_ptr)
      : handle_(std::move(handle)),
        offset_(offset),
        counter_ptr_(counter_ptr) {}

  CounterT counter_value() const {
    return CounterOps::load(counter_ptr_);
  }

  const std::string& handle() const {
    return handle_;
  }

  uint64_t offset() const {
    return offset_;
  }

  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }

 protected:
  std::string handle_;
  uint64_t offset_;
  CounterT* counter_ptr_;
  at::DataPtr original_ptr_;
};

template <typename CounterT, typename CounterOps>
struct RefCountersFile final {
  RefCountersFile(std::string handle, uint64_t size, at::DataPtr data_ptr)
      : size_(size),
        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  CounterT* counter_ptr() {
    return static_cast<CounterT*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  void set_counter(CounterT value) {
    CounterOps::store(counter_ptr(), value);
  }

  bool have_offsets() const {
    return next_offset_ < size_;
  }

  bool offsets_in_use() const {
    return used_slots_ > 0;
  }

  uint64_t get_offset() const {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t offset /* unused */) {
    (void)offset;
    used_slots_--;
  }

  const std::string& handle() const {
    return handle_;
  }

 private:
  uint64_t next_offset_{0};
  uint64_t size_;
  uint64_t used_slots_{0};
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

template <typename SentData>
class SentDataLimboBase {
 public:
  bool collect() {
    bool freed_memory = false;
    std::vector<std::unique_ptr<SentData>> reset_blocks;
    {
      std::lock_guard<std::mutex> lock(limbo_mutex_);
      std::vector<std::unique_ptr<SentData>> kept_blocks;
      kept_blocks.reserve(shared_blocks_.size());
      for (auto& sd : shared_blocks_) {
        if (sd->counter_value() > 0) {
          kept_blocks.push_back(std::move(sd));
        } else {
          freed_memory = true;
          reset_blocks.push_back(std::move(sd));
        }
      }
      shared_blocks_ = std::move(kept_blocks);
    }
    for (auto& sd : reset_blocks) {
      sd.reset();
    }
    return freed_memory;
  }

  void add(std::unique_ptr<SentData> shared_block) {
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    shared_blocks_.push_back(std::move(shared_block));
  }

  uint64_t size() const {
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    return shared_blocks_.size();
  }

 protected:
  mutable std::mutex limbo_mutex_;
  std::vector<std::unique_ptr<SentData>> shared_blocks_;
};

} // namespace at::ipc
