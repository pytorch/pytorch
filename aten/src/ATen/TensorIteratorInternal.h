#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/util/SmallBuffer.h>
#include <c10/util/irange.h>

namespace at {

struct DimCounter {
  DimCounter(IntArrayRef shape, Range range);

  void increment(const std::array<int64_t, 2>& step);
  bool is_done() const;
  std::array<int64_t, 2> max_2d_step() const;

  IntArrayRef shape;
  Range range;
  c10::SmallBuffer<int64_t, 4> values;
  int64_t offset;
};

namespace internal {

namespace {
bool output_has_update(
    DimCounter& counter,
    DimCounter& counter_pre,
    IntArrayRef& strides,
    int64_t ntensors) {
  int64_t ndim = counter.values.size();
  bool has_update = false;
  if (counter.is_done()) {
    return true;
  }
  for (const auto i : c10::irange(ndim)) {
    if (strides[i * ntensors] > 0) {
      if (has_update) {
        TORCH_INTERNAL_ASSERT(counter_pre.values[i] == counter.values[i]);
      }
      if (counter_pre.values[i] < counter.values[i]) {
        TORCH_INTERNAL_ASSERT(counter_pre.values[i] == counter.values[i] - 1);
        has_update = true;
      }
      counter_pre.values[i] = counter.values[i];
    }
  }
  return has_update;
}
} // namespace

inline void get_data_ptrs(
    char** ptrs,
    ArrayRef<char*> base,
    IntArrayRef strides,
    IntArrayRef counter) {
  const int64_t ntensors = base.size();
  const int64_t ndim = counter.size();
  std::copy(base.begin(), base.end(), ptrs);
  for (const auto dim : c10::irange(ndim)) {
    int64_t value = counter[dim];
    for (const auto arg : c10::irange(ntensors)) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
}

inline void serial_for_each_acc(
    IntArrayRef shape,
    IntArrayRef strides,
    char** base_ptrs,
    size_t ntensors,
    typename TensorIteratorBase::loop2d_t loop,
    typename TensorIteratorBase::sync_acc_t sync,
    Range range) {
  const auto ndim = shape.size();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      strides.size() == ntensors * std::max(size_t{2}, ndim));

  float acc_value = 0;
  char* buffer_ptr = reinterpret_cast<char*>(&acc_value);

  if (ndim <= 1) {
    if (range.begin == 0) {
      sync(static_cast<char*>(buffer_ptr), base_ptrs, strides.data(), true);
      auto orig_ptrs = base_ptrs[0];
      base_ptrs[0] = buffer_ptr;
      loop(base_ptrs, strides.data(), range.size(), 1);
      base_ptrs[0] = orig_ptrs;
      sync(static_cast<char*>(buffer_ptr), base_ptrs, strides.data(), false);
    } else {
      c10::SmallBuffer<char*, 4> ptrs(ntensors);
      get_data_ptrs(ptrs.data(), {base_ptrs, ntensors}, strides, {range.begin});
      sync(static_cast<char*>(buffer_ptr), ptrs.data(), strides.data(), true);
      auto orig_ptrs = ptrs.data()[0];
      ptrs.data()[0] = buffer_ptr;
      loop(ptrs.data(), strides.data(), range.size(), 1);
      ptrs.data()[0] = orig_ptrs;
      sync(static_cast<char*>(buffer_ptr), ptrs.data(), strides.data(), false);
    }
  } else {
    c10::SmallBuffer<char*, 4> ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    auto pre_counter = DimCounter(shape, range);
    bool fist_init = true;
    bool has_update = false;
    auto orig_ptrs = ptrs.data()[0];
    while (!counter.is_done()) {
      get_data_ptrs(
          ptrs.data(), {base_ptrs, ntensors}, strides, counter.values);

      if (fist_init || has_update) {
        sync(static_cast<char*>(buffer_ptr), ptrs.data(), strides.data(), true);
      }
      auto step = counter.max_2d_step();
      orig_ptrs = ptrs.data()[0];
      ptrs.data()[0] = buffer_ptr;
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      counter.increment(step);
      has_update = output_has_update(counter, pre_counter, strides, ntensors);

      if (fist_init || has_update) {
        fist_init = false;
        ptrs.data()[0] = orig_ptrs;
        sync(
            static_cast<char*>(buffer_ptr), ptrs.data(), strides.data(), false);
      }
    }
  }
}

inline void serial_for_each(
    IntArrayRef shape,
    IntArrayRef strides,
    char** base_ptrs,
    size_t ntensors,
    typename TensorIteratorBase::loop2d_t loop,
    Range range) {
  const auto ndim = shape.size();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      strides.size() == ntensors * std::max(size_t{2}, ndim));

  if (ndim <= 1) {
    if (range.begin == 0) {
      loop(base_ptrs, strides.data(), range.size(), 1);
    } else {
      c10::SmallBuffer<char*, 4> ptrs(ntensors);
      get_data_ptrs(ptrs.data(), {base_ptrs, ntensors}, strides, {range.begin});
      loop(ptrs.data(), strides.data(), range.size(), 1);
    }
  } else {
    c10::SmallBuffer<char*, 4> ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    while (!counter.is_done()) {
      get_data_ptrs(
          ptrs.data(), {base_ptrs, ntensors}, strides, counter.values);
      auto step = counter.max_2d_step();
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      counter.increment(step);
    }
  }
}

} // namespace internal
} // namespace at
