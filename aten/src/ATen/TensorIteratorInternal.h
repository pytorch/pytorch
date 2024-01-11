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

template <typename T>
inline void get_data_ptrs(
    T** ptrs,
    ArrayRef<T*> base,
    IntArrayRef strides,
    IntArrayRef counter) {
  static_assert(
      std::is_same<T, char>::value || std::is_same<T, const char>::value,
      "T must be a char or const char");
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

inline void serial_for_each(
    IntArrayRef shape,
    IntArrayRef mutable_strides,
    char** mutable_base_ptrs,
    size_t nmutabletensors,
    IntArrayRef const_strides,
    const char** const_base_ptrs,
    size_t nconsttensors,
    typename TensorIteratorBase::loop2d_with_const_t loop,
    Range range) {
  const auto ndim = shape.size();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      mutable_strides.size() == nmutabletensors * std::max(size_t{2}, ndim));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      const_strides.size() == nconsttensors * std::max(size_t{2}, ndim));

  if (ndim <= 1) {
    if (range.begin == 0) {
      loop(
          mutable_base_ptrs,
          mutable_strides.data(),
          const_base_ptrs,
          const_strides.data(),
          range.size(),
          1);
    } else {
      c10::SmallBuffer<char*, 4> mutable_ptrs(nmutabletensors);
      c10::SmallBuffer<const char*, 4> const_ptrs(nconsttensors);
      get_data_ptrs(
          mutable_ptrs.data(),
          {mutable_base_ptrs, nmutabletensors},
          mutable_strides,
          {range.begin});
      get_data_ptrs(
          const_ptrs.data(),
          {const_base_ptrs, nconsttensors},
          const_strides,
          {range.begin});
      loop(
          mutable_ptrs.data(),
          mutable_strides.data(),
          const_ptrs.data(),
          const_strides.data(),
          range.size(),
          1);
    }
  } else {
    c10::SmallBuffer<char*, 4> mutable_ptrs(nmutabletensors);
    c10::SmallBuffer<const char*, 4> const_ptrs(nconsttensors);
    auto counter = DimCounter(shape, range);
    while (!counter.is_done()) {
      get_data_ptrs(
          mutable_ptrs.data(),
          {mutable_base_ptrs, nmutabletensors},
          mutable_strides,
          counter.values);
      get_data_ptrs(
          const_ptrs.data(),
          {const_base_ptrs, nconsttensors},
          const_strides,
          counter.values);
      auto step = counter.max_2d_step();
      loop(
          mutable_ptrs.data(),
          mutable_strides.data(),
          const_ptrs.data(),
          const_strides.data(),
          step[0],
          step[1]);
      counter.increment(step);
    }
  }
}

} // namespace internal
} // namespace at
