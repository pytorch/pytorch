#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/util/SmallBuffer.h>

namespace at {

struct DimCounter {
  DimCounter(IntArrayRef shape, Range range);

  void increment(const std::array<int64_t, 2>& step);
  bool is_done() const;
  std::array<int64_t, 2> max_2d_step() const;

  IntArrayRef shape;
  Range range;
  DimVector values;
  int64_t offset;
};

namespace internal {

inline void get_data_ptrs(
    char** ptrs, ArrayRef<char*> base, IntArrayRef strides, IntArrayRef counter) {
  const int64_t ntensors = base.size();
  const int64_t ndim = counter.size();
  std::copy(base.begin(), base.end(), ptrs);
  for (int64_t dim = 0; dim < ndim; ++dim) {
    int64_t value = counter[dim];
    for (int64_t arg = 0; arg < ntensors; ++arg) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
}

inline void serial_for_each(
    IntArrayRef shape, IntArrayRef strides, ArrayRef<char*> base_ptrs,
    typename TensorIteratorBase::loop2d_t loop, Range range) {
  auto ntensors = base_ptrs.size();
  c10::SmallBuffer<char*, 4> ptrs(ntensors);

  if (strides.size() <= ntensors) {  // ndim <= 1
    if (range.begin == 0) {
      std::copy(base_ptrs.begin(), base_ptrs.end(), ptrs.begin());
    } else {
      get_data_ptrs(ptrs.data(), base_ptrs, strides, {range.begin});
    }
    // Pad strides to 2d
    c10::SmallBuffer<int64_t, 8> padded_strides(2 * ntensors);
    std::copy(strides.begin(), strides.end(), padded_strides.begin());
    std::fill(padded_strides.data() + strides.size(), padded_strides.end(), 0);
    loop(ptrs.data(), padded_strides.data(), range.size(), 1);
  } else {
    auto counter = DimCounter(shape, range);
    while (!counter.is_done()) {
      get_data_ptrs(ptrs.data(), base_ptrs, strides, counter.values);
      auto step = counter.max_2d_step();
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      counter.increment(step);
    }
  }

}

}}  // namespace at::internal
