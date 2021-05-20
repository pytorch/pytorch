#include <ATen/native/TensorIterator.h>

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

inline SmallVector<char*, 4> get_data_ptrs(
    ArrayRef<char*> base, IntArrayRef strides, IntArrayRef counter) {
  int64_t ntensors = base.size();
  int64_t ndim = strides.size() / ntensors;
  auto ptrs = SmallVector<char*, 4>(base);
  for (int dim = 0; dim < ndim; dim++) {
    int64_t value = counter[dim];
    for (int arg = 0; arg < ntensors; arg++) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
  return ptrs;
}

inline void serial_for_each(
    IntArrayRef shape, IntArrayRef strides, ArrayRef<char*> base_ptrs,
    typename TensorIteratorBase::loop2d_t loop, Range range) {
  auto ntensors = base_ptrs.size();

  if (strides.size() <= ntensors) {  // ndim <= 1
    auto ptrs = get_data_ptrs(base_ptrs, strides, {range.begin});
    // Pad strides to 2d
    DimVector padded_strides(2 * ntensors, 0);
    std::copy(strides.begin(), strides.end(), padded_strides.begin());
    loop(ptrs.data(), padded_strides.data(), range.size(), 1);
  } else {
    auto counter = DimCounter(shape, range);
    while (!counter.is_done()) {
      auto ptrs = get_data_ptrs(base_ptrs, strides, counter.values);
      auto step = counter.max_2d_step();
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      counter.increment(step);
    }
  }

}

}}  // namespace at::internal
