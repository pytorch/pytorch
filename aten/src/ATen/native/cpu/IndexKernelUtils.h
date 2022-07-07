#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/util/irange.h>

namespace at {
namespace native {

namespace {
static bool is_constant_index(int ntensor, const int64_t* strides) {
  AT_ASSERT(ntensor >= 3);
  for (const auto arg : c10::irange(2, ntensor)) {
    if (strides[arg] != 0) {
      return false;
    }
  }
  return true;
}


struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides,
          IntArrayRef original_sizes, IntArrayRef original_strides)
    : num_indexers(num_indexers)
    , indexers(indexers)
    , indexer_strides(indexer_strides)
    , original_strides(original_strides.data())
    , original_sizes(original_sizes.data()) {
    AT_ASSERT(static_cast<int64_t>(original_strides.size()) == num_indexers);
    AT_ASSERT(static_cast<int64_t>(original_sizes.size()) == num_indexers);
  }

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;
  const int64_t* original_strides;
  const int64_t* original_sizes;

  int64_t get(int64_t idx) {
    int64_t offset = 0;
    for (const auto j : c10::irange(num_indexers)) {
      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
      int64_t size = original_sizes[j];
      TORCH_CHECK_INDEX(value >= -size && value < size,
                        "index ", value, " is out of bounds for dimension ", j, " with size ", size);
      if (value < 0) {
        value += size;
      }
      offset += value * original_strides[j];
    }
    return offset;
  }
};
} // anonymous namespace

template <typename scalar_t, typename func_t>
void cpu_index_kernel(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride,
                      const func_t& f, bool serial_execution=false)
{
  int ntensor = iter.ntensors();
  // When launch the index parallel version, set a relative samll grain size less than the INTERNAL::GRAIN_SIZE
  // to make the whole available thread numbers get more balanced work load and a better cache location.
  // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
  const int index_parallel_grain_size = 3000;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
    char* dst = data[0];
    char* src = data[1];
    if (is_constant_index(ntensor, strides)) {
      // specialization for when every element uses the same index
      int64_t offset = indexer.get(0);
      if (strides[0] == sizeof(scalar_t) && strides[1] == sizeof(scalar_t)) {
        for (const auto i : c10::irange(n)) {
          f(dst + strides[0] * i, src + strides[1] * i, offset);
        }
      } else {
        for (const auto i : c10::irange(n)) {
          f(dst + strides[0] * i, src + strides[1] * i, offset);
        }
      }
    } else {
      for (const auto i : c10::irange(n)) {
        int64_t offset = indexer.get(i);
        f(dst + strides[0] * i, src + strides[1] * i, offset);
      }
    }
  };
  if (serial_execution) {
    iter.serial_for_each(loop, {0, iter.numel()});
  } else {
    iter.for_each(loop, index_parallel_grain_size);
  }
}
} // at
} // native
