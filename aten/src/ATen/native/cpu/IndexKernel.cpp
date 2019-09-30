#include <ATen/native/Indexing.h>

#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {
namespace {

using namespace vec256;

struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides,
          IntArrayRef original_sizes, IntArrayRef original_strides)
    : num_indexers(num_indexers)
    , indexers(indexers)
    , indexer_strides(indexer_strides)
    , original_strides(original_strides.data())
    , original_sizes(original_sizes.data()) {
    AT_ASSERT(original_strides.size() == num_indexers);
    AT_ASSERT(original_sizes.size() == num_indexers);
  }

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;
  const int64_t* original_strides;
  const int64_t* original_sizes;

  int64_t get(int64_t idx) {
    int64_t offset = 0;
    for (int j = 0; j < num_indexers; j++) {
      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
      int64_t size = original_sizes[j];
      if (value < -size || value >= size) {
        AT_INDEX_ERROR("index ", value, " is out of bounds for dimension ", j, " with size ", size);
      }
      if (value < 0) {
        value += size;
      }
      offset += value * original_strides[j];
    }
    return offset;
  }
};

static bool is_constant_index(int ntensor, const int64_t* strides) {
  AT_ASSERT(ntensor >= 3);
  for (int arg = 2; arg < ntensor; arg++) {
    if (strides[arg] != 0) {
      return false;
    }
  }
  return true;
}

template <typename scalar_t, typename func_t>
void cpu_index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride,
                      const func_t& f, bool serial_execution=false)
{
  int ntensor = iter.ntensors();
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
    char* dst = data[0];
    char* src = data[1];
    if (is_constant_index(ntensor, strides)) {
      // specialization for when every element uses the same index
      int64_t offset = indexer.get(0);
      if (strides[0] == sizeof(scalar_t) && strides[1] == sizeof(scalar_t)) {
        for (int64_t i = 0; i < n; i++) {
          f(dst + strides[0] * i, src + strides[1] * i, offset);
        }
      } else {
        for (int64_t i = 0; i < n; i++) {
          f(dst + strides[0] * i, src + strides[1] * i, offset);
        }
      }
    } else {
      for (int64_t i = 0; i < n; i++) {
        int64_t offset = indexer.get(i);
        f(dst + strides[0] * i, src + strides[1] * i, offset);
      }
    }
  };
  if (serial_execution) {
    iter.serial_for_each(loop, {0, iter.numel()});
  } else {
    iter.for_each(loop);
  }
}

void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool, iter.dtype(), "index_cpu", [&] {
    cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
      *(scalar_t*)dst = *(scalar_t*)(src + offset);
    });
  });
}

void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  // NOTE: duplicate indices are only supported if accumulate is true.
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool, iter.dtype(), "index_put", [&] {
    if (accumulate) {
      // TODO: investigate parallelization of the accumulate kernel. Unlike the non-accumulate case,
      // this needs to be thread-safe.
      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
        *(scalar_t*)(dst + offset) += *(scalar_t*)src;
      }, /*serial_execution=*/true);
    } else {
      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
        *(scalar_t*)(dst + offset) = *(scalar_t*)src;
      });
    }
  });
}

} // anonymous namespace


REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);

}} // namespace at::native
