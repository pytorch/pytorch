#include <ATen/native/TensorAdvancedIndexing.h>

#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/cpu/AtomicAddFloat.h>

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
    iter.for_each(loop, index_parallel_grain_size);
  }
}

void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(), "index_cpu", [&] {
    cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
      *(scalar_t*)dst = *(scalar_t*)(src + offset);
    });
  });
}

void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  // NOTE: duplicate indices are only supported if accumulate is true.
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(), "index_put", [&] {
    if (accumulate) {
      bool use_parallel_for = ((iter.numel() >= internal::GRAIN_SIZE) && (at::get_num_threads() > 1));
      if (iter.dtype() == at::ScalarType::Float && use_parallel_for) {
        cpu_index_kernel<float>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
          cpu_atomic_add_float((float*)(dst + offset), *(float*)src);
        });
      } else {
        // TODO: investigate parallelization of the accumulate kernel. Unlike the non-accumulate case,
        // this needs to be thread-safe.
        cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset) += *(scalar_t*)src;
        }, /*serial_execution=*/true);
      }
    } else {
      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
        *(scalar_t*)(dst + offset) = *(scalar_t*)src;
      });
    }
  });
}

template <typename scalar_t, typename mask_t>
void cpu_masked_fill_kernel(TensorIterator& iter, scalar_t value) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* mask = data[1];
    for (int64_t i = 0; i < n; i++) {
      mask_t mask_value = *(mask_t*)(mask + strides[1] * i);
      if (!is_mask_bool) {
        TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
      }
      if (mask_value) {
        *(scalar_t*)(dst + strides[0] * i) = value;
      }
    }
  };
  iter.for_each(loop);
}

void masked_fill_kernel(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(), "masked_fill", [&] {
      scalar_t scalar_val = value.to<scalar_t>();
      auto mask_dtype = iter.input_dtype(0);
      if (mask_dtype == at::ScalarType::Bool) {
        cpu_masked_fill_kernel<scalar_t, bool>(iter, scalar_val);
      } else {
        cpu_masked_fill_kernel<scalar_t, unsigned char>(iter, scalar_val);
      }
    });
}

} // anonymous namespace


REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);
REGISTER_DISPATCH(masked_fill_stub, &masked_fill_kernel);

}} // namespace at::native
