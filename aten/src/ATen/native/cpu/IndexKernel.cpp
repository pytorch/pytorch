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
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "index_cpu", [&] {
    cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
      *(scalar_t*)dst = *(scalar_t*)(src + offset);
    });
  });
}

void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  // NOTE: duplicate indices are only supported if accumulate is true.
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "index_put", [&] {
    if (accumulate) {
      // See Note [Enabling Deterministic Operations]
      // Parallel cpu_index_kernel with accumulation is nondeterministic, so we
      // must enable serial execution if deterministic algorithms are enabled.
      bool is_deterministic = at::globalContext().deterministicAlgorithms();
      bool use_parallel_for = (!is_deterministic) && (
        (iter.numel() >= internal::GRAIN_SIZE) && (at::get_num_threads() > 1));
      if (use_parallel_for && iter.dtype() == ScalarType::Float) {
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

void index_fill_kernel(
  TensorIterator& iter,
  int64_t dim,
  int64_t self_dim_size,
  int64_t self_dim_stride,
  const Scalar& source) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "index_fill_cpu", [&] {
    auto fill_val = source.to<scalar_t>();
    auto handle_nonzero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {
      auto* self_data_bytes = data[0];
      auto* index_data_bytes = data[1];
      for (int64_t elem = 0; elem < n; ++elem) {
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);
        auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
        TORCH_CHECK_INDEX(idx >= -self_dim_size && idx < self_dim_size,
                          "index ", idx, " is out of bounds for dimension ",
                          dim, " with size ", self_dim_size);
        if (idx < 0) {
          idx += self_dim_size;
        }

        self_data[idx * self_dim_stride] = fill_val;

        self_data_bytes += strides[0];
        index_data_bytes += strides[1];
      }
    };
    auto handle_zero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {
      auto* self_data_bytes = data[0];
      auto* index_data_bytes = data[1];
      auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
      TORCH_CHECK_INDEX(idx >= -self_dim_size && idx < self_dim_size,
                        "index ", idx, " is out of bounds for dimension ",
                        dim, " with size ", self_dim_size);
      if (idx < 0) {
        idx += self_dim_size;
      }
      for (int64_t elem = 0; elem < n; ++elem) {
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);

        self_data[idx * self_dim_stride] = fill_val;

        self_data_bytes += strides[0];
      }
    };

    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      auto idx_stride = strides[1];
      if (idx_stride) {
        handle_nonzero_idx_stride(data, strides, n);
      }
      else {
        handle_zero_idx_stride(data, strides, n);
      }
    };
    iter.for_each(loop);
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

void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
    iter.dtype(), "masked_fill", [&] {
      scalar_t scalar_val = value.to<scalar_t>();
      auto mask_dtype = iter.input_dtype(0);
      if (mask_dtype == ScalarType::Bool) {
        cpu_masked_fill_kernel<scalar_t, bool>(iter, scalar_val);
      } else {
        cpu_masked_fill_kernel<scalar_t, unsigned char>(iter, scalar_val);
      }
    });
}

template <typename scalar_t, typename mask_t>
void cpu_masked_scatter_kernel(TensorIterator& iter, const Tensor& source) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  std::ptrdiff_t source_cntr = 0;
  scalar_t* source_ptr = source.data_ptr<scalar_t>();
  auto numel = source.numel();

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    const int64_t dst_stride = strides[0];
    char* mask = data[1];
    const int64_t mask_stride = strides[1];
    for (int64_t i = 0; i < n; i++) {
      mask_t mask_value = *(mask_t*)(mask + mask_stride * i);
      if (!is_mask_bool) {
        TORCH_CHECK(mask_value <= static_cast<mask_t>(1), "Mask tensor can take 0 and 1 values only");
      }
      if (mask_value) {
        TORCH_CHECK(source_cntr < numel, "Number of elements of source < number of ones in mask");
        *(scalar_t*)(dst + dst_stride * i) = *(source_ptr);
        source_ptr++;
        source_cntr++;
      }
    }
  };
  iter.serial_for_each(loop, {0, iter.numel()});
}

void masked_scatter_kernel(TensorIterator& iter, const Tensor& source) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.dtype(),
      "masked_scatter",
      [&] {
        auto mask_dtype = iter.input_dtype(0);
        if (mask_dtype == ScalarType::Bool) {
          cpu_masked_scatter_kernel<scalar_t, bool>(iter, source);
        } else {
          cpu_masked_scatter_kernel<scalar_t, unsigned char>(iter, source);
        }
      });
}

template <typename scalar_t, typename mask_t, typename func_t>
void cpu_masked_select_serial_kernel(TensorIterator& iter, const func_t& f) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  int64_t offset = 0;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* src = data[1];
    char* mask = data[2];
    for (int64_t i = 0; i < n; i++) {
      mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
      if (!is_mask_bool) {
        TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
      }
      if (mask_value) {
        int64_t offset_bytes = offset * sizeof(scalar_t);
        f(dst, src + strides[1] * i, offset_bytes);
        offset++;
      }
    }
  };
  iter.serial_for_each(loop, {0, iter.numel()});
}

void masked_select_serial_kernel(TensorIterator& iter, int64_t result_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
    iter.dtype(), "masked_select", [&] {
      auto mask_dtype = iter.input_dtype(1);
      if (mask_dtype == ScalarType::Bool) {
        cpu_masked_select_serial_kernel<scalar_t, bool>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      } else {
        cpu_masked_select_serial_kernel<scalar_t, unsigned char>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      }
    });
}

template <typename scalar_t, typename mask_t, typename func_t>
void cpu_masked_select_kernel(TensorIterator& iter, const func_t& f) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* src = data[1];
    char* mask = data[2];
    char* mask_prefix_sum = data[3];
    for (int64_t i = 0; i < n; i++) {
      mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
      if (!is_mask_bool) {
        TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
      }
      if (mask_value) {
        int64_t offset = *(int64_t*)(mask_prefix_sum + strides[3] * i);
        int64_t offset_bytes = (offset - 1) * sizeof(scalar_t);
        f(dst, src + strides[1] * i, offset_bytes);
      }
    }
  };
  iter.for_each(loop);
}

void masked_select_kernel(TensorIterator& iter, int64_t result_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
    iter.dtype(), "masked_select", [&] {
      auto mask_dtype = iter.input_dtype(1);
      if (mask_dtype == ScalarType::Bool) {
        cpu_masked_select_kernel<scalar_t, bool>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      } else {
        cpu_masked_select_kernel<scalar_t, unsigned char>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      }
    });
}

} // anonymous namespace

REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_fill_stub, &index_fill_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);
REGISTER_DISPATCH(masked_fill_stub, &masked_fill_kernel);
REGISTER_DISPATCH(masked_select_serial_stub, &masked_select_serial_kernel);
REGISTER_DISPATCH(masked_select_stub, &masked_select_kernel);
REGISTER_DISPATCH(masked_scatter_stub, &masked_scatter_kernel);

}} // namespace at::native
