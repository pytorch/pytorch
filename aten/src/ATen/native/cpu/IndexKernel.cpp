#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/IndexKernel.h>

#include <cmath>
#include <iostream>

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/AtomicAddFloat.h>
#include <ATen/native/cpu/IndexKernelUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>
#include <c10/core/Scalar.h>

namespace at { namespace native {
namespace {

using namespace vec;

void index_kernel(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
    iter.dtype(), "index_cpu", [&] {
    cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
      *(scalar_t*)dst = *(scalar_t*)(src + offset);
    });
  });
}

// Given a linear index, returns the offset of the tensor.
// Implements the same algorithm as its (legacy) GPU version cuda::detail::IndexToOffset
// OffsetCalculator implements yet again the same algorithm but in a column-major order
struct IndexToOffset {
  const IntArrayRef sizes;
  const IntArrayRef strides;
  const int64_t ndim;
  explicit IndexToOffset(const TensorBase & tensor) :
      sizes(tensor.sizes()), strides(tensor.strides()), ndim(tensor.dim()) {
  }

  int64_t get(int64_t linear_index) const {
    int64_t offset = 0;
    for (int64_t i = ndim - 1; i > 0; i--) {
      offset += (linear_index % sizes[i]) * strides[i];
      linear_index /= sizes[i];
    }
    return offset + linear_index * strides[0];
  }
};

template <typename scalar_t, typename func_t>
void cpu_take_put_kernel(
    TensorIterator& iter,
    const TensorBase& indexed,
    const func_t& f,
    bool serial_execution=false) {
  // This kernel follows the same strategy as `cpu_index_kernel`
  // Even though the indexed_tensor is const, we modify it through the data_ptr
  // This is a bit dirty, but otherwise it would be necessary to innecessarily add tensor
  // with zero strides to `iter` which would not be much better

  // When launch the parallel version, set a relative small grain size less than the INTERNAL::GRAIN_SIZE
  // to make the whole available thread numbers get more balanced work load and a better cache location.
  // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
  // Perhaps tweak this number for `put_`? This number was tweaked for `index_put`
  constexpr int parallel_grain_size = 3000;
  const bool is_contiguous = indexed.is_contiguous();
  const auto numel = indexed.numel();
  const auto offset_indexed = IndexToOffset(indexed);

  auto* indexed_data = indexed.data_ptr<scalar_t>();
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* iterated_data_bytes = data[0];
    auto* index_data_bytes = data[1];
    for (const auto elem C10_UNUSED : c10::irange(n)) {
      auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
      auto& iterated = *reinterpret_cast<scalar_t*>(iterated_data_bytes);

      TORCH_CHECK_INDEX(idx >= -numel && idx < numel,
                        "out of range: tried to access index ",
                        idx, " on a tensor of ", numel, " elements.");
      if (idx < 0) {
        idx += numel;
      }
      if (!is_contiguous) {
        idx = offset_indexed.get(idx);
      }
      f(iterated, indexed_data, idx);
      iterated_data_bytes += strides[0];
      index_data_bytes += strides[1];
    }
  };
  if (serial_execution) {
    iter.serial_for_each(loop, {0, iter.numel()});
  } else {
    iter.for_each(loop, parallel_grain_size);
  }
}

void put_kernel(
  TensorIterator& iter,
  const TensorBase & self,
  const bool accumulate) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "take_put_cpu", [&] {
  // iter could be const, but for_each does not have a const version
    if (accumulate) {
      // nb. This deterministic issue the same as that of `index_put_kernel`
      // See Note [Enabling Deterministic Operations]
      // Parallel cpu_put_kernel with accumulation is nondeterministic, so we
      // must enable serial execution if deterministic algorithms are enabled.
      bool is_deterministic = at::globalContext().deterministicAlgorithms();
      bool use_parallel_for = (!is_deterministic) && (
        (iter.numel() >= internal::GRAIN_SIZE) && (at::get_num_threads() > 1));
      if (use_parallel_for && iter.dtype() == ScalarType::Float) {
        cpu_take_put_kernel<float>(iter, self,
            [](float& iterated, float* indexed, const int64_t idx) {
                cpu_atomic_add_float(indexed+idx, iterated);
              });
      } else {
        // TODO: investigate parallelization of the accumulate kernel.
        // Unlike the non-accumulate case, this needs to be thread-safe.
        cpu_take_put_kernel<scalar_t>(iter, self,
            [](scalar_t& iterated, scalar_t* indexed, const int64_t idx) {
                indexed[idx] += iterated;
              },
            /*serial_execution=*/true);
      }
    } else {
      cpu_take_put_kernel<scalar_t>(iter, self,
          [](scalar_t& iterated, scalar_t* indexed, const int64_t idx) {
              indexed[idx] = iterated;
            });
    }
  });
}

void take_kernel(
  TensorIterator& iter,
  const TensorBase & input) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "take_cpu", [&] {
      cpu_take_put_kernel<scalar_t>(iter, input,
          [](scalar_t& iterated, scalar_t* indexed, const int64_t idx) {
              iterated = indexed[idx];
            });
    });
}

void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  // NOTE: duplicate indices are only supported if accumulate is true.
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
    iter.dtype(), "index_put", [&] {
    // See Note [Enabling Deterministic Operations]
    // Parallel cpu_index_kernel with accumulation is nondeterministic, so we
    // must enable serial execution if deterministic algorithms are enabled.
    const bool is_deterministic = at::globalContext().deterministicAlgorithms();
    if (accumulate) {
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
      }, /*serial_execution=*/is_deterministic);
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
      for (const auto elem C10_UNUSED : c10::irange(n)) {
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
      for (const auto elem C10_UNUSED: c10::irange(n)) {
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

void index_copy_kernel(
  TensorIterator& iter,
  int64_t dim,
  int64_t self_dim_size,
  int64_t self_dim_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "index_copy_cpu", [&] {
    auto handle_nonzero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {
      auto* self_data_bytes = data[0];
      auto* index_data_bytes = data[1];
      auto* source_data_bytes = data[2];
      for (const auto elem C10_UNUSED : c10::irange(n)) {
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);
        auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
        auto* source_data = reinterpret_cast<scalar_t*>(source_data_bytes);
        TORCH_CHECK_INDEX(idx >= 0 && idx < self_dim_size,
              "index_copy_(): index ", idx, " is out of bounds for dimension ",
              dim, " with size ", self_dim_size);

        self_data[idx * self_dim_stride] = *source_data;

        self_data_bytes += strides[0];
        index_data_bytes += strides[1];
        source_data_bytes += strides[2];
      }
    };
    auto handle_zero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {
      auto* self_data_bytes = data[0];
      auto* index_data_bytes = data[1];
      auto* source_data_bytes = data[2];
      auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
      TORCH_CHECK_INDEX(idx >= 0 && idx < self_dim_size,
            "index_copy_(): index ", idx, " is out of bounds for dimension ",
            dim, " with size ", self_dim_size);
      for (const auto elem C10_UNUSED : c10::irange(n)) {
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);
        auto* source_data = reinterpret_cast<scalar_t*>(source_data_bytes);

        self_data[idx * self_dim_stride] = *source_data;

        self_data_bytes += strides[0];
        source_data_bytes += strides[2];
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
    bool is_deterministic = at::globalContext().deterministicAlgorithms();
    if (is_deterministic) {
      iter.serial_for_each(loop, {0, iter.numel()});
    } else {
      iter.for_each(loop);
    }
  });
}

template <typename scalar_t, typename mask_t>
void cpu_masked_fill_kernel(TensorIterator& iter, scalar_t value) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* mask = data[1];
    for (const auto i : c10::irange(n)) {
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
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kBool, kBFloat16, kHalf,
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
void cpu_masked_scatter_kernel(TensorIterator& iter, const TensorBase& source) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  std::ptrdiff_t source_cntr = 0;
  scalar_t* source_ptr = source.data_ptr<scalar_t>();
  auto numel = source.numel();

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    const int64_t dst_stride = strides[0];
    char* mask = data[1];
    const int64_t mask_stride = strides[1];
    for (const auto i : c10::irange(n)) {
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

void masked_scatter_kernel(TensorIterator& iter, const TensorBase& source) {
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
    for (const auto i : c10::irange(n)) {
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
    for (const auto i : c10::irange(n)) {
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

void flip_kernel(TensorIterator& iter, const bool quantized) {
  if (quantized) {
    AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(iter.dtype(), "flip_quantized_cpu",
        [&iter] { cpu_kernel(iter,
          [](scalar_t a, scalar_t /*dummy input*/) -> scalar_t {
            return a;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(), "flip_cpu",
        [&iter] { cpu_kernel_vec(iter,
          [](scalar_t a, scalar_t /*dummy input*/) -> scalar_t {
            return a;
        },
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> /*dummy input*/) -> Vectorized<scalar_t> {
            return a;
        });
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_fill_stub, &index_fill_kernel);
REGISTER_DISPATCH(index_copy_stub, &index_copy_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);
REGISTER_DISPATCH(put_stub, &put_kernel);
REGISTER_DISPATCH(take_stub, &take_kernel);
REGISTER_DISPATCH(masked_fill_stub, &masked_fill_kernel);
REGISTER_DISPATCH(masked_select_serial_stub, &masked_select_serial_kernel);
REGISTER_DISPATCH(masked_select_stub, &masked_select_kernel);
REGISTER_DISPATCH(masked_scatter_stub, &masked_scatter_kernel);
REGISTER_DISPATCH(flip_stub, &flip_kernel);

}} // namespace at::native
