#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/IndexKernel.h>

#include <cmath>
#include <iostream>

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/AtomicAddFloat.h>
#include <ATen/native/cpu/IndexKernelUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>
#include <c10/core/Scalar.h>

namespace at::native {
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
    bool is_indexed_data_mutated,
    const func_t& f,
    bool serial_execution=false) {
  // This kernel follows the same strategy as `cpu_index_kernel`
  // Even though the indexed_tensor is const, we modify it through the data_ptr
  // This is a bit dirty, but otherwise it would be necessary to unnecessarily add tensor
  // with zero strides to `iter` which would not be much better

  // When launch the parallel version, set a relative small grain size less than the INTERNAL::GRAIN_SIZE
  // to make the whole available thread numbers get more balanced work load and a better cache location.
  // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
  // Perhaps tweak this number for `put_`? This number was tweaked for `index_put`
  constexpr int parallel_grain_size = 3000;
  const bool is_contiguous = indexed.is_contiguous();
  const auto numel = indexed.numel();
  const auto offset_indexed = IndexToOffset(indexed);

  auto* indexed_data = is_indexed_data_mutated ?
   indexed.data_ptr<scalar_t>()
   : const_cast<scalar_t*>(indexed.const_data_ptr<scalar_t>());
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
        cpu_take_put_kernel<float>(iter, self, true,
            [](float& iterated, float* indexed, const int64_t idx) {
                cpu_atomic_add_float(indexed+idx, iterated);
              });
      } else {
        // TODO: investigate parallelization of the accumulate kernel.
        // Unlike the non-accumulate case, this needs to be thread-safe.
        cpu_take_put_kernel<scalar_t>(iter, self, true,
            [](scalar_t& iterated, scalar_t* indexed, const int64_t idx) {
                indexed[idx] += iterated;
              },
            /*serial_execution=*/true);
      }
    } else {
      cpu_take_put_kernel<scalar_t>(iter, self, true,
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
      cpu_take_put_kernel<scalar_t>(iter, input, false,
          [](scalar_t& iterated, const scalar_t* indexed, const int64_t idx) {
              iterated = indexed[idx];
            });
    });
}

void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  // NOTE: duplicate indices are only supported if accumulate is true.
  AT_DISPATCH_V2(
    iter.dtype(),
    "index_put",
    AT_WRAP([&] {
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
    }),
    AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
    AT_EXPAND(AT_FLOAT8_TYPES),
    kComplexHalf,
    kHalf,
    kBool,
    kBFloat16);
}

void index_fill_kernel(
  TensorIterator& iter,
  int64_t dim,
  int64_t self_dim_size,
  int64_t self_dim_stride,
  const Scalar& source) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, kComplexHalf,
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
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, kComplexHalf,
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

template <typename scalar_t>
void cpu_masked_fill_kernel(TensorIterator& iter, scalar_t value) {
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* mask = data[1];
    for (const auto i : c10::irange(n)) {
      bool mask_value = *reinterpret_cast<bool*>(mask + strides[1] * i);

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
      TORCH_CHECK(mask_dtype == ScalarType::Bool, "masked_fill only supports boolean masks, "
        "but got mask with dtype ", mask_dtype);
      cpu_masked_fill_kernel<scalar_t>(iter, scalar_val);
    });
}

template <typename scalar_t>
void cpu_masked_scatter_kernel(TensorIterator& iter, const TensorBase& source) {
  std::ptrdiff_t source_cntr = 0;
  const scalar_t* source_ptr = source.const_data_ptr<scalar_t>();
  auto numel = source.numel();

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    const int64_t dst_stride = strides[0];
    char* mask = data[1];
    const int64_t mask_stride = strides[1];
    for (const auto i : c10::irange(n)) {
      auto mask_value = *reinterpret_cast<bool*>(mask + mask_stride * i);
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
 TORCH_CHECK(iter.input_dtype() == ScalarType::Bool, "masked_scatter_ only supports boolean masks, "
    "but got mask with dtype ", iter.input_dtype());
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.dtype(),
      "masked_scatter",
      [&] {
          cpu_masked_scatter_kernel<scalar_t>(iter, source);
      });
}

template <typename scalar_t, typename mask_t, typename func_t>
void cpu_masked_select_serial_kernel(TensorIterator& iter, const func_t& f) {
  int64_t offset = 0;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* src = data[1];
    char* mask = data[2];
    for (const auto i : c10::irange(n)) {
      mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
      if constexpr (!std::is_same<mask_t, bool>::value) {
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
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* src = data[1];
    char* mask = data[2];
    char* mask_prefix_sum = data[3];
    for (const auto i : c10::irange(n)) {
      mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
      if constexpr (!std::is_same<mask_t, bool>::value) {
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

template <typename scalar_t>
void cpu_hflip_vec(at::TensorIterator& iter) {

  auto loop2d = [&](char** base, const int64_t *strides, int64_t size0, int64_t size1) {

    // Here ntensors is defined for output and 1 input. But tensor iterator has defined output, input
    // and restrided_input (see aten/src/ATen/native/TensorTransformations.cpp#L64-L66) but we use only
    // output and input.
    static constexpr int ntensors = 2;
    const int64_t *outer_strides = &strides[3];

    std::array<char*, ntensors> data_arr;
    std::copy_n(base, ntensors, data_arr.data());

    using Vec = Vectorized<scalar_t>;

    constexpr auto stride = sizeof(scalar_t);
    TORCH_INTERNAL_ASSERT(stride == -strides[0] && stride == strides[1]);

    for (const auto j C10_UNUSED : c10::irange(size1)) {

      // vectorized loop with negative stride for output
      char** C10_RESTRICT data_ = data_arr.data();
      int64_t n = size0;

      char* C10_RESTRICT data[ntensors];
      for (const auto arg : c10::irange(ntensors)) {
        data[arg] = data_[arg];
      }

      int64_t i = 0;

      // data[0] unaligned pre-pass
      int64_t offset = (j * n + (n - i - Vec::size())) % 32;
      offset = (offset >= n) ? n : offset;
      for (; i < offset; i++) {
        scalar_t* out_ptr = (scalar_t*)(data[0] - i * stride);
        *out_ptr = *(scalar_t *)(data[1] + i * stride);
      }
      // Empirically found that it is faster to process 3 data items together vs 2 or 4
      for (; i <= n - 3 * Vec::size(); i += 3 * Vec::size()) {
        auto out1 = Vec::loadu(data[1] + i * stride);
        auto out2 = Vec::loadu(data[1] + (i + Vec::size()) * stride);
        auto out3 = Vec::loadu(data[1] + (i + 2 * Vec::size()) * stride);
        // flip the vector: 1234 -> 4321
        out1 = flip(out1);
        out2 = flip(out2);
        out3 = flip(out3);
        out1.store(data[0] - (i + Vec::size() - 1) * stride);
        out2.store(data[0] - (i + 2 * Vec::size() - 1) * stride);
        out3.store(data[0] - (i + 3 * Vec::size() - 1) * stride);
      }
      if (i < n) {
        for (; i < n; i++) {
          scalar_t* out_ptr = (scalar_t*)(data[0] - i * stride);
          *out_ptr = *(scalar_t *)(data[1] + i * stride);
        }
      }

      // advance:
      for (const auto arg : c10::irange(ntensors)) {
        data_arr[arg] += outer_strides[arg];
      }
    }
  };

  int64_t grain_size = at::internal::GRAIN_SIZE;
  iter.for_each(loop2d, grain_size);
  iter.cast_outputs();
}

void cpu_vflip_memcpy(at::TensorIterator& iter) {
  // This is a vertical flip specialization using memcpy to speed-up the runtime

  auto loop2d = [&](char** base, const int64_t *strides, int64_t size0, int64_t size1) {

    // Here ntensors is defined for output and 1 input. But tensor iterator has defined output, input
    // and restrided_input (see aten/src/ATen/native/TensorTransformations.cpp#L64-L66) but we use only
    // output and input.
    static constexpr int ntensors = 2;
    const int64_t *outer_strides = &strides[3];

    std::array<char*, ntensors> data_arr;
    std::copy_n(base, ntensors, data_arr.data());

    TORCH_INTERNAL_ASSERT(strides[0] == strides[1]);
    const int64_t stride = strides[0];

    for (const auto j C10_UNUSED : c10::irange(size1)) {

      char** C10_RESTRICT data_ = data_arr.data();
      int64_t n = size0;

      char* C10_RESTRICT data[ntensors];
      for (const auto arg : c10::irange(ntensors)) {
        data[arg] = data_[arg];
      }

      memcpy(data[0], data[1], n * stride);

      // advance:
      for (const auto arg : c10::irange(data_arr.size())) {
        data_arr[arg] += outer_strides[arg];
      }
    }
  };

  int64_t grain_size = at::internal::GRAIN_SIZE;
  iter.for_each(loop2d, grain_size);
  iter.cast_outputs();
}

constexpr int64_t hflip_mask_size = 32;

std::array<char, hflip_mask_size> generate_vec_hflip_reg_mask(int64_t data_stride) {
    std::array<char, hflip_mask_size> mask;
    for (const auto k : c10::irange(hflip_mask_size / 2)) {
      int j = k / data_stride + 1;
      int v = (j * data_stride - 1) - (k % data_stride);
      v = std::min(v, (int) (hflip_mask_size / 2 - 1));
      mask[hflip_mask_size - 1 - k] = v;
      mask[hflip_mask_size / 2 - 1 - k] = v;
    }
    return mask;
}

int64_t vectorized_cpu_hflip_channels_last(
    char * C10_RESTRICT *data, const int64_t data_size, const int64_t data_stride, const std::array<char, 32> & mdata) {

  int64_t i = 0;
#ifdef CPU_CAPABILITY_AVX2

  constexpr auto vec_size = 256 / 8;

  if (data_size > vec_size) {

      // Example for num channels=3 and dtype=uint8
      // -> data_stride = 3
      // -> usable_vec_stride = 30
      // -> usable_vec_half_stride = 15
      // Data: (1 2 3) (4 5 6) (7 8 9) (10 11 12) (13 14 15) (16 17 18) (19 20 21) (22 23 24) (25 26 27) (28 29 30) (31 32 33)
      // load by 2 parts
      // R = [ (1 2 3) (4 5 6) (7 8 9) (10 11 12) (13 14 15) (16 | (16 17 18) (19 20 21) (22 23 24) (25 26 27) (28 29 30) (31 ]
      // flip(R) ->
      // R = [ 31 (28 29 30) (25 26 27) (22 23 24) (19 20 21) (16 17 18) | 16 (13 14 15) (10 11 12) (7 8 9) (4 5 6) (1 2 3) ]
      //
      // Write in 2 parts
      // Output pointer: output_ptr = data[0]                                                                                  v
      // - Init:
      //                (X X X)  (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X) (X X X) (X X X)
      // 0) Move to initial position: output_ptr = data[0] + data_stride - vec_size / 2;
      //                                                                          v
      //                (X X X)  (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X) (X X X) (X X X)
      // - In the loop:
      // 1) Write 1st block from output_ptr
      //                                                                            v
      //                                                                            |----> vec_size / 2 ---------------------------|
      // Output part 1: (X X X)  (X X X)    (X X X)    (X X X)    (X X X)     (X X 16)  (13 14 15) (10 11 12) (7 8 9) (4 5 6) (1 2 3)
      // 2) Write 2nd block from output_ptr - usable_vec_half_stride:
      //                                                                            v
      //                     |-----> vec_size / 2 ----------------------------------|
      // Output part 2: (X X 31) (28 29 30) (25 26 27) (22 23 24) (19 20 21) (16 17 18) (13 14 15) (10 11 12) (7 8 9) (4 5 6) (1 2 3)
      //
      // 3) Move to the next position: output_ptr -= usable_vec_stride
      //
      // - After the loop:
      // 4) Move to write position
      //                 v
      //                (X X 31) (28 29 30) (25 26 27) (22 23 24) (19 20 21) (16 17 18) (13 14 15) (10 11 12) (7 8 9) (4 5 6) (1 2 3)

    const __m256i mask = _mm256_loadu_si256((__m256i *) mdata.data());

    const auto usable_vec_stride = 2 * (vec_size / 2 / data_stride) * data_stride;
    const auto usable_vec_half_stride = usable_vec_stride / 2;

    auto output_ptr = data[0] + data_stride - vec_size / 2;
    auto input_ptr = data[1];

    for (; i < data_size - vec_size; i += usable_vec_stride) {

      // load 256-bits by two 128-bits parts
      auto a0 = _mm_loadu_si128((__m128i *) (input_ptr + i));
      auto b0 = _mm256_castsi128_si256(a0);
      auto a1 = _mm_loadu_si128((__m128i *) (input_ptr + i + usable_vec_half_stride));
      auto data_vec = _mm256_inserti128_si256(b0, a1, 1);

      auto reversed_vec = _mm256_shuffle_epi8(data_vec, mask);

      // write output in two parts
      auto rev_vec_h = _mm256_extracti128_si256(reversed_vec, 0);
      _mm_storeu_si128((__m128i *) (output_ptr - i), rev_vec_h);
      auto rev_vec_l = _mm256_extracti128_si256(reversed_vec, 1);
      _mm_storeu_si128((__m128i *) (output_ptr - i - usable_vec_half_stride), rev_vec_l);
    }

    data[0] -= i;
    data[1] += i;
  }
#endif
  return i;
}

void cpu_hflip_channels_last_vec(at::TensorIterator& iter) {

  auto input_strides = iter.strides(1);
  const auto data_stride = input_strides[1];

  // Generate avx mask once
  alignas(hflip_mask_size) auto mdata = generate_vec_hflip_reg_mask(data_stride);

  auto loop2d = [&](char** base, const int64_t *strides, int64_t size0, int64_t size1) {

    // Here ntensors is defined for output and 1 input. But tensor iterator has defined output, input
    // and restrided_input (see aten/src/ATen/native/TensorTransformations.cpp#L64-L66) but we use only
    // output and input.
    static constexpr int ntensors = 2;
    const int64_t *outer_strides = &strides[3];
    const int64_t stride = strides[0];

    TORCH_INTERNAL_ASSERT(stride == strides[1]);

    auto c = -outer_strides[0];
    TORCH_INTERNAL_ASSERT(c == outer_strides[1]);

    char* C10_RESTRICT data[ntensors] = {base[0], base[1]};
    const int64_t size = size0 * size1;

    int64_t i = 0;

    if (c >= 2 && c <= 16) {
      i = vectorized_cpu_hflip_channels_last(data, size * stride, c, mdata) / stride;
    }

    auto data_stride = size0 * stride;
    for (; i < size; i += size0) {

      memcpy(data[0], data[1], data_stride);

      // advance:
      for (const auto arg : c10::irange(ntensors)) {
        data[arg] += outer_strides[arg];
      }
    }

  };

  int64_t grain_size = at::internal::GRAIN_SIZE;
  iter.for_each(loop2d, grain_size);
  iter.cast_outputs();
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
    auto output_strides = iter.strides(0);
    auto input_strides = iter.strides(1);
    if (iter.ndim() > 0 && output_strides[0] == -iter.element_size(0) && input_strides[0] == iter.element_size(1)) {
      // Special case: horizontal flip with vectorization and input is contiguous
      // Context: horizontal flip leads to strides[0] < 0 and
      // thus is_contiguous condition is not satisfied and non-vectorized code path is taken.
      auto iter_dtype = iter.dtype();
      // Ignoring half and bfloat16 as cpu_hflip_vec is slower than cpu_kernel_vec
      if (isIntegralType(iter_dtype, true) || iter_dtype == kDouble || iter_dtype == kFloat) {
        // Replace AT_DISPATCH_ALL_TYPES_AND by manual if/else due to internal test failures:
        // - "dtype 'Float' not selected for kernel tag hflip_cpu"
        // - "dtype 'Long' not selected for kernel tag hflip_cpu"
        //
        // AT_DISPATCH_ALL_TYPES_AND(kBool,
        //     iter_dtype, "hflip_cpu", [&iter] {
        //       cpu_hflip_vec<scalar_t>(iter);
        // });

        if (iter_dtype == kByte) {
          return cpu_hflip_vec<uint8_t>(iter);
        } else if (iter_dtype == kChar) {
          return cpu_hflip_vec<int8_t>(iter);
        } else if (iter_dtype == kInt) {
          return cpu_hflip_vec<int32_t>(iter);
        } else if (iter_dtype == kLong) {
          return cpu_hflip_vec<int64_t>(iter);
        } else if (iter_dtype == kShort) {
          return cpu_hflip_vec<int16_t>(iter);
        } else if (iter_dtype == kBool) {
          return cpu_hflip_vec<bool>(iter);
        } else if (iter_dtype == kFloat) {
          return cpu_hflip_vec<float>(iter);
        } else if (iter_dtype == kDouble) {
          return cpu_hflip_vec<double>(iter);
        }
      }
      // other dtypes (float16, bfloat16, complex) are handled by cpu_kernel_vec (see below)
    } else if (iter.has_contiguous_first_dim()) {
      // Special cases:
      // a) channels last hflip on (N, C, H, W) and outer_stride(=dtype_size * C) in [2, 16]
      // b) flip dim=-2 on (N, ..., M, C) and outer_stride(=dtype_size * C) in [2, 16]
      auto output_strides_2 = iter.strides(0);
      auto input_strides_2 = iter.strides(1);
      auto c = -output_strides_2[1];
      if (c >= 2 && c <= 16 &&
          c == input_strides_2[1] &&
          c == iter.element_size(0) * iter.shape()[0]  // checks if dim=1 is contiguous as well
      ) {
        return cpu_hflip_channels_last_vec(iter);
      }
      // Special case: vertical flip using memcpy (faster than generic cpu_kernel_vec)
      return cpu_vflip_memcpy(iter);
    }

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

} // namespace at::native
