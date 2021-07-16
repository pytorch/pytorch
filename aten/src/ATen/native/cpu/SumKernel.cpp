#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/utils.h>

#include <algorithm>

namespace at {
namespace native {
namespace {

template <typename scalar_t>
struct LoadPolicy {
  static scalar_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = reinterpret_cast<const scalar_t*>(data + index * stride);
    return *ptr;
  }
};

template <typename scalar_t>
struct LoadPolicy<Vectorized<scalar_t>> {
  static Vectorized<scalar_t> load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = data + index * stride;
    return Vectorized<scalar_t>::loadu(ptr);
  }
};

template <typename scalar_t, typename acc_t>
struct CastLoadPolicy {
  static acc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    const auto val = LoadPolicy<scalar_t>::load(data, stride, index);
    return acc_t(val);
  }
};

template <typename scalar_t>
struct CastLoadPolicy<scalar_t, scalar_t>:
    LoadPolicy<scalar_t> {
};

// For inner sum, load full vec_t then sum partials down to vacc_t size
template <typename vec_t, typename vacc_t>
struct InnerSumCastLoadPolicy {
  using scalar_t = typename vec_t::value_type;
  using acc_t = typename vacc_t::value_type;

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    const auto val = LoadPolicy<vec_t>::load(data, stride, index);
    alignas(64) scalar_t values[vec_t::size()];
    val.store(values);

    constexpr int vstride = vec_t::size() / vacc_t::size();
    alignas(64) acc_t acc[vacc_t::size()];
    for (int i = 0; i < vacc_t::size(); ++i) {
      acc[i] = values[i * vstride];
    }
    for (int k = 1; k < vstride; ++k) {
      for (int i = 0; i < vacc_t::size(); ++i) {
        acc[i] += values[i * vstride + k];
      }
    }

    return vacc_t::loadu(acc);
  }
};

template <typename scalar_t>
struct InnerSumCastLoadPolicy<scalar_t, scalar_t>:
    LoadPolicy<scalar_t> {
};

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
template <>
struct InnerSumCastLoadPolicy<Vectorized<c10::BFloat16>, Vectorized<float>> {
  using vec_t = Vectorized<c10::BFloat16>;
  using vacc_t = Vectorized<float>;

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const __m256i*>(data + stride * index);
    __m256i values = _mm256_loadu_si256(ptr);
    __m256 first, second;
    cvtbf16_fp32(values, first, second);
    return _mm256_add_ps(first, second);
  }
};
#endif

// For outer sum, load a partial vec_t of size vacc_t then cast to vacc_t
template <typename vec_t, typename vacc_t>
struct OuterSumCastLoadPolicy {
  using scalar_t = typename vec_t::value_type;
  using acc_t = typename vacc_t::value_type;

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    const auto val = vec_t::loadu(data + stride * index, vacc_t::size());
    alignas(64) scalar_t values[vec_t::size()];
    val.store(values);

    alignas(64) acc_t acc[vacc_t::size()];
    for (int i = 0; i < vacc_t::size(); ++i) {
      acc[i] = values[i];
    }

    return vacc_t::loadu(acc);
  }
};

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
template <>
struct OuterSumCastLoadPolicy<Vectorized<c10::BFloat16>, Vectorized<float>> {
  using vec_t = Vectorized<c10::BFloat16>;
  using vacc_t = Vectorized<float>;

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const __m128i*>(data + stride * index);
    __m128i bf_vals = _mm_loadu_si128(ptr);
    __m256 f_vals;
    cvtbf16_fp32(bf_vals, f_vals);
    return f_vals;
  }
};
#endif

template <typename scalar_t>
struct OuterSumCastLoadPolicy<scalar_t, scalar_t>:
    LoadPolicy<scalar_t> {
};

template <typename scalar_t, typename acc_t>
struct CastStoreAccumulate {
  static void store(char * C10_RESTRICT data, int64_t stride, int64_t index, acc_t value) {
    auto * ptr = reinterpret_cast<scalar_t*>(data + index * stride);
    *ptr += value;
  }
};

template <typename StorePolicy, typename scalar_t>
static void store(char * C10_RESTRICT data, int64_t stride, int64_t index, scalar_t value) {
  StorePolicy::store(data, stride, index, value);
}

template <typename StorePolicy, typename scalar_t, size_t numel>
static void store(char * C10_RESTRICT data, int64_t stride, int64_t index,
                  const std::array<scalar_t, numel> &values) {
  auto *base_ptr = data + stride * index;
  for (size_t k = 0; k < numel; ++k) {
    auto val = values[k];
    StorePolicy::store(base_ptr, stride, k, val);
  }
}

template <typename StorePolicy, typename scalar_t>
static void store(char * C10_RESTRICT data, int64_t stride, int64_t index,
                  const Vectorized<scalar_t> &values) {
  using vec_t = Vectorized<scalar_t>;
  alignas(64) std::array<scalar_t, vec_t::size()> array_values;
  values.store(array_values.data());
  store<StorePolicy>(data, stride, index, array_values);
}

/** Simultaneously sum over n rows at once

This algorithm calculates the sum without loss of precision over large axes. It
does this by chunking the sum into groups of 16 or more elements. The sums of
these chunks are also summed in chunks and so on until there is just a single sum
value remaining. This means only numbers of a similar order of magnitude are
added together, thus minimising rounding errors.

This is done in a single linear pass over the data and with O(1) extra storage.
A simplified recursive implementation would look like this:

  scalar_t row_sum(const scalar_t * data, int64_t n) {
    // Note, in practice the chunk size can increase with n
    // This allows the recursion depth to be limited to O(1).
    constexpr int64_t min_chunk_size = 16;

    scalar_t sum = 0;
    if (n <= min_chunk_size) {
      // Recursive base case, calculate a simple running sum
      for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
      }
      return sum;
    }

    // Recursively sum larger chunks of elements
    const int64_t chunk_size = std::max(divup(n, min_chunk_size), min_chunk_size);
    for (int64_t i = 0; i < n; i += chunk_size) {
      sum += row_sum(data + i, std::min(chunk_size, n - i));
    }
    return sum;
  }
*/
template <typename scalar_t, int64_t nrows, typename LoadPolicy>
std::array<scalar_t, nrows> multi_row_sum(
    const char * C10_RESTRICT in_data,
    const int64_t row_stride,
    const int64_t col_stride,
    const int64_t size) {
  constexpr int64_t num_levels = 4;

  const int64_t level_power =
      std::max(int64_t(4), utils::CeilLog2(size) / num_levels);
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  scalar_t acc[num_levels][nrows];
  std::fill_n(&acc[0][0], num_levels * nrows, scalar_t(0));

  int64_t i = 0;
  for (; i + level_step <= size;) {
    for (int64_t j = 0; j < level_step; ++j, ++i) {
      const char * sum_base = in_data + i * row_stride;
      #if !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (int64_t k = 0; k < nrows; ++k) {
        acc[0][k] += LoadPolicy::load(sum_base, col_stride, k);
      }
    }

    for (int64_t j = 1; j < num_levels; ++j) {
      #if !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (int64_t k = 0; k < nrows; ++k) {
        acc[j][k] += acc[j-1][k];
        acc[j-1][k] = scalar_t(0);
      }

      const auto mask = (level_mask << (j * level_power));
      if ((i & mask) != 0) {
        break;
      }
    }
  }

  for (; i < size; ++i) {
    const char * sum_base = in_data + i * row_stride;
    #if !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t k = 0; k < nrows; ++k) {
      acc[0][k] += LoadPolicy::load(sum_base, col_stride, k);
    }
  }

  for (int64_t j = 1; j < num_levels; ++j) {
    #if !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t k = 0; k < nrows; ++k) {
      acc[0][k] += acc[j][k];
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<scalar_t, nrows> ret;
  for (int64_t k = 0; k < nrows; ++k) {
    ret[k] = acc[0][k];
  }
  return ret;
}

template <typename scalar_t, typename LoadPolicy>
scalar_t row_sum(const char * C10_RESTRICT in_data,
                 const int64_t in_stride, const int64_t size) {
  constexpr int64_t ilp_factor = 4;

  // Interpret row as a (-1, ilp_factor) shaped array to find partial sums
  const int64_t size_ilp = size / ilp_factor;
  auto partial_sums = multi_row_sum<scalar_t, ilp_factor, LoadPolicy>(
      in_data, in_stride * ilp_factor, in_stride, size_ilp);

  for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
    partial_sums[0] += LoadPolicy::load(in_data, in_stride, i);
  }

  for (int64_t k = 1; k < ilp_factor; ++k) {
    partial_sums[0] += partial_sums[k];
  }

  return partial_sums[0];
}

template <typename scalar_t>
void vectorized_inner_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t outer_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
  using vec_t = Vectorized<scalar_t>;
  using acc_t = at::acc_type<scalar_t, true>;
  using vacc_t = Vectorized<acc_t>;
  using VecLoadPolicy = InnerSumCastLoadPolicy<vec_t, vacc_t>;
  using ScalarLoadPolicy = CastLoadPolicy<scalar_t, acc_t>;
  using StorePolicy = CastStoreAccumulate<scalar_t, acc_t>;
  constexpr int64_t vec_stride = vec_t::size() * sizeof(scalar_t);
  const int64_t vec_size = size0 / vec_t::size();

  // Input is contiguous over the first (reduced) dimension
  for (int64_t j = 0; j < size1; ++j) {
    const auto *row_in = data[1] + j * outer_stride;
    auto vec_acc = row_sum<vacc_t, VecLoadPolicy>(row_in, vec_stride, vec_size);

    acc_t final_acc = 0;
    for (int64_t k = vec_size * vec_t::size(); k < size0; ++k) {
      final_acc += ScalarLoadPolicy::load(row_in, sizeof(scalar_t), k);
    }

    alignas(64) std::array<acc_t, vacc_t::size()> partials{};
    vec_acc.store(partials.data());
    for (size_t k = 0; k < partials.size(); ++k) {
      final_acc += partials[k];
    }
    store<StorePolicy>(data[0], out_stride, j, final_acc);
  }
}

template <typename scalar_t>
void scalar_inner_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {
  using acc_t = at::acc_type<scalar_t, true>;
  using LoadPolicy = CastLoadPolicy<scalar_t, acc_t>;
  using StorePolicy = CastStoreAccumulate<scalar_t, acc_t>;

  for (int64_t j = 0; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto ans = row_sum<acc_t, LoadPolicy>(row_in, in_strides[0], size0);
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

template <typename scalar_t>
void vectorized_outer_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t inner_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
  using vec_t = Vectorized<scalar_t>;
  using acc_t = at::acc_type<scalar_t, true>;
  using vacc_t = Vectorized<acc_t>;
  using VecLoadPolicy = OuterSumCastLoadPolicy<vec_t, vacc_t>;
  using ScalarLoadPolicy = CastLoadPolicy<scalar_t, acc_t>;
  using StorePolicy = CastStoreAccumulate<scalar_t, acc_t>;

  constexpr int64_t vec_stride = vacc_t::size() * sizeof(scalar_t);
  constexpr int64_t nrows = 4;

  // Input is contiguous over the second (non-reduced) dimension
  int64_t j = 0;
  for (; j + nrows * vacc_t::size() <= size1; j += nrows * vacc_t::size()) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    auto sums = multi_row_sum<vacc_t, nrows, VecLoadPolicy>(
        row_in, inner_stride, vec_stride, size0);

    for (int64_t i = 0; i < nrows; ++i) {
      const int64_t base_idx = j + i * vacc_t::size();
      store<StorePolicy>(data[0], out_stride, base_idx, sums[i]);
    }
  }

  for (; j + vacc_t::size() <= size1; j += vacc_t::size()) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    const vacc_t sums = row_sum<vacc_t, VecLoadPolicy>(
        row_in, inner_stride, size0);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    store<StorePolicy>(data[0], out_stride, j, sums);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    auto ans = row_sum<acc_t, ScalarLoadPolicy>(row_in, inner_stride, size0);
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

template <typename scalar_t>
void scalar_outer_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {
  using acc_t = at::acc_type<scalar_t, true>;
  using LoadPolicy = CastLoadPolicy<scalar_t, acc_t>;
  using StorePolicy = CastStoreAccumulate<scalar_t, acc_t>;

  constexpr int64_t nrows = 4;
  int64_t j = 0;
  for (; j + (nrows - 1) < size1; j += nrows) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto sums = multi_row_sum<acc_t, nrows, LoadPolicy>(
        row_in, in_strides[0], in_strides[1], size0);
    store<StorePolicy>(data[0], out_stride, j, sums);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto ans = row_sum<acc_t, LoadPolicy>(
        row_in, in_strides[0], size0);
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

void sum_kernel_impl(TensorIterator &iter) {
  if (isIntegralType(iter.dtype(), /*includeBool=*/ true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(ScalarType::Bool, iter.dtype(), "sum_cpu",
      [&] {
        binary_kernel_reduce_vec(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
            [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a + b; });
      });
    return;
  }

  // Custom floating point sum for better accuracy
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "sum_cpu",
    [&] {
      iter.output().fill_(scalar_t(0));
      iter.parallel_reduce(
        [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
          // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
          int64_t in_strides[] = { strides[1], strides[3] };
          // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
          int64_t out_strides[] = { strides[0], strides[2] };

          // Move reduction to be the 1st dim
          if (out_strides[0] != 0 && out_strides[1] == 0) {
            std::swap(in_strides[0], in_strides[1]);
            std::swap(out_strides[0], out_strides[1]);
            std::swap(size0, size1);
          }

          // Special case? - not a true reduction
          if (out_strides[0] != 0 && out_strides[1] != 0) {
            // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
            int64_t outer_strides[] = { strides[2], strides[3] };
            UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
              // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
              char* ptrs[3] = { data[0], data[0], data[1] };
              // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
              int64_t inner_strides[3] = { strides[0], strides[0], strides[1] };
              basic_loop(ptrs, inner_strides, 0, size0, [](scalar_t a, scalar_t b) { return a + b; });
            });
            return;
          }

          const int64_t out_stride = out_strides[1];
          TORCH_INTERNAL_ASSERT(out_strides[0] == 0);

          if (in_strides[0] == sizeof(scalar_t) && size0 >= Vectorized<scalar_t>::size()) {
            // Contiguous inner reduction
            vectorized_inner_sum<scalar_t>(data, in_strides[1], out_stride, size0, size1);
          } else if (in_strides[1] == sizeof(scalar_t) && size1 >= Vectorized<scalar_t>::size()) {
            // Contiguous outer reduction
            vectorized_outer_sum<scalar_t>(data, in_strides[0], out_stride, size0, size1);
          } else if (in_strides[0] < in_strides[1]) {
            scalar_inner_sum<scalar_t>(data, in_strides, out_stride, size0, size1);
          } else {
            scalar_outer_sum<scalar_t>(data, in_strides, out_stride, size0, size1);
          }
        });
    });
}

}  // namespace (anonymous)

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);

}}  // namespace at::native
