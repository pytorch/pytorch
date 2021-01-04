#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <c10/util/llvmMathExtras.h>

#include <algorithm>


namespace at {
namespace native {
namespace {

template <typename scalar_t>
struct LoadImpl {
  static scalar_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = reinterpret_cast<const scalar_t*>(data + index * stride);
    return *ptr;
  }
};

template <typename scalar_t>
struct LoadImpl<Vec256<scalar_t>> {
  static Vec256<scalar_t> load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = data + index * stride;
    return Vec256<scalar_t>::loadu(ptr);
  }
};

template <typename T>
T load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
  return LoadImpl<T>::load(data, stride, index);
}

template <typename scalar_t>
void accumulate_result(char * C10_RESTRICT data, int64_t stride, int64_t index, scalar_t value) {
  auto * ptr = reinterpret_cast<scalar_t*>(data + index * stride);
  *ptr += value;
}

template <typename scalar_t, size_t numel>
void accumulate_result(char * C10_RESTRICT data, int64_t stride, int64_t index,
    const std::array<scalar_t, numel> &values) {
  auto *base_ptr = data + stride * index;
  for (int64_t k = 0; k < numel; ++k) {
    accumulate_result(base_ptr, stride, k, values[k]);
  }
}

int64_t ceil_log2(int64_t x) {
  if (x <= 2) {
    return 1;
  }

  auto ux = static_cast<uint64_t>(x);
  // Last set bit is floor(log2(x)), floor + 1 is ceil
  // except when x is an exact powers of 2, so subtract 1 first
  return static_cast<int64_t>(llvm::findLastSet(ux - 1)) + 1;
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
template <typename scalar_t, int64_t nrows>
std::array<scalar_t, nrows> multi_row_sum(
    const char * C10_RESTRICT in_data,
    const int64_t row_stride,
    const int64_t col_stride,
    const int64_t size) {
  constexpr int64_t num_levels = 4;

  const int64_t level_power =
      std::max(int64_t(4), ceil_log2(size) / num_levels);
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

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
        acc[0][k] += load<scalar_t>(sum_base, col_stride, k);
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
      acc[0][k] += load<scalar_t>(sum_base, col_stride, k);
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

  std::array<scalar_t, nrows> ret;
  for (int64_t k = 0; k < nrows; ++k) {
    ret[k] = acc[0][k];
  }
  return ret;
}

template <typename scalar_t>
scalar_t row_sum(const char * C10_RESTRICT in_data,
                 const int64_t in_stride, const int64_t size) {
  constexpr int64_t ilp_factor = 4;

  // Interpret row as a (-1, ilp_factor) shaped array to find partial sums
  const int64_t size_ilp = size / ilp_factor;
  auto partial_sums = multi_row_sum<scalar_t, ilp_factor>(
      in_data, in_stride * ilp_factor, in_stride, size_ilp);

  for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
    partial_sums[0] += load<scalar_t>(in_data, in_stride, i);
  }

  for (int64_t k = 1; k < ilp_factor; ++k) {
    partial_sums[0] += partial_sums[k];
  }

  return partial_sums[0];
}

template <typename scalar_t>
void vectorized_inner_sum(
    char * C10_RESTRICT data[2], int64_t outer_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
  using vec_t = Vec256<scalar_t>;
  constexpr int64_t vec_stride = vec_t::size() * sizeof(scalar_t);
  const int64_t vec_size = size0 / vec_t::size();

  // Input is contiguous over the first (reduced) dimension
  for (int64_t j = 0; j < size1; ++j) {
    const auto *row_in = data[1] + j * outer_stride;
    auto vec_acc = row_sum<vec_t>(row_in, vec_stride, vec_size);

    scalar_t final_acc = 0;
    for (int64_t k = vec_size * vec_t::size(); k < size0; ++k) {
      final_acc += load<scalar_t>(row_in, sizeof(scalar_t), k);
    }

    scalar_t partials[vec_t::size()];
    vec_acc.store(partials);
    for (int64_t k = 0; k < vec_t::size(); ++k) {
      final_acc += partials[k];
    }
    accumulate_result(data[0], out_stride, j, final_acc);
  }
}

template <typename scalar_t>
void scalar_inner_sum(
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {
  for (int64_t j = 0; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    scalar_t ans = row_sum<scalar_t>(row_in, in_strides[0], size0);
    accumulate_result(data[0], out_stride, j, ans);
  }
}

template <typename scalar_t>
void vectorized_outer_sum(
    char * C10_RESTRICT data[2], int64_t inner_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
  using vec_t = Vec256<scalar_t>;
  constexpr int64_t nrows = 4;
  constexpr int64_t vec_stride = vec_t::size() * sizeof(scalar_t);

  // Input is contiguous over the second (non-reduced) dimension
  int64_t j = 0;
  for (; j + nrows * vec_t::size() <= size1; j += nrows * vec_t::size()) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    auto sums = multi_row_sum<vec_t, nrows>(row_in, inner_stride, vec_stride, size0);

    for (int64_t i = 0; i < nrows; ++i) {
      const int64_t base_idx = j + i * vec_t::size();

      std::array<scalar_t, vec_t::size()> ans;
      sums[i].store(ans.data());
      accumulate_result(data[0], out_stride, base_idx, ans);
    }
  }

  for (; j + vec_t::size() <= size1; j += vec_t::size()) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    const vec_t sums = row_sum<vec_t>(row_in, inner_stride, size0);

    std::array<scalar_t, vec_t::size()> ans;
    sums.store(ans.data());
    accumulate_result(data[0], out_stride, j, ans);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    scalar_t ans = row_sum<scalar_t>(row_in, inner_stride, size0);
    accumulate_result(data[0], out_stride, j, ans);
  }
}

template <typename scalar_t>
void scalar_outer_sum(
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {

  constexpr int64_t nrows = 4;
  int64_t j = 0;
  for (; j + (nrows - 1) < size1; j += nrows) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto sums = multi_row_sum<scalar_t, nrows>(
        row_in, in_strides[0], in_strides[1], size0);
    accumulate_result(data[0], out_stride, j, sums);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    scalar_t ans = row_sum<scalar_t>(row_in, in_strides[0], size0);
    accumulate_result(data[0], out_stride, j, ans);
  }
}

void sum_kernel_impl(TensorIterator &iter) {
  if (isIntegralType(iter.dtype(), /*includeBool=*/ true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(ScalarType::Bool, iter.dtype(), "sum_cpu",
      [&] {
        binary_kernel_reduce_vec(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
            [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a + b; });
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
          int64_t in_strides[] = { strides[1], strides[3] };
          int64_t out_strides[] = { strides[0], strides[2] };

          // Move reduction to be the 1st dim
          if (out_strides[0] != 0 && out_strides[1] == 0) {
            std::swap(in_strides[0], in_strides[1]);
            std::swap(out_strides[0], out_strides[1]);
            std::swap(size0, size1);
          }

          // Special case? - not a true reduction
          if (out_strides[0] != 0 && out_strides[1] != 0) {
            int64_t outer_strides[] = { strides[2], strides[3] };
            UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
              char* ptrs[3] = { data[0], data[0], data[1] };
              int64_t inner_strides[3] = { strides[0], strides[0], strides[1] };
              basic_loop(ptrs, inner_strides, 0, size0, [](scalar_t a, scalar_t b) { return a + b; });
            });
            return;
          }

          const int64_t out_stride = out_strides[1];
          TORCH_INTERNAL_ASSERT(out_strides[0] == 0);

          if (in_strides[0] == sizeof(scalar_t) && size0 >= Vec256<scalar_t>::size()) {
            // Contiguous inner reduction
            vectorized_inner_sum<scalar_t>(data, in_strides[1], out_stride, size0, size1);
          } else if (in_strides[1] == sizeof(scalar_t) && size1 >= Vec256<scalar_t>::size()) {
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

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);

}}  // namespace at::native
