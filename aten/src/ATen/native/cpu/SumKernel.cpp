#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/cpu/Reduce.h>

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
void store_result(char * C10_RESTRICT data, int64_t stride, int64_t index, scalar_t value) {
  auto * ptr = reinterpret_cast<scalar_t*>(data + index * stride);
  *ptr += value;
}

template <typename scalar_t>
scalar_t row_sum(const char * C10_RESTRICT in_data, const int64_t in_stride, const int64_t size) {
  constexpr int64_t num_levels = 4;
  constexpr int64_t ilp_factor = 4;

  const int64_t level_power =
    std::max(4l, std::lround(std::ceil(std::log2(size) / (num_levels + 1))));
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  scalar_t acc[num_levels][ilp_factor];
  std::fill_n(&acc[0][0], num_levels * ilp_factor, scalar_t(0));

  int64_t i = 0;
  for (; i + level_step * ilp_factor < size;) {
    for (int64_t j = 0; j < level_step; ++j, i += ilp_factor) {
      #pragma unroll
      for (int64_t k = 0; k < ilp_factor; ++k) {
        acc[0][k] += load<scalar_t>(in_data, in_stride, i + k);
      }
    }

    for (int64_t j = 1; j < num_levels; ++j) {
      #pragma unroll
      for (int64_t k = 0; k < ilp_factor; ++k) {
        acc[j][k] += acc[j-1][k];
        acc[j-1][k] = scalar_t(0);
      }

      const auto mask = (level_mask << (j * level_power));
      if ((i & mask) != mask) {
        break;
      }
    }
  }

  for (; i + ilp_factor < size; i += ilp_factor) {
    #pragma unroll
    for (int64_t k = 0; k < ilp_factor; ++k) {
      acc[0][k] += load<scalar_t>(in_data, in_stride, i + k);
    }
  }

  for (; i < size; ++i) {
    acc[0][0] += load<scalar_t>(in_data, in_stride, i);
  }

  for (int64_t i = 1; i < ilp_factor; ++i) {
    for (int64_t j = 0; j < num_levels; ++j) {
      acc[j][0] += acc[j][i];
    }
  }

  scalar_t acc_sum = acc[0][0];
  for (int64_t j = 1; j < num_levels; ++j) {
    acc_sum += acc[j][0];
  }

  return acc_sum;
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
    store_result(data[0], out_stride, j, final_acc);
  }
}

template <typename scalar_t>
void vectorized_outer_sum(
    char * C10_RESTRICT data[2], int64_t inner_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
  using vec_t = Vec256<scalar_t>;
  constexpr int64_t vec_stride = vec_t::size() * sizeof(scalar_t);
  const int64_t vec_size = size1 / vec_t::size();

  // Input is contiguous over the second (non-reduced) dimension
  for (int64_t j = 0; j < vec_size; ++j) {
    const auto *row_in = data[1] + j * vec_stride;
    auto vec_acc = row_sum<vec_t>(row_in, inner_stride, size0);

    scalar_t ans[vec_t::size()];
    vec_acc.store(ans);
    for (int64_t k = 0; k < vec_t::size(); ++k) {
      store_result(data[0], out_stride, j * vec_t::size() + k, ans[k]);
    }
  }

  for (int64_t j = vec_size * vec_t::size(); j < size1; ++j) {
    auto ans = row_sum<scalar_t>(data[1] + j * sizeof(scalar_t), inner_stride, size0);
    store_result(data[0], out_stride, j, ans);
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
          } else {
            // Generic reduction
            for (int64_t j = 0; j < size1; ++j) {
              auto ans = row_sum<scalar_t>(data[1] + j * in_strides[1], in_strides[0], size0);
              store_result(data[0], out_stride, j, ans);
            }
          }
        });
    });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);

}}  // namespace at::native
