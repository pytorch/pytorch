#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/cpu/Reduce.h>


namespace at {
namespace native {

namespace {

template <typename scalar_t>
scalar_t &dereference(char * C10_RESTRICT data, int64_t stride, int64_t index) {
  auto * ptr = data + index * stride;
  return *reinterpret_cast<scalar_t*>(ptr);
}

template <typename scalar_t>
const scalar_t &dereference(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
  auto * ptr = data + index * stride;
  return *reinterpret_cast<const scalar_t*>(ptr);
}


template <typename scalar_t>
scalar_t row_sum(const char * C10_RESTRICT in_data, const int64_t in_stride, const int64_t size) {
  constexpr int64_t num_levels = 4;
  constexpr int64_t ilp_factor = 8;

  const int64_t level_power = std::max(4l, std::lround(std::floor(std::log2(size) / num_levels)));
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  scalar_t acc[num_levels][ilp_factor] = {0};

  int64_t i = 0;
  for (; i + level_step * ilp_factor < size;) {
    for (int64_t j = 0; j < level_step; ++j, i += ilp_factor) {
      #pragma unroll
      for (int64_t k = 0; k < ilp_factor; ++k) {
        acc[0][k] += dereference<scalar_t>(in_data, in_stride, i + k);
      }
    }

    for (int64_t j = 1; j < num_levels; ++j) {
      #pragma unroll
      for (int64_t k = 0; k < ilp_factor; ++k) {
        acc[j][k] += acc[j-1][k];
        acc[j-1][k] = 0;
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
      acc[0][k] += dereference<scalar_t>(in_data, in_stride, i + k);
    }
  }

  for (; i < size; ++i) {
    acc[0][0] += dereference<scalar_t>(in_data, in_stride, i);
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
          // Move reduction to be the 1st dim
          int64_t in_strides[2], out_stride;
          if (strides[0] == 0) {
            out_stride = strides[2];
            in_strides[0] = strides[1];
            in_strides[1] = strides[3];
          } else {
            TORCH_INTERNAL_ASSERT(strides[2] == 0);
            out_stride = strides[0];
            in_strides[0] = strides[3];
            in_strides[1] = strides[1];
            std::swap(size0, size1);
          }

          // TORCH_WARN(strides[0], ", ", strides[2]);
          for (int64_t j = 0; j < size1; ++j) {
            auto ans = row_sum<scalar_t>(data[1] + j * in_strides[1], in_strides[0], size0);
            dereference<scalar_t>(data[0], out_stride, j) += ans;
          }
        });
    });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);

}}  // namespace at::native
