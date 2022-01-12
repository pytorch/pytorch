#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Cross.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

namespace at { namespace native {

template <typename T, typename OffsetCalc, typename StrideType>
__global__ void cross_kernel(
    int numel, T* out, const T* x1, const T* x2, OffsetCalc offset_calculator,
    StrideType ostride, StrideType x1stride, StrideType x2stride) {
  CUDA_KERNEL_LOOP(i, numel) {
    const auto offsets = offset_calculator.get(i);
    auto* out_row = out + offsets[0];
    const auto* x1_row = x1 + offsets[1];
    const auto* x2_row = x2 + offsets[2];

    const T val0 = (x1_row[1 * x1stride] * x2_row[2 * x2stride] -
                    x1_row[2 * x1stride] * x2_row[1 * x2stride]);

    const T val1 = (x1_row[2 * x1stride] * x2_row[0 * x2stride] -
                    x1_row[0 * x1stride] * x2_row[2 * x2stride]);

    const T val2 = (x1_row[0 * x1stride] * x2_row[1 * x2stride] -
                    x1_row[1 * x1stride] * x2_row[0 * x2stride]);


    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

void launch_cross_kernel(const TensorIteratorBase& iter, int64_t ostride,
                         int64_t x1stride, int64_t x2stride) {
  const auto N = iter.numel();
  auto offset_calculator = make_element_offset_calculator<3>(iter);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + num_threads() - 1) / num_threads();
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kHalf, iter.common_dtype(), "cross_cuda", [&] {
    auto out = static_cast<scalar_t*>(iter.data_ptr(0));
    auto x1 = static_cast<const scalar_t*>(iter.data_ptr(1));
    auto x2 = static_cast<const scalar_t*>(iter.data_ptr(2));
    constexpr int64_t int_max = std::numeric_limits<int>::max();
    if (ostride * 2 > int_max || x1stride * 2 > int_max || x2stride * 2 > int_max) {
      cross_kernel<<<grid, num_threads(), 0, stream>>>(
          N, out, x1, x2, offset_calculator, ostride, x1stride, x2stride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      cross_kernel<<<grid, num_threads(), 0, stream>>>(
          N, out, x1, x2, offset_calculator,
          static_cast<int>(ostride),
          static_cast<int>(x1stride),
          static_cast<int>(x2stride));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });
}

void cross_impl(Tensor& result, const Tensor& x1, const Tensor& x2, int64_t dim) {
  const int64_t ostride = result.stride(dim);
  const int64_t x1stride = x1.stride(dim);
  const int64_t x2stride = x2.stride(dim);

  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_input(x1)
      .add_input(x2)
      .resize_outputs(false)
      .declare_static_shape(result.sizes(), /*squash_dims=*/dim)
      .build();

  if (iter.numel() == 0) {
    return;
  }

  if (iter.can_use_32bit_indexing()) {
    launch_cross_kernel(iter, ostride, x1stride, x2stride);
  } else {
    for (auto&& sub_iter: iter.with_32bit_indexing()) {
      launch_cross_kernel(sub_iter, ostride, x1stride, x2stride);
    }
  }
}

REGISTER_DISPATCH(cross_stub, &cross_impl);

}}
