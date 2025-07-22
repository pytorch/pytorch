#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/RangeUtils.h>
#include <cmath>
#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/linspace_native.h>
#include <ATen/ops/logspace_native.h>
#include <ATen/ops/range_native.h>
#endif

#define GPU_LAMBDA __device__ __host__

namespace {

#if defined(USE_ROCM)
constexpr int num_threads() {
  return 128;
}
#else
constexpr int num_threads() {
  return C10_WARP_SIZE * 2;
}
#endif
constexpr int thread_work_size = 1;
constexpr int block_work_size = thread_work_size * num_threads();

template<typename index_t, typename func_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void elementwise_kernel_with_index(index_t N, func_t f, typename function_traits<func_t>::result_type *data) {
  #pragma unroll
  for (int i = 0; i < thread_work_size; i++) {
    index_t idx = block_work_size * blockIdx.x + num_threads() * i + threadIdx.x;
    if (idx < N) {
      data[idx] = f(idx);
    }
  }
}

template<typename func_t>
void gpu_kernel_with_index(at::Tensor &output, func_t f) {
  int64_t N = output.numel();
  if (N == 0) {
    return;
  }
  int64_t grid = (N + block_work_size - 1) / block_work_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = typename function_traits<func_t>::result_type;
  if (N <= std::numeric_limits<int>::max()) {
    elementwise_kernel_with_index<int><<<grid, num_threads(), 0, stream>>>(N, f, output.mutable_data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    elementwise_kernel_with_index<int64_t><<<grid, num_threads(), 0, stream>>>(N, f, output.mutable_data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace

namespace at::native {

Tensor& linspace_cuda_out(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else if (isIntegralType(r.scalar_type(), 0)) {
    AT_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "linspace_cuda", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      // Cast `end` and `start` to `float`, since range can be larger than scalar_t for integral types
      float step = (static_cast<float>(scalar_end) - static_cast<float>(scalar_start)) / (steps - 1);
      const int64_t halfway = steps / 2;
      gpu_kernel_with_index(r, [scalar_start, scalar_end, steps, step, halfway]GPU_LAMBDA(int64_t ind) -> scalar_t {
        if (ind < halfway) {
          return scalar_start + (step * ind);
        }

        return scalar_end - step * (steps - ind - 1);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, r.scalar_type(), "linspace_cuda", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      const int64_t halfway = steps / 2;
      gpu_kernel_with_index(r, [scalar_start, scalar_end, steps, step, halfway]GPU_LAMBDA(int64_t ind) -> scalar_t {
        if (ind < halfway) {
          return scalar_start + (step * ind);
        }

        return scalar_end - step * (steps - ind - 1);
      });
    });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}

Tensor& logspace_cuda_out(const Scalar& start, const Scalar& end, int64_t steps, double base, Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    if (isComplexType(r.scalar_type())){
      r.fill_(std::pow(base, start.to<c10::complex<double>>()));
    } else {
      r.fill_(std::pow(base, start.to<double>()));
    }
  } else if (isIntegralType(r.scalar_type(), 0)) {
    AT_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "logspace_cuda", [&]() {
      float scalar_base = static_cast<float>(base); // Use float to avoid promotion to double
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      float step = static_cast<float>(scalar_end - scalar_start) / (steps - 1);
      const int64_t halfway = steps / 2;
      gpu_kernel_with_index(r, [scalar_start, scalar_end, scalar_base, steps, step, halfway]GPU_LAMBDA(int64_t ind) -> scalar_t {
        if (ind < halfway) {
          return std::pow(scalar_base, scalar_start + step * ind);
        }
        return std::pow(scalar_base, scalar_end - step * (steps - ind - 1));
      });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, r.scalar_type(), "logspace_cuda", [&]() {
      scalar_t scalar_base = static_cast<scalar_t>(base);
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      const int64_t halfway = steps / 2;
      gpu_kernel_with_index(r, [scalar_start, scalar_end, scalar_base, steps, step, halfway]GPU_LAMBDA(int64_t ind) -> scalar_t {
        if (ind < halfway) {
          return std::pow(scalar_base, scalar_start + step * ind);
        }
        return std::pow(scalar_base, scalar_end - step * (steps - ind - 1));
      });
    });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}

Tensor& range_cuda_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, result.scalar_type(), "range_cuda", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    arange_check_bounds(start, end, step);

    int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);

    if (result.numel() != size) {
      result.resize_({size});
    }
    bool is_contiguous = result.is_contiguous();
    Tensor r = !is_contiguous ?  at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

    gpu_kernel_with_index(r, [xstart, xstep]GPU_LAMBDA(int64_t ind) -> scalar_t {
        accscalar_t inc = xstep * static_cast<accscalar_t>(ind);
        accscalar_t val = xstart + inc;
        return static_cast<scalar_t>(val);
    });

    if(!is_contiguous) {
      result.copy_(r);
    }

  });

  return result;
}

Tensor& arange_cuda_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, result.scalar_type(), "arange_cuda", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    arange_check_bounds(start, end, step);

    // we use double precision for (start - end) / step
    // to compute size_d for consistency across devices.
    // The problem with using accscalar_t is that accscalar_t might be float32 on gpu for a float32 scalar_t,
    // but double on cpu for the same,
    // and the effective output size starts differing on CPU vs GPU because of precision issues, which
    // we dont want.
    // the corner-case we do want to take into account is int64_t, which has higher precision than double
    double size_d;
    if constexpr (std::is_same_v<scalar_t, int64_t>) {
      int64_t sgn = (xstep > 0) - (xstep < 0);
      size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
    } else {
      size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>())
                          / step.to<double>());
    }

    TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
              "invalid size, possible overflow?");
    int64_t size = static_cast<int64_t>(size_d);
    int64_t numel = result.numel();

    if (numel != size) {
      if(numel > 0){
        TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(),
                    " is ", numel, " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. "
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
      }
      result.resize_({size});
    }
    bool is_contiguous = result.is_contiguous();
    Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

    gpu_kernel_with_index(r, [xstart, xstep]GPU_LAMBDA(int64_t ind) -> scalar_t {
        accscalar_t inc = xstep * static_cast<accscalar_t>(ind);
        accscalar_t val = xstart + inc;
        return static_cast<scalar_t>(val);
    });

    if(!is_contiguous) {
      result.copy_(r);
    }
  });

  return result;
}

} // namespace at::native
