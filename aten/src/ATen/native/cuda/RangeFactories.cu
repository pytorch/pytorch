#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <limits>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

namespace at {
namespace native {

template<typename T, typename accT = T>
struct LinspaceOp {
  __host__ __device__ LinspaceOp(accT start, accT step):
    start_(start), step_(step) { }
  __device__ __forceinline__ T operator()(ptrdiff_t index) {
    accT increment = step_ * static_cast<accT>(index);
    accT value = start_ + increment;
    return static_cast<T>(value);
  }

  const accT start_, step_;
};

template<typename T, typename accT = T>
struct LogspaceOp {
  __host__ __device__ LogspaceOp(accT start, accT step, accT base):
    start_(start), step_(step), base_(base) { }
  __device__ __forceinline__ T operator()(ptrdiff_t index) {
    accT increment = step_ * static_cast<accT>(index);
    accT value = std::pow(base_, start_ + increment);
    return static_cast<T>(value);
  }

  const accT start_, step_, base_;
};

Tensor& linspace_cuda_out(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else {
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "linspace_cuda", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      LinspaceOp<scalar_t> linspace_method(scalar_start, step);
      thrust::device_ptr<scalar_t> data_(r.data_ptr<scalar_t>());
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      auto policy = thrust::cuda::par.on(stream);
      thrust::tabulate(policy, data_, data_ + steps, linspace_method);
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

Tensor& logspace_cuda_out(Tensor& result, Scalar start, Scalar end, int64_t steps, double base) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(std::pow(base, start.to<double>()));
  } else {
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "logspace_cuda", [&]() {
      scalar_t scalar_base = static_cast<scalar_t>(base);
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      LogspaceOp<scalar_t> logspace_method(scalar_start, step, scalar_base);
      thrust::device_ptr<scalar_t> data_(r.data_ptr<scalar_t>());
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      auto policy = thrust::cuda::par.on(stream);
      thrust::tabulate(policy, data_, data_ + steps, logspace_method);
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

Tensor& range_cuda_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, result.scalar_type(), "range_cuda", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
             std::isfinite(static_cast<double>(xend)),
             "unsupported range: ", xstart, " -> ", xend);
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
             "upper bound and larger bound inconsistent with step sign");
    int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);
    if (result.numel() != size) {
      result.resize_({size});
    }
    Tensor r = result.is_contiguous() ? result : result.contiguous();
    LinspaceOp<scalar_t, accscalar_t> linspace_method(xstart, xstep);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<scalar_t> data_ptr(r.data_ptr<scalar_t>());
    thrust::tabulate(policy, data_ptr, data_ptr + size, linspace_method);

    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  });

  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

Tensor& arange_cuda_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, result.scalar_type(), "arange_cuda", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    // we use double precision for (start - end) / step
    // to compute size_d for consistency across devices.
    // The problem with using accscalar_t is that accscalar_t might be float32 on gpu for a float32 scalar_t,
    // but double on cpu for the same,
    // and the effective output size starts differing on CPU vs GPU because of precision issues, which
    // we dont want.
    // the corner-case we do want to take into account is int64_t, which has higher precision than double
    double size_d;
    if (std::is_same<scalar_t, int64_t>::value) {
      size_d = std::ceil(static_cast<double>(end.to<accscalar_t>() - start.to<accscalar_t>())
                         / step.to<accscalar_t>());
    } else {
      size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>())
                         / step.to<double>());
    }

    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
             std::isfinite(static_cast<double>(xend)),
             "unsupported range: ", xstart, " -> ", xend);
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
             "upper bound and larger bound inconsistent with step sign");

    TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
             "invalid size, possible overflow?");
    int64_t size = static_cast<int64_t>(size_d);

    if (result.numel() != size) {
      result.resize_({size});
    }
    Tensor r = result.is_contiguous() ? result : result.contiguous();
    LinspaceOp<scalar_t, accscalar_t> linspace_method(xstart, xstep);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<scalar_t> data_ptr(r.data_ptr<scalar_t>());
    thrust::tabulate(policy, data_ptr, data_ptr + size, linspace_method);

    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  });

  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

}} // namespace at::native
