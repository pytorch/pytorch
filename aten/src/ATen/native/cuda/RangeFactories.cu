#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/Exceptions.h>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

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
  __host__ __device__ LogspaceOp(accT start, accT step):
    start_(start), step_(step) { }
  __device__ __forceinline__ T operator()(ptrdiff_t index) {
    accT increment = step_ * static_cast<accT>(index);
    accT base10 = 10;
    accT value = std::pow(base10, start_ + increment);
    return static_cast<T>(value);
  }

  const accT start_, step_;
};

Tensor& linspace_cuda_out(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  AT_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else {
    AT_DISPATCH_FLOATING_TYPES(r.type(), "linspace", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      LinspaceOp<scalar_t> linspace_method(scalar_start, step);
      thrust::device_ptr<scalar_t> data_(r.data<scalar_t>());
      thrust::tabulate(data_, data_ + steps, linspace_method);
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

Tensor& logspace_cuda_out(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  AT_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(std::pow(10.0, start.to<double>()));
  } else {
    AT_DISPATCH_FLOATING_TYPES(r.type(), "logspace", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      LogspaceOp<scalar_t> logspace_method(scalar_start, step);
      thrust::device_ptr<scalar_t> data_(r.data<scalar_t>());
      thrust::tabulate(data_, data_ + steps, logspace_method);
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

}} // namespace at::native
