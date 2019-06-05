#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <limits>

namespace at { namespace native {

template <typename scalar_t>
void fill_kernel_impl(TensorIterator& iter, Scalar value_scalar) {
  auto value = value_scalar.to<scalar_t>();
  gpu_nullary_kernel(iter, [value]GPU_LAMBDA() -> scalar_t {
    return value;
  });
}

static void fill_kernel_cuda(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Bool, at::ScalarType::Half, iter.dtype(), "fill_cuda", [&]() {
    fill_kernel_impl<scalar_t>(iter, value);
  });
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_cuda);

}}
