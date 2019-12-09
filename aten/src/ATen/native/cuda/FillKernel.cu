#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>

namespace at { namespace native {

void fill_kernel_cuda(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "fill_cuda", [&]() {
    auto value_converted = value.to<scalar_t>();
    gpu_kernel(iter, [value_converted]GPU_LAMBDA() -> scalar_t {
      return value_converted;
    });
  });
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_cuda);

} // namespace native
} // namespace at
