#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>
#include <c10/core/Scalar.h>

namespace at { namespace native {

template<typename scalar_t>
struct FillFunctor {
  FillFunctor(scalar_t v): value(v) {}
  __device__ __forceinline__ scalar_t operator() () const {
    return value;
  }
  private:
    scalar_t value;
};

void fill_kernel_cuda(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "fill_cuda", [&]() {
    gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
  });
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_cuda);

} // namespace native
} // namespace at
