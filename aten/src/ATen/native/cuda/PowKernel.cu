#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>

namespace at { namespace native {

namespace {

void pow_tensor_tensor_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "pow_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
      return std::pow(base, exp);
    });
  });
}

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar) {
  const auto exp = exp_scalar.to<double>();
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "pow_cuda", [&]() {
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
      return std::pow((double)base, exp);
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);

}} // namespace at::native
