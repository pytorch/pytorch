#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Lerp.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/OpMathType.h>

namespace at {
namespace native {
namespace {

void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.common_dtype(), "lerp_cuda",
      [&] {
        at::native::gpu_kernel(
            iter,
            [] GPU_LAMBDA(
                scalar_t self_val,
                scalar_t end_val,
                scalar_t weight_val) -> scalar_t {
              return lerp(self_val, end_val, weight_val);
            });
      });
}

void lerp_scalar_kernel(at::TensorIteratorBase& iter, const c10::Scalar& weight) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.common_dtype(), "lerp_cuda",
      [&]{
        using opmath_t = at::opmath_type<scalar_t>;
        auto weight_val = weight.to<opmath_t>();
        at::native::gpu_kernel(
            iter, [=] GPU_LAMBDA(scalar_t self_val, scalar_t end_val) {
              return lerp(self_val, end_val, weight_val);
            });
      });
    }
} // anonymous namespace

REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_tensor_kernel);
REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_kernel);

} // namespace native
} // namespace at
