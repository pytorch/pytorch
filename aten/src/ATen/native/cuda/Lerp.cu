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
        using opmath_t = at::opmath_type<scalar_t>;
        at::native::gpu_kernel(
            iter,
            [] GPU_LAMBDA(
                scalar_t self_val,
                scalar_t end_val,
                scalar_t weight_val) -> scalar_t {
              opmath_t self_val_f = self_val;
              opmath_t end_val_f = end_val;
              opmath_t weight_val_f = weight_val;
              // Conditional for better numeric. This has been discussed in
              // https://github.com/pytorch/pytorch/pull/18871
              return (std::abs(weight_val_f) < 0.5)
                  ? self_val_f + weight_val_f * (end_val_f - self_val_f)
                  : end_val_f -
                      (end_val_f - self_val_f) *
                          (opmath_t{1} - weight_val_f);
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
              opmath_t self_val_f = self_val;
              opmath_t end_val_f = end_val;
              // Conditional for better numeric. This has been discussed in
              // https://github.com/pytorch/pytorch/pull/18871
              return (std::abs(weight_val) < 0.5)
                  ? self_val_f + weight_val * (end_val_f - self_val_f)
                  : end_val_f -
                      (end_val_f - self_val_f) * (opmath_t{1} - weight_val);
            });
      });
    }
} // anonymous namespace

REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_tensor_kernel);
REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_kernel);

} // namespace native
} // namespace at
