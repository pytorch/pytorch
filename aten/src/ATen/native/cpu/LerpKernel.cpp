#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Lerp.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {
namespace {

void lerp_scalar_kernel(at::TensorIteratorBase& iter, const Scalar& weight) {
  if (iter.common_dtype() == kBFloat16) {
    using bVec = Vectorized<BFloat16>;
    using fVec = Vectorized<float>;
    float weight_val = weight.to<float>();
    auto weight_vec = fVec(weight_val);
    auto threshold_vec = fVec(0.5);
    auto one_vec = fVec(1);
    at::native::cpu_kernel_vec(
      iter,
      [weight_val](BFloat16 self_val, BFloat16 end_val) -> BFloat16 {
        return lerp(self_val, end_val, weight_val);
      },
      [=](bVec self_vec, bVec end_vec) -> bVec {
          fVec self_vec0, self_vec1, end_vec0, end_vec1;
          std::tie(self_vec0, self_vec1) = convert_bfloat16_float(self_vec);
          std::tie(end_vec0, end_vec1) = convert_bfloat16_float(end_vec);
          auto result0 = fVec::blendv(
            end_vec0 - (end_vec0 - self_vec0) * (one_vec - weight_vec),
            self_vec0 + weight_vec * (end_vec0 - self_vec0),
            weight_vec < threshold_vec);
          auto result1 = fVec::blendv(
            end_vec1 - (end_vec1 - self_vec1) * (one_vec - weight_vec),
            self_vec1 + weight_vec * (end_vec1 - self_vec1),
            weight_vec < threshold_vec);
          return convert_float_bfloat16(result0, result1);
      });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "lerp_kernel_scalar", [&] {
      auto weight_val = weight.to<scalar_t>();
      at::native::cpu_kernel(
          iter,
          [weight_val](scalar_t self_val, scalar_t end_val) {
            return lerp(self_val, end_val, weight_val);
          });
    });
  }
}

void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  if (iter.common_dtype() == kBFloat16) {
    using bVec = Vectorized<BFloat16>;
    using fVec = Vectorized<float>;
    auto one_vec = fVec(1);
    auto threshold_vec = fVec(0.5);
    at::native::cpu_kernel_vec(
      iter,
      [=](BFloat16 self_val, BFloat16 end_val, BFloat16 weight_val) -> BFloat16 {
        return lerp(self_val, end_val, weight_val);
      },
      [=](bVec self_vec, bVec end_vec, bVec weight_vec) -> bVec {
          fVec self_vec0, self_vec1, end_vec0, end_vec1, weight_vec0, weight_vec1;
          std::tie(self_vec0, self_vec1) = convert_bfloat16_float(self_vec);
          std::tie(end_vec0, end_vec1) = convert_bfloat16_float(end_vec);
          std::tie(weight_vec0, weight_vec1) = convert_bfloat16_float(weight_vec);
          auto result0 = fVec::blendv(
            end_vec0 - (end_vec0 - self_vec0) * (one_vec - weight_vec0),
            self_vec0 + weight_vec0 * (end_vec0 - self_vec0),
            weight_vec0 < threshold_vec);
          auto result1 = fVec::blendv(
            end_vec1 - (end_vec1 - self_vec1) * (one_vec - weight_vec1),
            self_vec1 + weight_vec1 * (end_vec1 - self_vec1),
            weight_vec1 < threshold_vec);
          return convert_float_bfloat16(result0, result1);
      });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "lerp_kernel_tensor", [&] {
      at::native::cpu_kernel(
          iter,
          [](scalar_t self_val, scalar_t end_val, scalar_t weight_val) {
            return lerp(self_val, end_val, weight_val);
          });
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_kernel);
REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_tensor_kernel);

} // namespace native
} // namespace at
