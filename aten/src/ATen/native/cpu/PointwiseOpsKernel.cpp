// Ternary and higher-order pointwise operations
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/core/Scalar.h>
#include <ATen/cpu/vec/functional.h>
namespace at::native {
namespace {

void addcmul_cpu_kernel(TensorIteratorBase& iter, const Scalar& value) {
  ScalarType dtype = iter.common_dtype();
  if (at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "addcmul_cpu_out", [&]() {
      float float_val = value.to<float>();
      auto float_vec = Vectorized<float>(float_val);
      cpu_kernel_vec(
          iter,
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return float(self_val) + float_val * float(t1_val) * float(t2_val);
          },
          [=](Vectorized<scalar_t> self_vec,
            Vectorized<scalar_t> t1_vec,
            Vectorized<scalar_t> t2_vec) -> Vectorized<scalar_t> {
            auto [self_vec0, self_vec1] = convert_to_float<scalar_t>(self_vec);
            auto [t1_vec0, t1_vec1] = convert_to_float<scalar_t>(t1_vec);
            auto [t2_vec0, t2_vec1] = convert_to_float<scalar_t>(t2_vec);
            self_vec0 = self_vec0 + float_vec * t1_vec0 * t2_vec0;
            self_vec1 = self_vec1 + float_vec * t1_vec1 * t2_vec1;
            return convert_from_float<scalar_t>(self_vec0, self_vec1);
          });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::ComplexHalf,
                                           dtype, "addcmul_cpu_out", [&] {
      scalar_t scalar_val = value.to<scalar_t>();
      auto scalar_vec = Vectorized<scalar_t>(scalar_val);
      cpu_kernel_vec(
          iter,
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return self_val + scalar_val * t1_val * t2_val;
          },
          [=](Vectorized<scalar_t> self_vec,
              Vectorized<scalar_t> t1_vec,
              Vectorized<scalar_t> t2_vec) {
            return self_vec + scalar_vec * t1_vec * t2_vec;
          });
    });
  }
}

void addcdiv_cpu_kernel(TensorIteratorBase& iter, const Scalar& value) {
  ScalarType dtype = iter.common_dtype();
  if (at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "addcdiv_cpu_out", [&]() {
      float float_val = value.to<float>();
      auto float_vec = Vectorized<float>(float_val);
      cpu_kernel_vec(
          iter,
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return float(self_val) + float_val * float(t1_val) / float(t2_val);
          },
          [=](Vectorized<scalar_t> self_vec,
              Vectorized<scalar_t> t1_vec,
              Vectorized<scalar_t> t2_vec) -> Vectorized<scalar_t> {
              auto [self_vec0, self_vec1] = convert_to_float<scalar_t>(self_vec);
              auto [t1_vec0, t1_vec1] = convert_to_float<scalar_t>(t1_vec);
              auto [t2_vec0, t2_vec1] = convert_to_float<scalar_t>(t2_vec);
              self_vec0 = self_vec0 + float_vec * t1_vec0 / t2_vec0;
              self_vec1 = self_vec1 + float_vec * t1_vec1 / t2_vec1;
              return convert_from_float<scalar_t>(self_vec0, self_vec1);
          });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "addcdiv_cpu_out", [&] {
      scalar_t scalar_val = value.to<scalar_t>();
      auto scalar_vec = Vectorized<scalar_t>(scalar_val);
      cpu_kernel_vec(
          iter,
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return self_val + scalar_val * t1_val / t2_val;
          },
          [=](Vectorized<scalar_t> self_vec,
              Vectorized<scalar_t> t1_vec,
              Vectorized<scalar_t> t2_vec) {
            return self_vec + scalar_vec * t1_vec / t2_vec;
          });
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(addcmul_stub, &addcmul_cpu_kernel)
REGISTER_DISPATCH(addcdiv_stub, &addcdiv_cpu_kernel)

} // namespace at::native
