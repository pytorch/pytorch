#include <ATen/ATen.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native { namespace {

void addr_kernel(TensorIterator &iter,
                 Scalar beta, Scalar alpha) {
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();
    cpu_kernel(iter,
     [=](scalar_t self_val,
         scalar_t vec1_val,
         scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
        return beta_val * self_val + alpha_val * vec1_val * vec2_val;
      }
    );
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
      iter.dtype(), "addr_cpu", [&]() {
        auto beta_val = beta.to<scalar_t>();
        auto alpha_val = alpha.to<scalar_t>();

        auto beta_vec = Vec256<scalar_t>(beta_val);
        auto alpha_vec = Vec256<scalar_t>(alpha_val);

        cpu_kernel_vec(iter,
          [=](scalar_t self_val,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return beta_val * self_val + alpha_val * vec1_val * vec2_val;
          },
          [=](Vec256<scalar_t> self_vec,
              Vec256<scalar_t> vec1_vec,
              Vec256<scalar_t> vec2_vec) __ubsan_ignore_undefined__ {
            return beta_vec * self_vec + alpha_vec * vec1_vec * vec2_vec;
          }
        );
      }
    );
  }
}

} // anonymous namespace

REGISTER_DISPATCH(addr_stub, &addr_kernel);

}} // namespace at::native
