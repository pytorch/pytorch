#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>

namespace at::native { namespace {

void addr_kernel(TensorIterator &iter,
                 const Scalar& beta, const Scalar& alpha) {
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      cpu_kernel(iter,
        [=](scalar_t /*self_val*/,
            scalar_t vec1_val,
            scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      cpu_kernel(iter,
        [=](scalar_t self_val,
            scalar_t vec1_val,
            scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
          return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        }
      );
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
    iter.dtype(), "addr_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;

      auto beta_val = beta.to<scalar_t>();
      auto alpha_val = alpha.to<scalar_t>();

      auto beta_vec = Vec(beta_val);
      auto alpha_vec = Vec(alpha_val);

      const scalar_t zero_val(0);
      // when beta == 0, values in self should be ignored,
      // nans and infs in self should not propagate.
      if (beta_val == zero_val) {
        cpu_kernel_vec(iter,
          [=](scalar_t /*self_val*/,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return alpha_val * vec1_val * vec2_val;
          },
          [=](Vec /*self_vec*/,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return alpha_vec * vec1_vec * vec2_vec;
          }
        );
      } else {
        cpu_kernel_vec(iter,
          [=](scalar_t self_val,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return beta_val * self_val + alpha_val * vec1_val * vec2_val;
          },
          [=](Vec self_vec,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return beta_vec * self_vec + alpha_vec * vec1_vec * vec2_vec;
          }
        );
      }
    }
  );
}

} // anonymous namespace

REGISTER_DISPATCH(addr_stub, &addr_kernel);
} // namespace at::native
