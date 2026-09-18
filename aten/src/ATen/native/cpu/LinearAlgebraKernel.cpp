#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>

// gcc defaults to -ffp-contract=fast, which fuses the multiply-add in
// `addr_kernel` even across the rounding `c10::Half`'s operators impose. That
// drops a rounding step the reference implementations keep, leaving float16 an
// ulp off on hardware with native fp16. No source-level barrier blocks the
// fusion -- `beta * self + x` is itself a fusable pair -- so this has to be
// file-scoped, and float and double give up their fma as a result. clang and
// MSVC (/fp:strict, see cmake/Codegen.cmake) honour the rounding already.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace at::native { namespace {

// Must mirror whatever the compiler did to the scalar lambda in `addr_kernel`:
// clang contracts the float and double cases but leaves c10::Half alone, MSVC
// contracts nothing under /fp:strict, and gcc contracts nothing thanks to the
// pragma above. This has to be a function rather than an `#if` in the lambda
// itself: the lambda is an AT_DISPATCH macro argument, and a preprocessor
// directive inside a macro argument is undefined -- MSVC rejects it outright.
template <typename T>
inline Vectorized<T> mul_add(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& acc) {
#if defined(_MSC_VER) || (defined(__GNUC__) && !defined(__clang__))
  return a * b + acc;
#else
  return vec::fmadd(a, b, acc);
#endif
}

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
            scalar_t vec2_val) -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      cpu_kernel(iter,
        [=](scalar_t self_val,
            scalar_t vec1_val,
            scalar_t vec2_val) -> scalar_t {
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
            return mul_add(beta_vec, self_vec, alpha_vec * vec1_vec * vec2_vec);
          }
        );
      }
    }
  );
}

} // anonymous namespace

REGISTER_DISPATCH(addr_stub, &addr_kernel)
} // namespace at::native
