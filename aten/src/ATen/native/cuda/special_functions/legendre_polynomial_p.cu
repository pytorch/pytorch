#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>

namespace at {
namespace native {
namespace {
const auto legendre_polynomial_p_string = jiterator_stringify(
    template<typename T>
    T legendre_polynomial_p(T x, int64_t n) {
        if (n < 0) {
            return T(0.0);
        }

        if (abs(x) == T(1.0)) {
            if (x > T(0.0) || n % 2 == 0) {
                return T(1.0);
            }

            return T(-1.0);
        }

        if (n == 0) {
            return T(1.0);
        }

        if (n == 1) {
            return x;
        }

        T p = T(1.0);
        T q = x;
        T r;

        for (int64_t k = 1; k < n; k++) {
            r = ((k + k + 1) * x * q - k * p) / (k + 1);
            p = q;
            q = r;
        }

        return r;
    } // legendre_polynomial_p(T x, int64_t n)

    template<typename T>
    T legendre_polynomial_p(T x, T n) {
        return legendre_polynomial_p(x, static_cast<int64_t>(n));
    } // legendre_polynomial_p(T x, T n)
); // legendre_polynomial_p_string

const char legendre_polynomial_p_name[] = "legendre_polynomial_p";

void legendre_polynomial_p_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cuda", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<legendre_polynomial_p_name, scalar_t, scalar_t>(iterator, legendre_polynomial_p_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      return legendre_polynomial_p<scalar_t, true>(x, y);
    });
  });
#endif
} // void legendre_polynomial_p_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(legendre_polynomial_p_stub, &legendre_polynomial_p_kernel_cuda);
} // namespace native
} // namespace at
