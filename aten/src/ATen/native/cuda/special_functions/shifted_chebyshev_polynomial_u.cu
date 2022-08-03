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
const auto shifted_chebyshev_polynomial_u_string = jiterator_stringify(
    template<typename T>
    T shifted_chebyshev_polynomial_u_forward(T x, int64_t n) {
        if (n < 0) {
            return T(0.0);
        }

        if (x == T(1.0)) {
            return n + 1;
        }

        if (x == T(0.0)) {
            if (n % 2 == 0) {
                return n + 1;
            }

            return -(n + 1);
        }

        if ((n > 6) && (abs(x + x - T(1.0)) < T(1.0))) {
            if (sin(acos(x + x - T(1.0))) != T(0.0)) {
                return sin((n + 1) * acos(x + x - T(1.0))) / sin(acos(x + x - T(1.0)));
            }

            return (n + 1) * cos((n + 1) * acos(x + x - T(1.0))) / (x + x - T(1.0));
        }

        if (n == 0) {
            return T(1.0);
        }

        if (n == 1) {
            return x + x - T(1.0) + (x + x - T(1.0));
        }

        T p = T(1.0);
        T q = x + x - T(1.0) + (x + x - T(1.0));
        T r;

        for (int64_t k = 2; k <= n; k++) {
            r = (x + x - T(1.0) + (x + x - T(1.0))) * q - p;
            p = q;
            q = r;
        }

        return r;
    } // shifted_chebyshev_polynomial_u_forward(T x, int64_t n)

    template<typename T>
    T shifted_chebyshev_polynomial_u_forward(T x, T n) {
        return shifted_chebyshev_polynomial_u_forward(x, static_cast<int64_t>(n));
    } // shifted_chebyshev_polynomial_u_forward(T x, T n)
); // shifted_chebyshev_polynomial_u_string

const char shifted_chebyshev_polynomial_u_name[] = "shifted_chebyshev_polynomial_u_forward";

void shifted_chebyshev_polynomial_u_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cuda", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<shifted_chebyshev_polynomial_u_name, scalar_t, scalar_t>(iterator, shifted_chebyshev_polynomial_u_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
      return shifted_chebyshev_polynomial_u_forward<scalar_t, true>(x, n);
    });
  });
#endif
} // void shifted_chebyshev_polynomial_u_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(shifted_chebyshev_polynomial_u_stub, &shifted_chebyshev_polynomial_u_kernel_cuda);
} // namespace native
} // namespace at
