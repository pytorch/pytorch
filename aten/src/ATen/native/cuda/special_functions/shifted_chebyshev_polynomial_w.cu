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
const auto shifted_chebyshev_polynomial_w_string = jiterator_stringify(
    template<typename T>
    T shifted_chebyshev_polynomial_w(T x, int64_t n) {
        if (n < 0) {
            return T(0.0);
        }

        if (x == T(1.0)) {
            return n + n + 1;
        }

        if (x == T(0.0)) {
            if (n % 2 == 0) {
                return T(1.0);
            }

            return T(-1.0);
        }

        if ((n > 4) && (abs(x + x - T(1.0)) < T(1.0))) {
            if (cos(acos(x + x - T(1.0)) / T(2.0)) != T(1.0)) {
                return sin((n + T(0.5)) * acos(x + x - T(1.0))) / sin(acos(x + x - T(1.0)) / T(2.0));
            }

            if (n % 2 == 0) {
                return T(1.0);
            }

            return T(-1.0);
        }

        if (n == 0) {
            return T(1.0);
        }

        if (n == 1) {
            return x + x - T(1.0) + (x + x - T(1.0)) + T(1.0);
        }

        T p = T(1.0);
        T q = x + x - T(1.0) + (x + x - T(1.0)) + T(1.0);
        T r;

        for (int64_t k = 2; k <= n; k++) {
            r = (x + x - T(1.0) + (x + x - T(1.0))) * q - p;
            p = q;
            q = r;
        }

        return r;
    } // shifted_chebyshev_polynomial_w(T x, int64_t n)

    template<typename T>
    T shifted_chebyshev_polynomial_w(T x, T n) {
        return shifted_chebyshev_polynomial_w(x, static_cast<int64_t>(n));
    } // shifted_chebyshev_polynomial_w(T x, T n)
); // shifted_chebyshev_polynomial_w_string

const char shifted_chebyshev_polynomial_w_name[] = "shifted_chebyshev_polynomial_w";

void shifted_chebyshev_polynomial_w_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_cuda", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<shifted_chebyshev_polynomial_w_name, scalar_t, scalar_t>(iterator, shifted_chebyshev_polynomial_w_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      return shifted_chebyshev_polynomial_w<scalar_t, true>(x, y);
    });
  });
#endif
} // void shifted_chebyshev_polynomial_w_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(shifted_chebyshev_polynomial_w_stub, &shifted_chebyshev_polynomial_w_kernel_cuda);
} // namespace native
} // namespace at
