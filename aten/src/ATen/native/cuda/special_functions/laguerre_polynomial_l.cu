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
const auto laguerre_polynomial_l_string = jiterator_stringify(
    template<typename T>
    T laguerre_polynomial_l(T x, int64_t n) {
        if (n < 0) {
            return T(0.0);
        }

        if (abs(x) == T(0.0)) {
            return T(1.0);
        }

        if (n == 0) {
            return T(1.0);
        }

        if (n == 1) {
            return T(1.0) - x;
        }

        T p = T(1.0);
        T q = T(1.0) - x;
        T r;

        for (int64_t k = 1; k < n; k++) {
            r = (((k + k) + (T(1.0) - x)) * q - k * p) / (k + 1);
            p = q;
            q = r;
        }

        return r;
    } // laguerre_polynomial_l(T x, int64_t n)

    template<typename T>
    T laguerre_polynomial_l(T x, T n) {
        return laguerre_polynomial_l(x, static_cast<int64_t>(n));
    } // laguerre_polynomial_l(T x, T n)
); // laguerre_polynomial_l_string

const char laguerre_polynomial_l_name[] = "laguerre_polynomial_l";

void laguerre_polynomial_l_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cuda", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<laguerre_polynomial_l_name, scalar_t, scalar_t>(iterator, laguerre_polynomial_l_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      return laguerre_polynomial_l<scalar_t, true>(x, y);
    });
  });
#endif
} // void laguerre_polynomial_l_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(laguerre_polynomial_l_stub, &laguerre_polynomial_l_kernel_cuda);
} // namespace native
} // namespace at
