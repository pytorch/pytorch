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
const auto hermite_polynomial_h_string = jiterator_stringify(
    template<typename T>
    T hermite_polynomial_h(T x, int64_t n) {
        if (n < 0) {
            return T(0.0);
        }

        if (n == 0) {
            return T(1.0);
        }

        if (n == 1) {
            return x + x;
        }

        T p = T(1.0);
        T q = x + x;
        T r;

        for (int64_t k = 2; k < n + n; k += 2) {
            r = (x + x) * q - k * p;
            p = q;
            q = r;
        }

        return r;
    } // hermite_polynomial_h(T x, int64_t n)

    template<typename T>
    T hermite_polynomial_h(T x, T n) {
        return hermite_polynomial_h(x, static_cast<int64_t>(n));
    } // hermite_polynomial_h(T x, T n)
); // hermite_polynomial_h_string

const char hermite_polynomial_h_name[] = "hermite_polynomial_h";

void hermite_polynomial_h_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_h_cuda", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<hermite_polynomial_h_name, scalar_t, scalar_t>(iterator, hermite_polynomial_h_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_h_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      return hermite_polynomial_h<scalar_t, true>(x, y);
    });
  });
#endif
} // void hermite_polynomial_h_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(hermite_polynomial_h_stub, &hermite_polynomial_h_kernel_cuda);
} // namespace native
} // namespace at
