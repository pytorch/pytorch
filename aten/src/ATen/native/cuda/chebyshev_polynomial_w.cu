#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>

namespace at::native {
        namespace {
            CONSTEXPR_EXCEPT_WIN_CUDA char chebyshev_polynomial_w_name[] = "chebyshev_polynomial_w_forward";

            void chebyshev_polynomial_w_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_w_cuda", [&]() {
                    opmath_jitted_gpu_kernel_with_scalars<chebyshev_polynomial_w_name, scalar_t, scalar_t>(iterator, chebyshev_polynomial_w_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_w_cuda", [&]() {
                    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
                        return chebyshev_polynomial_w_forward<scalar_t, true>(x, n);
                    });
                });
#endif
            } // chebyshev_polynomial_w_kernel_cuda
        } // namespace (anonymous)

        REGISTER_DISPATCH(chebyshev_polynomial_w_stub, &chebyshev_polynomial_w_kernel_cuda);
} // namespace at::native
