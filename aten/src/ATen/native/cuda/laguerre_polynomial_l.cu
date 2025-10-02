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
            constexpr char laguerre_polynomial_l_name[] = "laguerre_polynomial_l_forward";

            void laguerre_polynomial_l_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cuda", [&]() {
                    opmath_jitted_gpu_kernel_with_scalars<laguerre_polynomial_l_name, scalar_t, scalar_t>(iterator, laguerre_polynomial_l_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cuda", [&]() {
                    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
                        return laguerre_polynomial_l_forward<scalar_t, true>(x, n);
                    });
                });
#endif
            } // laguerre_polynomial_l_kernel_cuda
        } // namespace (anonymous)

        REGISTER_DISPATCH(laguerre_polynomial_l_stub, &laguerre_polynomial_l_kernel_cuda)
} // namespace at::native
