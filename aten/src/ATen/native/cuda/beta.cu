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
            const char beta_name[] = "beta";

            void beta_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "beta_cuda", [&]() {
                    opmath_jitted_gpu_kernel_with_scalars<beta_name, scalar_t, scalar_t>(iterator, beta_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "beta_cuda", [&]() {
                    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
                        return beta<scalar_t, true>(x, y);
                    });
                });
#endif
            } // beta_kernel_cuda
        } // namespace (anonymous)

        REGISTER_DISPATCH(beta_stub, &beta_kernel_cuda);
    } // namespace native
} // namespace at
