#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>

namespace at {
    namespace native {
        namespace {
            const char hyperbolic_integral_shi_name[] = "hyperbolic_integral_shi_forward";

            void hyperbolic_integral_shi_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_integral_shi_cuda", [&]() {
                    jitted_gpu_kernel<hyperbolic_integral_shi_name, scalar_t, scalar_t, 1>(iterator, hyperbolic_integral_shi_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_integral_shi_cuda", [&]() {
                    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
                        return hyperbolic_integral_shi_forward(a);
                    });
                });
#endif // AT_USE_JITERATOR()
            }
        }

        REGISTER_DISPATCH(special_hyperbolic_integral_shi_stub, &hyperbolic_integral_shi_kernel_cuda);
    } // namespace native
} // namespace at
