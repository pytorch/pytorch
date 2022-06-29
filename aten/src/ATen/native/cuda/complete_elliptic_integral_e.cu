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
            const char complete_elliptic_integral_e_name[] = "complete_elliptic_integral_e_forward";

            void complete_elliptic_integral_e_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_e_cuda", [&]() {
                    jitted_gpu_kernel<complete_elliptic_integral_e_name, scalar_t, scalar_t, 1>(iterator, complete_elliptic_integral_e_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_e_cuda", [&]() {
                    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
                        return complete_elliptic_integral_e_forward(a);
                    });
                });
#endif // AT_USE_JITERATOR()
            }
        }

        REGISTER_DISPATCH(special_complete_elliptic_integral_e_stub, &complete_elliptic_integral_e_kernel_cuda);
    } // namespace native
} // namespace at
