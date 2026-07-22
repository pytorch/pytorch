#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at::native {
inline namespace CPU_CAPABILITY {
    static void spherical_bessel_j0_kernel(TensorIteratorBase& iterator) {
        TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j0_cpu", [&]() {
            cpu_kernel(iterator, [](scalar_t x) {
                return spherical_bessel_j0_forward(x);
           });
        });
    } // spherical_bessel_j0_kernel(TensorIteratorBase& iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(special_spherical_bessel_j0_stub, &CPU_CAPABILITY::spherical_bessel_j0_kernel)
} // namespace at::native
