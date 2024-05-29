#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at::native {
inline namespace CPU_CAPABILITY {
    static void scaled_modified_bessel_k0_kernel(TensorIteratorBase& iterator) {
        TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_k0_cpu", [&]() {
            cpu_kernel(iterator, [](scalar_t x) {
                return scaled_modified_bessel_k0_forward(x);
            });
        });
    } // scaled_modified_bessel_k0_kernel(TensorIteratorBase& iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(special_scaled_modified_bessel_k0_stub, &CPU_CAPABILITY::scaled_modified_bessel_k0_kernel);
} // namespace at::native
