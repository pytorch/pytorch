#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at::native {
inline namespace CPU_CAPABILITY {
static void airy_ai_kernel(TensorIteratorBase& iterator) {
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x) {
            return airy_ai_forward(x);
        });
    });
} // airy_ai_kernel(TensorIteratorBase& iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(special_airy_ai_stub, &CPU_CAPABILITY::airy_ai_kernel);
} // namespace at::native
