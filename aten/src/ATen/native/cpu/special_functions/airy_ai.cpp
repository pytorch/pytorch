#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
static void airy_ai_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cpu", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return special_functions::airy_ai(x);
    });
  });
} // static void airy_ai_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY
REGISTER_DISPATCH(special_airy_ai_stub, &CPU_CAPABILITY::airy_ai_kernel);
} // namespace native
} // namespace at
