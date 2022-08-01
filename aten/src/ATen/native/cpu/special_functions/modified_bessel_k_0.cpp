#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Generator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
static void modified_bessel_k_0_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_0_cpu", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return special_functions::modified_bessel_k_0(x);
    });
  });
} // static void modified_bessel_k_0_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY
REGISTER_DISPATCH(special_modified_bessel_k_0_stub, &CPU_CAPABILITY::modified_bessel_k_0_kernel);
} // namespace native
} // namespace at
