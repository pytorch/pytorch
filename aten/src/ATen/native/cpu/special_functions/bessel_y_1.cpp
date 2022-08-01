#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Generator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
static void bessel_y_1_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_1_cpu", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return special_functions::bessel_y_1(x);
    });
  });
} // static void bessel_y_1_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY
REGISTER_DISPATCH(special_bessel_y_1_stub, &CPU_CAPABILITY::bessel_y_1_kernel);
} // namespace native
} // namespace at
