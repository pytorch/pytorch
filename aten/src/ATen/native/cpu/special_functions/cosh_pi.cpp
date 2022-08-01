#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Generator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
static void cosh_pi_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosh_pi_cpu", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return special_functions::cosh_pi(x);
    });
  });
} // static void cosh_pi_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY
REGISTER_DISPATCH(special_cosh_pi_stub, &CPU_CAPABILITY::cosh_pi_kernel);
} // namespace native
} // namespace at
