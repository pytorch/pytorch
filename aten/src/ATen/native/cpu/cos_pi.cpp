#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/special_functions.h>
#include <ATen/native/special_functions/cos_pi.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
void cos_pi_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iterator.common_dtype(), "cos_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t z) {
      return at::native::special_functions::cos_pi(z);
    });
  });
} // void cos_pi_cpu_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(special_cos_pi_stub, &CPU_CAPABILITY::cos_pi_cpu_kernel);
} // namespace native
} // namespace at
