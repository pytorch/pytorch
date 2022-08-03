#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/special_functions/elliptic_theta_1.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
static void elliptic_theta_1_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "elliptic_theta_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return at::native::special_functions::elliptic_theta_1(x);
    });
  });
} // void elliptic_theta_1_cpu_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(special_elliptic_theta_1_stub, &CPU_CAPABILITY::elliptic_theta_1_cpu_kernel);
} // namespace native
} // namespace at
