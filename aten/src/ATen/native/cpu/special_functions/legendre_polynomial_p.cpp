#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/special_functions/legendre_polynomial_p.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
static void legendre_polynomial_p_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cpu", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
      return at::native::special_functions::legendre_polynomial_p(x, n);
    });
  });
} // static void legendre_polynomial_p_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(legendre_polynomial_p_stub, &legendre_polynomial_p_kernel);
} // namespace native
} // namespace at
