#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/special_functions/laguerre_polynomial_l.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
static void laguerre_polynomial_l_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cpu", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
      return at::native::special_functions::laguerre_polynomial_l(x, n);
    });
  });
} // static void laguerre_polynomial_l_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(laguerre_polynomial_l_stub, &laguerre_polynomial_l_kernel);
} // namespace native
} // namespace at
