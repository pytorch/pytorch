#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/special_functions/chebyshev_polynomial_w.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
static void chebyshev_polynomial_w_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_w_cpu", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
      return at::native::special_functions::chebyshev_polynomial_w(x, n);
    });
  });
} // static void chebyshev_polynomial_w_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(chebyshev_polynomial_w_stub, &chebyshev_polynomial_w_kernel);
} // namespace native
} // namespace at
