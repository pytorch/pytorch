#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/special_functions/shifted_chebyshev_polynomial_v.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
void shifted_chebyshev_polynomial_v_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_v_cpu", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
      return at::native::special_functions::shifted_chebyshev_polynomial_v(x, n);
    });
  });
} // static void shifted_chebyshev_polynomial_v_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(shifted_chebyshev_polynomial_v_stub, &shifted_chebyshev_polynomial_v_kernel);
} // namespace native
} // namespace at
