#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/complete_legendre_elliptic_integral_k.h>

#include <cmath>
#include <limits>
#include <type_traits>

#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vml.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/OpMathType.h>

#include <c10/util/math_compat.h>
#include <c10/util/MathConstants.h>
#include <c10/core/Scalar.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
void complete_legendre_elliptic_integral_k_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iterator.common_dtype(), "complete_legendre_elliptic_integral_k_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t k) {
      return at::native::special::complete_legendre_elliptic_integral_k(k);
    });
  });
} // void complete_legendre_elliptic_integral_k_cpu_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(special_complete_legendre_elliptic_integral_k_stub, &CPU_CAPABILITY::complete_legendre_elliptic_integral_k_cpu_kernel);
} // namespace native
} // namespace at
