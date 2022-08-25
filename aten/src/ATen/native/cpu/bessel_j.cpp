#pragma once

#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/bessel_j.h>

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
void bessel_j_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t v, scalar_t z) {
      return at::native::special::bessel_j(v, z);
    });
  });
} // void bessel_j_cpu_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(special_bessel_j_stub, &CPU_CAPABILITY::bessel_j_cpu_kernel);
} // namespace native
} // namespace at
