#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

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
        static void gamma_kernel(TensorIteratorBase& iterator) {
            TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

            AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gamma_cpu", [&]() {
                cpu_kernel(iterator, [](scalar_t x) {
                    return gamma_forward(x);
                });
            });
        } // gamma_kernel(TensorIteratorBase& iterator)

        REGISTER_DISPATCH(special_gamma_stub, &gamma_kernel);
    } // namespace native
} // namespace at
