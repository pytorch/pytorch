#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/RangeFactories.h>
#include <cmath>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>

#include <ATen/AccumulateType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/Loops.h>

#include <c10/core/Scalar.h>

namespace at::native {
namespace {

using namespace vec;

static void arange_kernel(TensorIterator& iter, const Scalar& scalar_start, const Scalar& scalar_steps, const Scalar& scalar_step) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "arange_cpu", [&]() {
    using accscalar_t = at::acc_type<scalar_t, false>;
    auto start = scalar_start.to<accscalar_t>();
    auto steps = scalar_steps.to<accscalar_t>();
    auto step = scalar_step.to<accscalar_t>();
    at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      int64_t idx(p_begin);
      TensorIterator it(iter);
      cpu_serial_kernel_vec(
          it,
          [start, step, &idx]() -> scalar_t {
            return start + step * (idx++);
          },
          [start, step, &idx]() -> Vectorized<scalar_t> {
            Vectorized<scalar_t> res;
            res = Vectorized<scalar_t>::arange(start + step * idx, step);
            idx += Vectorized<scalar_t>::size();
            return res;
          }, {p_begin, p_end});
    });
  });
}

static void linspace_kernel(TensorIterator& iter, const Scalar& scalar_start, const Scalar& scalar_end, int64_t steps) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, iter.dtype(), "linspace_cpu", [&]() {
    // step should be of double type for all integral types
    using step_t = std::conditional_t<std::is_integral_v<scalar_t>, double, scalar_t>;
    const scalar_t start = scalar_start.to<scalar_t>();
    const scalar_t end = scalar_end.to<scalar_t>();
    // Cast `end` and `start` to `step_t`, since range can be larger than scalar_t for integral types
    const step_t step = (static_cast<step_t>(end) - static_cast<step_t>(start)) / (steps - 1);
    int64_t halfway = steps / 2;
    at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      int64_t idx(p_begin);
      TensorIterator it(iter);
      // Remove vectorization implementation, due to the precision issue between integer and double.
      // Will not harm the performance.
      cpu_serial_kernel(
          it,
          [start, end, step, halfway, steps, &idx]() -> scalar_t {
            if (idx < halfway) {
              return start + step * (idx++);
            } else {
              return end - step * (steps - (idx++) - 1);
            }
          }, {p_begin, p_end});
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(arange_stub, &arange_kernel)
REGISTER_DISPATCH(linspace_stub, &linspace_kernel)

} // namespace at::native
