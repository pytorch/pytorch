#include <cmath>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>

#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>


namespace at { namespace native {
namespace {

using namespace vec256;

static void linspace_kernel(TensorIterator& iter, Scalar scalar_start, Scalar scalar_end, int64_t steps) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "linspace_cpu", [&]() {
    // step should be of double type for all integral types
    using step_t = std::conditional_t<std::is_integral<scalar_t>::value, double, scalar_t>;
    int64_t idx(0);
    scalar_t start = scalar_start.to<scalar_t>();
    scalar_t end = scalar_end.to<scalar_t>();
    step_t step = (end - start) / static_cast<step_t>(steps - 1);
    cpu_serial_kernel(
        iter,
        [start, step, &idx]() -> scalar_t {
          return start + static_cast<scalar_t>(idx ++) * step;
        });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(linspace_stub, &linspace_kernel);

}} // namespace at::native
