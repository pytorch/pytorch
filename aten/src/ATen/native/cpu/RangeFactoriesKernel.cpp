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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "linspace_cpu", [&]() {
    int64_t idx = 0;
    scalar_t start = scalar_start.to<scalar_t>();
    scalar_t end = scalar_end.to<scalar_t>();
    scalar_t step = (end - start) / static_cast<scalar_t>(steps - 1);
    cpu_kernel_vec(
        iter,
        [start, step, &idx]() -> scalar_t {
          scalar_t res = start + idx * step;
          ++ idx;
          return res;
        },
        [start, step, &idx]() -> Vec256<scalar_t> {
          auto res = Vec256<scalar_t>::arange(start + idx * step, step);
          idx += Vec256<scalar_t>::size();
          return res;
        });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(linspace_stub, &linspace_kernel);

}} // namespace at::native
