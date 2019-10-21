#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {
namespace {

static void copy_kernel(TensorIterator& iter, bool non_blocking) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND_QINTS_AND3(
    ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, iter.common_dtype(), "copy_",
    [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t a) -> scalar_t { return a; },
        [](Vec256<scalar_t> a) { return a; });
    });
}

} // anonymous namespace

REGISTER_DISPATCH(copy_stub, &copy_kernel);

} // namespace native
} // namespace at
