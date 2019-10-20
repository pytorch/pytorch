#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {
namespace {

static void copy_kernel(TensorIterator& iter, bool non_blocking) {
  ScalarType dtype = iter.dtype(0);
  bool can_vectorize = (dtype == iter.dtype(1)) && (
    dtype != ScalarType::Half && dtype != ScalarType::BFloat16 &&
    !isQIntType(dtype) && !isComplexType(dtype));
  if (can_vectorize) {
    AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, dtype, "copy_kernel",
      [&] {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return a; },
            [=](Vec256<scalar_t> a) { return a; });
      });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, dtype, "copy_", [&] {
      cpu_kernel(iter, [](scalar_t a) -> scalar_t { return a; });
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(copy_stub, &copy_kernel);

} // namespace native
} // namespace at
