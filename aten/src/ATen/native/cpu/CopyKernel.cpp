#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/TypeCast.h>

namespace at {
namespace native {
namespace {

static void copy_kernel(TensorIterator& iter, bool non_blocking) {
  ScalarType dtype = iter.dtype(0);
  if (dtype == iter.dtype(1)) {
    if (dtype == ScalarType::Half) {
      cpu_kernel(iter, [=](at::Half a) -> at::Half { return a; });
    } else if (dtype == ScalarType::BFloat16) {
      cpu_kernel(iter, [=](at::BFloat16 a) -> at::BFloat16 { return a; });
    } else if (isQIntType(dtype)) {
      AT_DISPATCH_QINT_TYPES(dtype, "copy_kernel", [&] {
        cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t {return a; });
      });
    } else if (isComplexType(dtype)) {
      AT_DISPATCH_COMPLEX_TYPES(dtype, "copy_kernel", [&] {
          cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t { return a; });
        });
    } else {
      AT_DISPATCH_ALL_TYPES_AND(
          ScalarType::Bool, dtype, "copy_kernel", [&] {
            cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return a; },
                [=](Vec256<scalar_t> a) { return a; });
          });
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, dtype, "copy_", [&] {
      using dest_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, iter.dtype(1), "copy_", [&] {
        cpu_kernel(iter, c10::static_cast_with_inter_type<dest_t, scalar_t>::apply);
      });
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(copy_stub, &copy_kernel);

} // namespace native
} // namespace at
