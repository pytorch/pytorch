#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {
namespace {

static void addcmul_cpu_kernel(TensorIterator& iter, Scalar value) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_ALL_TYPES(dtype, "addcmul_cpu_out", [&] {
    scalar_t scalar_val = value.to<scalar_t>();
    at::native::cpu_kernel(
        iter, [&](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) {
          return self_val + scalar_val * t1_val * t2_val;
        });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(addcmul_stub, &addcmul_cpu_kernel);

} // namespace native
} // namespace at
