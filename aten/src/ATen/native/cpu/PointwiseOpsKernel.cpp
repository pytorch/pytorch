// Ternary and higher-order pointwise operations
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
    auto scalar_vec = Vec256<scalar_t>(scalar_val);
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
          return self_val + scalar_val * t1_val * t2_val;
        },
        [=](Vec256<scalar_t> self_vec,
            Vec256<scalar_t> t1_vec,
            Vec256<scalar_t> t2_vec) {
          return self_vec + scalar_vec * t1_vec * t2_vec;
        });
  });
}

static void addcdiv_cpu_kernel(TensorIterator& iter, Scalar value) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_ALL_TYPES(dtype, "addcdiv_cpu_out", [&] {
    scalar_t scalar_val = value.to<scalar_t>();
    auto scalar_vec = Vec256<scalar_t>(scalar_val);
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
          return self_val + scalar_val * t1_val / t2_val;
        },
        [=](Vec256<scalar_t> self_vec,
            Vec256<scalar_t> t1_vec,
            Vec256<scalar_t> t2_vec) {
          return self_vec + scalar_vec * t1_vec / t2_vec;
        });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(addcmul_stub, &addcmul_cpu_kernel);
REGISTER_DISPATCH(addcdiv_stub, &addcdiv_cpu_kernel);

} // namespace native
} // namespace at
