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
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "addcmul_cpu_out", [&] {
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
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "addcdiv_cpu_out", [&] {
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

static void smooth_l1_backward_cpu_kernel(TensorIterator& iter, Scalar norm) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_ALL_TYPES(dtype, "smooth_l1_backward_cpu_out", [&] {
    auto norm_val = norm.to<scalar_t>();
    cpu_kernel(iter,
      [=](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
        const auto x = input - target;
        if (x < -1.)
          return -norm_val * grad_output;
        else if (x > 1.)
          return norm_val * grad_output;
        else
          return norm_val * x * grad_output;
      }
    );
  });
}

} // anonymous namespace

REGISTER_DISPATCH(addcmul_stub, &addcmul_cpu_kernel);
REGISTER_DISPATCH(addcdiv_stub, &addcdiv_cpu_kernel);
REGISTER_DISPATCH(smooth_l1_backward_stub, &smooth_l1_backward_cpu_kernel);

} // namespace native
} // namespace at
