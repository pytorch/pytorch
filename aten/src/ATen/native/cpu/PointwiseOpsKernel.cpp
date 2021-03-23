// Ternary and higher-order pointwise operations
#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {
namespace {

static void addcmul_cpu_kernel(TensorIterator& iter, const Scalar& value) {
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

static void addcdiv_cpu_kernel(TensorIterator& iter, const Scalar& value) {
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

static void smooth_l1_backward_cpu_kernel(TensorIterator& iter, const Scalar& norm, double beta) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_ALL_TYPES(dtype, "smooth_l1_backward_cpu_out", [&] {
    auto norm_val = norm.to<scalar_t>();
    scalar_t beta_val(beta);
    auto norm_val_vec = Vec256<scalar_t>(norm_val);
    auto beta_val_vec = Vec256<scalar_t>(beta_val);
    const auto neg_1_vec = Vec256<scalar_t>(-1);
    const auto zero_vec = Vec256<scalar_t>(0);
    const auto pos_1_vec = Vec256<scalar_t>(1);
    cpu_kernel_vec(iter,
      [=](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
        const auto x = input - target;
        if (x <= -beta)
          return -norm_val * grad_output;
        else if (x >= beta)
          return norm_val * grad_output;
        else
          return norm_val * x * grad_output / beta;
      },
      [norm_val_vec, beta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
         Vec256<scalar_t> input, Vec256<scalar_t> target, Vec256<scalar_t> grad_output) -> Vec256<scalar_t> {
        // using two blendv calls to simulate the 3 cases
        // 1        if  x >= beta
        // -1       if x <= -beta
        // x / beta if |x| < beta
        const auto x = input - target;
        const auto pos_or_neg_1_vec = Vec256<scalar_t>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        const auto x_abs = x.abs();
        const auto output = Vec256<scalar_t>::blendv(
            x / beta_val_vec, pos_or_neg_1_vec, x_abs >= beta_val_vec);
        return norm_val_vec * output * grad_output;
      }
    );
  });
}

static void huber_backward_cpu_kernel(TensorIterator& iter, const Scalar& norm, double delta) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "huber_backward_cpu_out", [&] {
    auto norm_val = norm.to<scalar_t>();
    scalar_t delta_val(delta);
    auto norm_val_vec = Vec256<scalar_t>(norm_val);
    auto delta_val_vec = Vec256<scalar_t>(delta_val);
    const auto neg_1_vec = Vec256<scalar_t>(-1);
    const auto zero_vec = Vec256<scalar_t>(0);
    const auto pos_1_vec = Vec256<scalar_t>(1);
    cpu_kernel_vec(iter,
      [=](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
        const auto x = input - target;
        if (x <= -delta) {
          return -norm_val * grad_output * delta;
        } else if (x >= delta) {
          return norm_val * grad_output * delta;
        } else {
          return norm_val * x * grad_output;
        }
      },
      [norm_val_vec, delta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
         Vec256<scalar_t> input, Vec256<scalar_t> target, Vec256<scalar_t> grad_output) -> Vec256<scalar_t> {
        // using two blendv calls to simulate the 3 cases
        // delta     if  x >= delta
        // -delta    if x <= -delta
        // x        if |x| < delta
        const auto x = input - target;
        const auto pos_or_neg_1_vec = Vec256<scalar_t>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        const auto x_abs = x.abs();
        const auto output = Vec256<scalar_t>::blendv(
            x, pos_or_neg_1_vec * delta_val_vec, x_abs >= delta_val_vec);
        return norm_val_vec * output * grad_output;
      }
    );
  });
}

static void mse_backward_cpu_kernel(TensorIterator& iter, const Scalar& value) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_ALL_TYPES(dtype, "mse_backward_cpu_out", [&] {
    scalar_t scalar_val = value.to<scalar_t>();
    auto scalar_vec = Vec256<scalar_t>(scalar_val);
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
          return scalar_val * (self_val - t1_val) * t2_val;
        },
        [=](Vec256<scalar_t> self_vec,
            Vec256<scalar_t> t1_vec,
            Vec256<scalar_t> t2_vec) {
          return scalar_vec * (self_vec - t1_vec) *  t2_vec;
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(addcmul_stub, &addcmul_cpu_kernel);
REGISTER_DISPATCH(addcdiv_stub, &addcdiv_cpu_kernel);
REGISTER_DISPATCH(smooth_l1_backward_stub, &smooth_l1_backward_cpu_kernel);
REGISTER_DISPATCH(huber_backward_stub, &huber_backward_cpu_kernel);
REGISTER_DISPATCH(mse_backward_stub, &mse_backward_cpu_kernel);

} // namespace native
} // namespace at
