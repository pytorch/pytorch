// Ternary and higher-order pointwise operations
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/core/Scalar.h>
#include <ATen/cpu/vec/functional.h>
namespace at::native {
namespace {

void huber_backward_cpu_kernel(TensorIterator& iter, const Scalar& norm, double delta) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "huber_backward_cpu_out", [&] {
    auto norm_val = norm.to<scalar_t>();
    scalar_t delta_val(delta);
    auto norm_val_vec = Vectorized<scalar_t>(norm_val);
    auto delta_val_vec = Vectorized<scalar_t>(delta_val);
    const auto neg_1_vec = Vectorized<scalar_t>(-1);
    const auto zero_vec = Vectorized<scalar_t>(0);
    const auto pos_1_vec = Vectorized<scalar_t>(1);
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
         Vectorized<scalar_t> input, Vectorized<scalar_t> target, Vectorized<scalar_t> grad_output) -> Vectorized<scalar_t> {
        // using two blendv calls to simulate the 3 cases
        // delta     if  x >= delta
        // -delta    if x <= -delta
        // x        if |x| < delta
        const auto x = input - target;
        const auto pos_or_neg_1_vec = Vectorized<scalar_t>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        const auto x_abs = x.abs();
        const auto output = Vectorized<scalar_t>::blendv(
            x, pos_or_neg_1_vec * delta_val_vec, x_abs >= delta_val_vec);
        return norm_val_vec * output * grad_output;
      }
    );
  });
}

void mse_backward_cpu_kernel(TensorIterator& iter, const Scalar& value) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "mse_backward_cpu_out", [&] {
    scalar_t scalar_val = value.to<scalar_t>();
    auto scalar_vec = Vectorized<scalar_t>(scalar_val);
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
          return scalar_val * (self_val - t1_val) * t2_val;
        },
        [=](Vectorized<scalar_t> self_vec,
            Vectorized<scalar_t> t1_vec,
            Vectorized<scalar_t> t2_vec) {
          return scalar_vec * (self_vec - t1_vec) *  t2_vec;
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(huber_backward_stub, &huber_backward_cpu_kernel)
REGISTER_DISPATCH(mse_backward_stub, &mse_backward_cpu_kernel)

} // namespace at::native
