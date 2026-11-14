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

void smooth_l1_backward_cpu_kernel(TensorIterator& iter, const Scalar& norm, double beta) {
  ScalarType dtype = iter.dtype(0);
  if (at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "smooth_l1_backward_cpu_out", [&]() {
    auto norm_val = norm.to<float>();
    float beta_val(beta);
    auto norm_val_vec = Vectorized<float>(norm_val);
    auto beta_val_vec = Vectorized<float>(beta_val);
    const auto neg_1_vec = Vectorized<float>(-1);
    const auto zero_vec = Vectorized<float>(0);
    const auto pos_1_vec = Vectorized<float>(1);
    cpu_kernel_vec(iter,
      [=](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
        const auto x = float(input) - float(target);
        if (x <= -beta) {
          return -norm_val * float(grad_output);
        }else if (x >= beta){
          return norm_val * float(grad_output);
        }else{
          return norm_val * x * float(grad_output) / beta;
        }
      },
      [norm_val_vec, beta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
         Vectorized<scalar_t> input, Vectorized<scalar_t> target, Vectorized<scalar_t> grad_output) -> Vectorized<scalar_t> {
        // using two blendv calls to simulate the 3 cases
        // 1        if  x >= beta
        // -1       if x <= -beta
        // x / beta if |x| < beta
        auto [input0, input1] = convert_to_float(input);
        auto [target0, target1] = convert_to_float(target);
        auto [grad_output0, grad_output1] = convert_to_float(grad_output);
        auto x = input0 - target0;
        auto pos_or_neg_1_vec = Vectorized<float>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        auto x_abs = x.abs();
        auto output = Vectorized<float>::blendv(
            x / beta_val_vec, pos_or_neg_1_vec, x_abs >= beta_val_vec);
        input0 = norm_val_vec * output * grad_output0;

        x = input1 - target1;
        pos_or_neg_1_vec = Vectorized<float>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        x_abs = x.abs();
        output = Vectorized<float>::blendv(
            x / beta_val_vec, pos_or_neg_1_vec, x_abs >= beta_val_vec);
        input1 = norm_val_vec * output * grad_output1;
        return convert_from_float<scalar_t>(input0, input1);
      }
    );
  });
  } else {
    AT_DISPATCH_ALL_TYPES(dtype, "smooth_l1_backward_cpu_out", [&] {
    auto norm_val = norm.to<scalar_t>();
    scalar_t beta_val(beta);
    auto norm_val_vec = Vectorized<scalar_t>(norm_val);
    auto beta_val_vec = Vectorized<scalar_t>(beta_val);
    const auto neg_1_vec = Vectorized<scalar_t>(-1);
    const auto zero_vec = Vectorized<scalar_t>(0);
    const auto pos_1_vec = Vectorized<scalar_t>(1);
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
         Vectorized<scalar_t> input, Vectorized<scalar_t> target, Vectorized<scalar_t> grad_output) -> Vectorized<scalar_t> {
        // using two blendv calls to simulate the 3 cases
        // 1        if  x >= beta
        // -1       if x <= -beta
        // x / beta if |x| < beta
        const auto x = input - target;
        const auto pos_or_neg_1_vec = Vectorized<scalar_t>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        const auto x_abs = x.abs();
        const auto output = Vectorized<scalar_t>::blendv(
            x / beta_val_vec, pos_or_neg_1_vec, x_abs >= beta_val_vec);
        return norm_val_vec * output * grad_output;
      }
    );
  });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(smooth_l1_backward_stub, &smooth_l1_backward_cpu_kernel)

} // namespace at::native
