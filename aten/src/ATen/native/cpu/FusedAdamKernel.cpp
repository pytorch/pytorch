#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/fused_adam.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
namespace at::native {

namespace{

template <typename scalar_t>
void adam_fused_step_impl(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& max_exp_avg_sq,
    const at::Tensor& state_step,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const float* grad_scale_ptr) {
  int64_t step = state_step.item<int64_t>();
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* exp_avg_data = exp_avg.data_ptr<scalar_t>();
  scalar_t* exp_avg_sq_data = exp_avg_sq.data_ptr<scalar_t>();
  scalar_t* max_exp_avg_sq_data = amsgrad ? max_exp_avg_sq.data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_data = grad.data_ptr<scalar_t>();

  // need to use double here to align with non-fused adam
  double bias_correction1 = 1 - std::pow(beta1, step);
  double step_size =  lr / bias_correction1;
  double bias_correction2 = 1 - std::pow(beta2, step);
  double exp_avg_grad_coefficient = 1 - beta1;
  double exp_avg_sq_grad_coefficient = 1 - beta2;
  double bias_correction2_sqrt = std::sqrt(bias_correction2);

  using Vec = at::vec::Vectorized<scalar_t>;

  // update momentum vt and mt
  // also accumulate sum of param_norm and rtw_norm
  at::parallel_for(
      0, param.numel(), 0, [&](int64_t begin, int64_t end) {
        // local pointers
        scalar_t* param_ptr = param_data + begin;
        scalar_t* exp_avg_ptr = exp_avg_data + begin;
        scalar_t* exp_avg_sq_ptr = exp_avg_sq_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* max_exp_avg_sq_ptr = max_exp_avg_sq_data + begin;

        const int64_t size = end - begin;
        Vec grad_vec_to_store;

        int64_t d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          Vec param_vec = Vec::loadu(param_ptr + d);
          Vec grad_vec = Vec::loadu(grad_ptr + d);
          if (grad_scale_ptr) {
            grad_vec = grad_vec / Vec(scalar_t(*grad_scale_ptr));
            grad_vec_to_store = grad_vec;
            grad_vec_to_store.store(grad_ptr + d);
          }
          if (maximize) grad_vec = grad_vec * Vec(scalar_t(-1.0));
          if (weight_decay != 0.f){
            grad_vec += param_vec * Vec(scalar_t(weight_decay));
          }
          Vec exp_avg_vec = Vec::loadu(exp_avg_ptr + d) * Vec(scalar_t(beta1)) +
              grad_vec * Vec(scalar_t(exp_avg_grad_coefficient));
          Vec exp_avg_sq_vec = Vec::loadu(exp_avg_sq_ptr + d) * Vec(scalar_t(beta2)) +
              grad_vec * grad_vec * Vec(scalar_t(exp_avg_sq_grad_coefficient));
          exp_avg_vec.store(exp_avg_ptr + d);
          exp_avg_sq_vec.store(exp_avg_sq_ptr + d);

          Vec denom_vec;
          if (amsgrad) {
            Vec max_exp_avg_sq_vec =
                maximum(Vec::loadu(max_exp_avg_sq_ptr + d), exp_avg_sq_vec);
            max_exp_avg_sq_vec.store(max_exp_avg_sq_ptr + d);
            denom_vec =
                (max_exp_avg_sq_vec.sqrt() / Vec(scalar_t(bias_correction2_sqrt))) + Vec(scalar_t(eps));
          } else {
            denom_vec =
                (exp_avg_sq_vec.sqrt() / Vec(scalar_t(bias_correction2_sqrt))) + Vec(scalar_t(eps));
          }

          param_vec = param_vec - Vec(scalar_t(step_size)) * exp_avg_vec / denom_vec;
          param_vec.store(param_ptr + d);
        }
        scalar_t grad_val_to_store;
        for (; d < size; d++) {
          scalar_t grad_val = grad_ptr[d];
          if (grad_scale_ptr) {
            grad_val = grad_ptr[d] / scalar_t(*grad_scale_ptr);
            grad_val_to_store = grad_val;
            grad_ptr[d] = grad_val_to_store;
          }
          if (maximize) grad_val = -grad_val;
          if (weight_decay != 0.f){
            grad_val += param_ptr[d] * weight_decay;
          }
          exp_avg_ptr[d] =
              exp_avg_ptr[d] * beta1 + grad_val * exp_avg_grad_coefficient;
          exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] * beta2 +
              grad_val * grad_val * (exp_avg_sq_grad_coefficient);
          scalar_t demon_val;
          if (amsgrad) {
            max_exp_avg_sq_ptr[d] =
                std::max(max_exp_avg_sq_ptr[d], exp_avg_sq_ptr[d]);
            demon_val =
                std::sqrt(max_exp_avg_sq_ptr[d]) / bias_correction2_sqrt + eps;
          } else {
            demon_val = std::sqrt(exp_avg_sq_ptr[d]) / bias_correction2_sqrt + eps;
          }
          param_ptr[d] = param_ptr[d] - step_size * exp_avg_ptr[d] / demon_val;
        }
      });
}

void fused_adam_kernel(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& max_exp_avg_sq,
    const at::Tensor& state_step,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const float* grad_scale_ptr) {
  Tensor grad_contiguous = grad.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, param.scalar_type(), "fused_adam_kernel", [&] {
    adam_fused_step_impl<scalar_t>(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale_ptr);
  });
}

}

REGISTER_DISPATCH(fused_adam_stub, &fused_adam_kernel);
} // namespace at::native
