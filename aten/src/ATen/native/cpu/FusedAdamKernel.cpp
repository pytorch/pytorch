#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedAdam.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
namespace at::native {

namespace{

template <typename scalar_t, typename opmath_t, ADAM_MODE adam_mode>
typename std::enable_if<
    std::is_same<scalar_t, Half>::value || std::is_same<scalar_t, BFloat16>::value,
    void>::
    type inline adam_math(
  scalar_t* param_ptr,
  scalar_t* exp_avg_ptr,
  scalar_t* exp_avg_sq_ptr,
  scalar_t* grad_ptr,
  scalar_t* max_exp_avg_sq_ptr,
  double lr,
  double bias_correction1,
  double bias_correction2,
  double exp_avg_grad_coefficient,
  double exp_avg_sq_grad_coefficient,
  double bias_correction2_sqrt,
  double eps,
  double weight_decay,
  double beta2,
  bool amsgrad,
  bool maximize,
  const float* grad_scale_ptr,
  int64_t size
){
  double step_size = lr / bias_correction1;
  using lpVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<opmath_t>;
  lpVec grad_vec_to_store;
  int64_t d = 0;
  fVec param_vec1, param_vec2;
  fVec grad_vec1, grad_vec2;
  fVec exp_avg_vec1, exp_avg_vec2;
  fVec exp_avg_sq_vec1, exp_avg_sq_vec2;
  fVec max_exp_avg_sq_vec1, max_exp_avg_sq_vec2;
  for (; d < size - (size % lpVec::size()); d += lpVec::size()) {
    lpVec param_lpvec = lpVec::loadu(param_ptr + d);
    std::tie(param_vec1, param_vec2) = vec::convert_to_float<scalar_t>(param_lpvec);
    lpVec grad_lpvec = lpVec::loadu(grad_ptr + d);
    std::tie(grad_vec1, grad_vec2) = vec::convert_to_float<scalar_t>(grad_lpvec);
    if (grad_scale_ptr) {
      grad_vec1 = grad_vec1 / fVec(float(*grad_scale_ptr));
      grad_vec2 = grad_vec2 / fVec(float(*grad_scale_ptr));
      grad_vec_to_store = vec::convert_from_float<scalar_t>(grad_vec1, grad_vec2);
      grad_vec_to_store.store(grad_ptr + d);
    }
    if (maximize){
      grad_vec1 = grad_vec1 * fVec(opmath_t(-1.0));
      grad_vec2 = grad_vec2 * fVec(opmath_t(-1.0));
    }
    if (weight_decay != 0.f){
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_vec1 += param_vec1 * fVec(opmath_t(weight_decay));
        grad_vec2 += param_vec2 * fVec(opmath_t(weight_decay));
       } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_vec1 = param_vec1 * fVec(opmath_t(1 - lr * weight_decay));
        param_vec2 = param_vec2 * fVec(opmath_t(1 - lr * weight_decay));
      }
    }

    lpVec exp_avg_lpvec = lpVec::loadu(exp_avg_ptr + d);
    std::tie(exp_avg_vec1, exp_avg_vec2) = vec::convert_to_float<scalar_t>(exp_avg_lpvec);

    // exp_avg.lerp_(grad, 1 - beta1)
    const fVec lerp_weight = fVec(opmath_t(exp_avg_grad_coefficient));
    auto mask = lerp_weight.abs() < fVec(0.5);
    auto coeff = fVec::blendv(lerp_weight - fVec(1), lerp_weight, mask);

    auto base1 = fVec::blendv(grad_vec1, exp_avg_vec1, mask);
    exp_avg_vec1 = vec::fmadd(coeff, grad_vec1 - exp_avg_vec1, base1);

    auto base2 = fVec::blendv(grad_vec2, exp_avg_vec2, mask);
    exp_avg_vec2 = vec::fmadd(coeff, grad_vec2 - exp_avg_vec2, base2);

    lpVec exp_avg_sq_lpvec = lpVec::loadu(exp_avg_sq_ptr + d);
    std::tie(exp_avg_sq_vec1, exp_avg_sq_vec2) = vec::convert_to_float<scalar_t>(exp_avg_sq_lpvec);
    exp_avg_sq_vec1 = exp_avg_sq_vec1 * fVec(opmath_t(beta2)) +
        fVec(opmath_t(exp_avg_sq_grad_coefficient)) * grad_vec1 * grad_vec1;
    exp_avg_sq_vec2 = exp_avg_sq_vec2 * fVec(opmath_t(beta2)) +
        fVec(opmath_t(exp_avg_sq_grad_coefficient)) * grad_vec2 * grad_vec2;

    vec::convert_from_float<scalar_t>(exp_avg_vec1, exp_avg_vec2).store(exp_avg_ptr + d);
    vec::convert_from_float<scalar_t>(exp_avg_sq_vec1, exp_avg_sq_vec2).store(exp_avg_sq_ptr + d);

    fVec denom_vec1, denom_vec2;
    if (amsgrad) {
      lpVec max_exp_avg_sq_lpvec = lpVec::loadu(max_exp_avg_sq_ptr + d);
      std::tie(max_exp_avg_sq_vec1, max_exp_avg_sq_vec2) = vec::convert_to_float<scalar_t>(max_exp_avg_sq_lpvec);
      max_exp_avg_sq_vec1 = maximum(max_exp_avg_sq_vec1, exp_avg_sq_vec1);
      max_exp_avg_sq_vec2 = maximum(max_exp_avg_sq_vec2, exp_avg_sq_vec2);
      vec::convert_from_float<scalar_t>(max_exp_avg_sq_vec1, max_exp_avg_sq_vec2).store(max_exp_avg_sq_ptr + d);
      denom_vec1 =
          (max_exp_avg_sq_vec1.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
      denom_vec2 =
          (max_exp_avg_sq_vec2.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
    } else {
      denom_vec1 =
          (exp_avg_sq_vec1.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
      denom_vec2 =
          (exp_avg_sq_vec2.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
    }
    param_vec1 = param_vec1 + fVec(opmath_t(-step_size)) * exp_avg_vec1 / denom_vec1;
    param_vec2 = param_vec2 + fVec(opmath_t(-step_size)) * exp_avg_vec2 / denom_vec2;
    vec::convert_from_float<scalar_t>(param_vec1, param_vec2).store(param_ptr + d);
  }
  scalar_t grad_val_to_store;
  for (; d < size; d++) {
    opmath_t grad_val = grad_ptr[d];
    opmath_t param_val = param_ptr[d];
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / float(*grad_scale_ptr);
      grad_val_to_store = scalar_t(grad_val);
      grad_ptr[d] = grad_val_to_store;
    }
    if (maximize) grad_val = -grad_val;
    if (weight_decay != 0.f){
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_val += param_val * opmath_t(weight_decay);
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_val = param_val * opmath_t(1 - lr * weight_decay);
      }
    }
    // exp_avg.lerp_(grad, 1 - beta1)
    opmath_t exp_avg_var = exp_avg_ptr[d];
    auto is_lerp_weight_small = std::abs(opmath_t(exp_avg_grad_coefficient)) < opmath_t(0.5);
    if (is_lerp_weight_small) {
      exp_avg_var = exp_avg_var + opmath_t(exp_avg_grad_coefficient) * (grad_val - exp_avg_var);
    } else {
      exp_avg_var = grad_val - (grad_val - exp_avg_var) * (opmath_t(1) - opmath_t(exp_avg_grad_coefficient));
    }
    exp_avg_ptr[d] = scalar_t(exp_avg_var);
    opmath_t exp_avg_sq_var = exp_avg_sq_ptr[d];
    exp_avg_sq_var = exp_avg_sq_var * opmath_t(beta2);
    exp_avg_sq_var = exp_avg_sq_var +
        opmath_t(exp_avg_sq_grad_coefficient) * grad_val * grad_val;
    exp_avg_sq_ptr[d] = scalar_t(exp_avg_sq_var);
    opmath_t demon_val;
    if (amsgrad) {
      opmath_t max_exp_avg_sq_var = max_exp_avg_sq_ptr[d];
      max_exp_avg_sq_var = std::max(max_exp_avg_sq_var, exp_avg_sq_var);
      max_exp_avg_sq_ptr[d] =
          scalar_t(max_exp_avg_sq_var);
      demon_val =
          std::sqrt(max_exp_avg_sq_var) / opmath_t(bias_correction2_sqrt) + opmath_t(eps);
    } else {
      demon_val = std::sqrt(exp_avg_sq_var) / opmath_t(bias_correction2_sqrt) + opmath_t(eps);
    }
    param_ptr[d] = param_val - opmath_t(step_size) * exp_avg_var / demon_val;
  }
}


template <typename scalar_t, typename opmath_t, ADAM_MODE adam_mode>
typename std::enable_if<
    std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value,
    void>::
    type inline adam_math(
  scalar_t* param_ptr,
  scalar_t* exp_avg_ptr,
  scalar_t* exp_avg_sq_ptr,
  scalar_t* grad_ptr,
  scalar_t* max_exp_avg_sq_ptr,
  double lr,
  double bias_correction1,
  double bias_correction2,
  double exp_avg_grad_coefficient,
  double exp_avg_sq_grad_coefficient,
  double bias_correction2_sqrt,
  double eps,
  double weight_decay,
  double beta2,
  bool amsgrad,
  bool maximize,
  const float* grad_scale_ptr,
  int64_t size
){
  double step_size = lr / bias_correction1;
  using Vec = at::vec::Vectorized<scalar_t>;
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
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_vec += param_vec * Vec(scalar_t(weight_decay));
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_vec = param_vec * Vec(scalar_t(1 - lr * weight_decay));
      }
    }
    Vec exp_avg_vec = Vec::loadu(exp_avg_ptr + d);
    // exp_avg.lerp_(grad, 1 - beta1)
    const Vec lerp_weight = Vec(scalar_t(exp_avg_grad_coefficient));
    auto mask = lerp_weight.abs() < Vec(0.5);
    auto coeff = Vec::blendv(lerp_weight - Vec(1), lerp_weight, mask);
    auto base = Vec::blendv(grad_vec, exp_avg_vec, mask);
    exp_avg_vec = vec::fmadd(coeff, grad_vec - exp_avg_vec, base);

    Vec exp_avg_sq_vec = Vec::loadu(exp_avg_sq_ptr + d) * Vec(scalar_t(beta2)) +
        Vec(scalar_t(exp_avg_sq_grad_coefficient)) * grad_vec * grad_vec;
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
    param_vec = param_vec + Vec(scalar_t(-step_size)) * exp_avg_vec / denom_vec;
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
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_val += param_ptr[d] * scalar_t(weight_decay);
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_ptr[d] = param_ptr[d] * scalar_t(1 - lr * weight_decay);
      }
    }
    // exp_avg.lerp_(grad, 1 - beta1)
    auto is_lerp_weight_small = std::abs(scalar_t(exp_avg_grad_coefficient)) < scalar_t(0.5);
    if (is_lerp_weight_small) {
      exp_avg_ptr[d] = exp_avg_ptr[d] + scalar_t(exp_avg_grad_coefficient) * (grad_val - exp_avg_ptr[d]);
    } else {
      exp_avg_ptr[d] = grad_val - (grad_val - exp_avg_ptr[d]) * (scalar_t(1) - scalar_t(exp_avg_grad_coefficient));
    }
    exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] * scalar_t(beta2);
    exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] +
        scalar_t(exp_avg_sq_grad_coefficient) * grad_val * grad_val;
    scalar_t demon_val;
    if (amsgrad) {
      max_exp_avg_sq_ptr[d] =
          std::max(max_exp_avg_sq_ptr[d], exp_avg_sq_ptr[d]);
      demon_val =
          std::sqrt(max_exp_avg_sq_ptr[d]) / scalar_t(bias_correction2_sqrt) + scalar_t(eps);
    } else {
      demon_val = std::sqrt(exp_avg_sq_ptr[d]) / scalar_t(bias_correction2_sqrt) + scalar_t(eps);
    }
    param_ptr[d] = param_ptr[d] - scalar_t(step_size) * exp_avg_ptr[d] / demon_val;
  }
}


template <typename scalar_t, ADAM_MODE adam_mode>
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
  using opmath_t = at::opmath_type<scalar_t>;
  double step = state_step.item<float>();
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* exp_avg_data = exp_avg.data_ptr<scalar_t>();
  scalar_t* exp_avg_sq_data = exp_avg_sq.data_ptr<scalar_t>();
  scalar_t* max_exp_avg_sq_data = amsgrad ? max_exp_avg_sq.data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_data = grad.data_ptr<scalar_t>();

  // need to use double here to align with non-fused adam
  double bias_correction1 = 1 - std::pow(beta1, step);
  double bias_correction2 = 1 - std::pow(beta2, step);
  double exp_avg_grad_coefficient = 1 - beta1;
  double exp_avg_sq_grad_coefficient = 1 - beta2;
  double bias_correction2_sqrt = std::sqrt(bias_correction2);

  /*
  calc num tasks and grain size for cache line alignment
  For OPENMP, the chunk_size in invoke_parallel is calculated as:
    int64_t chunk_size = divup(ntasks, get_num_threads());
  While in TBB and native backend, the chunk_size in invoke_parallel is calculated as:
    int64_t chunk_size = divup(ntasks, get_num_threads());
    chunk_size = std::max(grain_size, chunk_size);
  Which means for non-OPENMO backene, we can achieve memory alignment by setting a proper grain_size
  to achieve cache line alignment.
  e.g. N_threads=4, nele_per_cache_line=4, ntasks = 412, we can set grain_size=104
  ThreadID     Thread0 Thread1 Thread2 Thread3
  Threadtask   104     104     104     100  -> cache line aligned
  But for OPENMP backend, we need not only change the grain_size, but also the ntasks to
  achieve cache line alignment for main loop.
  ThreadID     Thread0 Thread1 Thread2 Thread3
  Threadtask   103     103     103     103  -> cache line not aligned
  |
  Threadtask   100     100     100     100  -> cachle line aligned (ntask=0-399)
  Threadtask   3       3       3       3    -> cache line not aligned tail (ntask=400-411)
  The usage will be:
    std::tie(cache_line_align_num_tasks, grain_size) =
        at::calc_num_tasks_and_grain_size_for_cache_line_aligment(num_tasks, sizeof(scalar_type));
    at::parallel_for(0, cache_line_align_num_tasks, grain_size, fn);
    // if there are remaining tasks for OPENMP backend, just finish the work
    if (cache_line_align_num_tasks < num_tasks) {
      at::parallel_for(cache_line_align_num_tasks, num_tasks, 0, fn);
    }
  */
  constexpr size_t cacche_line_size = 64;
  size_t num_tasks = param.numel();
  size_t n_task_cache_line_aligned = num_tasks;
  size_t size_type = sizeof(scalar_t);
  size_t num_threads = get_num_threads();
  int64_t chunk_size = divup(num_tasks, num_threads);
  size_t nelement_per_cache_line = cacche_line_size / size_type;
  size_t chunk_size_cache_line_aligned = std::floor(chunk_size / nelement_per_cache_line) * nelement_per_cache_line;
#if AT_PARALLEL_OPENMP
  n_task_cache_line_aligned = chunk_size_cache_line_aligned * num_threads;
#else
if (n_task_cache_line_aligned < num_tasks) {
  chunk_size_cache_line_aligned += nelement_per_cache_line;
}
#endif

  auto adam_fn = [&](int64_t begin, int64_t end) {
        // local pointers
        scalar_t* param_ptr = param_data + begin;
        scalar_t* exp_avg_ptr = exp_avg_data + begin;
        scalar_t* exp_avg_sq_ptr = exp_avg_sq_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* max_exp_avg_sq_ptr = amsgrad ? max_exp_avg_sq_data + begin : nullptr;

        const int64_t size = end - begin;
        adam_math<scalar_t, opmath_t, adam_mode>(
          param_ptr,
          exp_avg_ptr,
          exp_avg_sq_ptr,
          grad_ptr,
          max_exp_avg_sq_ptr,
          lr,
          bias_correction1,
          bias_correction2,
          exp_avg_grad_coefficient,
          exp_avg_sq_grad_coefficient,
          bias_correction2_sqrt,
          eps,
          weight_decay,
          beta2,
          amsgrad,
          maximize,
          grad_scale_ptr,
          size
        );
      };
  at::parallel_for(
      0, n_task_cache_line_aligned, chunk_size_cache_line_aligned, adam_fn);
  if (n_task_cache_line_aligned < num_tasks) {
    at::parallel_for(
        n_task_cache_line_aligned, num_tasks, 0, adam_fn);
  }
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
    const float* grad_scale_ptr,
    const ADAM_MODE adam_mode
  ) {
  Tensor grad_contiguous = grad.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, param.scalar_type(), "fused_adam_kernel", [&] {
    if(adam_mode == ADAM_MODE::ORIGINAL){
      adam_fused_step_impl<scalar_t, ADAM_MODE::ORIGINAL>(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale_ptr);
    } else {
      adam_fused_step_impl<scalar_t, ADAM_MODE::ADAMW>(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale_ptr);
    }

  });
}

}

REGISTER_DISPATCH(fused_adam_stub, &fused_adam_kernel);
} // namespace at::native
