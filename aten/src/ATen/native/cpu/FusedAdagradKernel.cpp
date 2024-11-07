#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedAdagrad.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
namespace at::native {

namespace{

template <typename scalar_t, typename opmath_t>
std::enable_if_t<
    std::is_same_v<scalar_t, Half> || std::is_same_v<scalar_t, BFloat16>,
    void>
    inline adagrad_math(
  scalar_t* param_ptr,
  scalar_t* grad_ptr,
  scalar_t* state_sum_ptr,
  const double clr,
  const double eps,
  const double weight_decay,
  const bool maximize,
  const float* grad_scale_ptr,
  int64_t size
){
  using lpVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<opmath_t>;
  int64_t d = 0;
  for (; d < size - (size % lpVec::size()); d += lpVec::size()) {
    lpVec param_lpvec = lpVec::loadu(param_ptr + d);
    auto [param_vec1, param_vec2] = vec::convert_to_float<scalar_t>(param_lpvec);
    lpVec grad_lpvec = lpVec::loadu(grad_ptr + d);
    auto [grad_vec1, grad_vec2] = vec::convert_to_float<scalar_t>(grad_lpvec);
    if (grad_scale_ptr) {
      grad_vec1 = grad_vec1 / fVec(float(*grad_scale_ptr));
      grad_vec2 = grad_vec2 / fVec(float(*grad_scale_ptr));
      lpVec grad_vec_to_store = vec::convert_from_float<scalar_t>(grad_vec1, grad_vec2);
      grad_vec_to_store.store(grad_ptr + d);
    }
    if (maximize){
      grad_vec1 = grad_vec1 * fVec(opmath_t(-1.0));
      grad_vec2 = grad_vec2 * fVec(opmath_t(-1.0));
    }
    if (weight_decay != 0.0){
      grad_vec1 += param_vec1 * fVec(scalar_t(weight_decay));
      grad_vec2 += param_vec2 * fVec(scalar_t(weight_decay));
    }
    auto [state_sum_vec1, state_sum_vec2] = vec::convert_to_float<scalar_t>(lpVec::loadu(state_sum_ptr + d));
    state_sum_vec1 += grad_vec1 * grad_vec1;
    state_sum_vec2 += grad_vec2 * grad_vec2;
    vec::convert_from_float<scalar_t>(state_sum_vec1, state_sum_vec2).store(state_sum_ptr + d);

    fVec std_vec1 = state_sum_vec1.sqrt() + fVec(scalar_t(eps));
    fVec std_vec2 = state_sum_vec2.sqrt() + fVec(scalar_t(eps));
    param_vec1 = param_vec1 - fVec(scalar_t(clr)) * grad_vec1 / std_vec1;
    param_vec2 = param_vec2 - fVec(scalar_t(clr)) * grad_vec2 / std_vec2;
    vec::convert_from_float<scalar_t>(param_vec1, param_vec2).store(param_ptr + d);
  }
  for (; d < size; d++) {
    opmath_t grad_val = grad_ptr[d];
    opmath_t param_val = param_ptr[d];
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / opmath_t(*grad_scale_ptr);
      grad_ptr[d] = grad_val;
    }
    if (maximize) grad_val = -grad_val;
    if (weight_decay != 0.0){
      grad_val += param_val * opmath_t(weight_decay);
    }
    opmath_t state_sum_val = state_sum_ptr[d];
    state_sum_val += grad_val * grad_val;
    state_sum_ptr[d] = state_sum_val;
    opmath_t std_val = std::sqrt(state_sum_val) + opmath_t(eps);
    param_val -= opmath_t(clr) * grad_val / std_val;
    param_ptr[d] = param_val;
  }
}


template <typename scalar_t, typename opmath_t>
std::enable_if_t<
    std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
    void>
    inline adagrad_math(
  scalar_t* param_ptr,
  scalar_t* grad_ptr,
  scalar_t* state_sum_ptr,
  const double clr,
  const double eps,
  const double weight_decay,
  const bool maximize,
  const float* grad_scale_ptr,
  int64_t size
){
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec param_vec = Vec::loadu(param_ptr + d);
    Vec grad_vec = Vec::loadu(grad_ptr + d);
    if (grad_scale_ptr) {
      grad_vec = grad_vec / Vec(scalar_t(*grad_scale_ptr));
      Vec grad_vec_to_store = grad_vec;
      grad_vec_to_store.store(grad_ptr + d);
    }
    if (maximize) grad_vec = grad_vec * Vec(scalar_t(-1.0));
    if (weight_decay != 0.0){
      grad_vec += param_vec * Vec(scalar_t(weight_decay));
    }

    Vec sum_vec = Vec::loadu(state_sum_ptr + d) + grad_vec * grad_vec;
    sum_vec.store(state_sum_ptr + d);

    Vec std_vec = sum_vec.sqrt() + Vec(scalar_t(eps));
    param_vec = param_vec - Vec(scalar_t(clr)) * grad_vec / std_vec;
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
    if (weight_decay != 0.0){
      grad_val += param_ptr[d] * scalar_t(weight_decay);
    }
    state_sum_ptr[d] += grad_val * grad_val;

    scalar_t std_val = std::sqrt(state_sum_ptr[d]) + scalar_t(eps);
    param_ptr[d] -= scalar_t(clr) * grad_val / std_val;
  }
}

template <typename scalar_t>
void adagrad_fused_step_impl(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& state_sum,
    const at::Tensor& state_step,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const float* grad_scale_ptr) {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* grad_data = grad.data_ptr<scalar_t>();
  scalar_t* state_sum_data = state_sum.data_ptr<scalar_t>();
  double step = state_step.item<float>();
  double clr = lr / (1.0 + (step - 1.0) * lr_decay);

  constexpr size_t cache_line_size = 64;
  constexpr int64_t cache_line_aligned_task_unit = cache_line_size / sizeof(scalar_t);
  size_t num_units = divup(param.numel(), cache_line_aligned_task_unit);

  auto adagrad_fn = [&](int64_t begin, int64_t end) {
        // local pointers
        begin *= cache_line_aligned_task_unit;
        end = std::min(end * cache_line_aligned_task_unit, param.numel());
        scalar_t* param_ptr = param_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* state_sum_ptr = state_sum_data + begin;

        const int64_t size = end - begin;
        adagrad_math<scalar_t, opmath_t>(
          param_ptr,
          grad_ptr,
          state_sum_ptr,
          clr,
          eps,
          weight_decay,
          maximize,
          grad_scale_ptr,
          size
        );
      };
  at::parallel_for(
      0, num_units, 0, adagrad_fn);
}

void fused_adagrad_kernel(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& state_sum,
    const at::Tensor& state_step,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const float* grad_scale_ptr
  ) {
  Tensor grad_contiguous = grad.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, param.scalar_type(), "fused_adagrad_kernel", [&] {
    adagrad_fused_step_impl<scalar_t>(
      param,
      grad,
      state_sum,
      state_step,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale_ptr);
  });
}

}

REGISTER_DISPATCH(fused_adagrad_stub, &fused_adagrad_kernel)
} // namespace at::native
