#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedSGD.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
namespace at::native {

namespace{

template <typename scalar_t, typename opmath_t>
typename std::enable_if<
    std::is_same<scalar_t, Half>::value || std::is_same<scalar_t, BFloat16>::value,
    void>::
    type inline sgd_math(
  scalar_t* param_ptr,
  scalar_t* grad_ptr,
  scalar_t* momentum_buf_ptr,
  const double weight_decay,
  const double momentum,
  const double lr,
  const double dampening,
  const bool nesterov,
  const bool maximize,
  const bool is_first_step,
  const float* grad_scale_ptr,
  int64_t size
){
  using lpVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<opmath_t>;
  lpVec grad_vec_to_store;
  fVec param_vec1, param_vec2;
  fVec grad_vec1, grad_vec2;
  fVec momentum_buffer_vec1, momentum_buffer_vec2;
  int64_t d = 0;
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
    if (weight_decay != 0.0){
      grad_vec1 = vec::fmadd(param_vec1, fVec(scalar_t(weight_decay)), grad_vec1);
      grad_vec2 = vec::fmadd(param_vec2, fVec(scalar_t(weight_decay)), grad_vec2);
    }
    if (momentum != 0.0) {
      fVec momentum_vec1, momentum_vec2;
      if (is_first_step) {
        momentum_vec1 = grad_vec1;
        momentum_vec2 = grad_vec2;
      } else {

        momentum_vec1 = fVec::loadu(momentum_buf_ptr + d) * fVec(scalar_t(momentum));
        momentum_vec2 = fVec::loadu(momentum_buf_ptr + d + fVec::size()) * fVec(scalar_t(momentum));
        momentum_vec1 = vec::fmadd(fVec(scalar_t(1 - dampening)), grad_vec1, momentum_vec1);
        momentum_vec2 = vec::fmadd(fVec(scalar_t(1 - dampening)), grad_vec2, momentum_vec2);
      }
      vec::convert_from_float<scalar_t>(momentum_vec1, momentum_vec2).store(momentum_buf_ptr + d);;
      if (nesterov) {
        grad_vec1 = vec::fmadd(momentum_vec1, fVec(scalar_t(momentum)), grad_vec1);
        grad_vec2 = vec::fmadd(momentum_vec2, fVec(scalar_t(momentum)), grad_vec2);
      } else {
        grad_vec1 = momentum_vec1;
        grad_vec2 = momentum_vec2;
      }
    }
  }
  scalar_t grad_val_to_store;
  for (; d < size; d++) {
    opmath_t grad_val = grad_ptr[d];
    opmath_t param_val = param_ptr[d];
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / opmath_t(*grad_scale_ptr);
      grad_val_to_store = grad_val;
      grad_ptr[d] = grad_val_to_store;
    }
    if (maximize) grad_val = -grad_val;
    if (weight_decay != 0.0){
      grad_val += param_val * opmath_t(weight_decay);
    }
    if (momentum != 0.0) {
      opmath_t momentum_buf_var = momentum_buf_ptr[d];
      if (is_first_step) {
        momentum_buf_var = grad_val;
      } else {
        momentum_buf_var = momentum_buf_var * opmath_t(momentum) +
            grad_val * opmath_t(1 - dampening);
      }
      momentum_buf_ptr[d] = momentum_buf_var;
      if (nesterov) {
        grad_val += momentum_buf_var * opmath_t(momentum);
      } else {
        grad_val = momentum_buf_var;
      }
    }
    param_ptr[d] = param_val - grad_val * opmath_t(lr);
  }
}


template <typename scalar_t, typename opmath_t>
typename std::enable_if<
    std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value,
    void>::
    type inline sgd_math(
  scalar_t* param_ptr,
  scalar_t* grad_ptr,
  scalar_t* momentum_buf_ptr,
  const double weight_decay,
  const double momentum,
  const double lr,
  const double dampening,
  const bool nesterov,
  const bool maximize,
  const bool is_first_step,
  const float* grad_scale_ptr,
  int64_t size
){
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
    if (weight_decay != 0.0){
      grad_vec = vec::fmadd(param_vec, Vec(scalar_t(weight_decay)), grad_vec);
    }
    if (momentum != 0.0) {
      Vec momentum_vec;
      if (is_first_step) {
        momentum_vec = grad_vec;
      } else {
        momentum_vec =
            Vec::loadu(momentum_buf_ptr + d) * Vec(scalar_t(momentum));
        momentum_vec = vec::fmadd(Vec(scalar_t(1 - dampening)), grad_vec, momentum_vec);
      }
      momentum_vec.store(momentum_buf_ptr + d);
      if (nesterov) {
        grad_vec =  vec::fmadd(momentum_vec, Vec(scalar_t(momentum)), grad_vec);
      } else {
        grad_vec = momentum_vec;
      }
    }
    param_vec += grad_vec * Vec(scalar_t(-lr));
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
    if (momentum != 0.0) {
      if (is_first_step) {
        momentum_buf_ptr[d] = grad_val;
      } else {
        momentum_buf_ptr[d] = momentum_buf_ptr[d] * scalar_t(momentum) +
            grad_val * scalar_t(1 - dampening);
      }
      if (nesterov) {
        grad_val += momentum_buf_ptr[d] * scalar_t(momentum);
      } else {
        grad_val = momentum_buf_ptr[d];
      }
    }
    param_ptr[d] -= grad_val * scalar_t(lr);
  }
}

template <typename scalar_t>
void sgd_fused_step_impl(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& momentum_buffer,
    const double weight_decay,
    const double momentum,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr) {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* grad_data = grad.data_ptr<scalar_t>();
  bool has_momentum_buffer = momentum != 0.0;
  scalar_t* momentum_buffer_data = has_momentum_buffer ? momentum_buffer.data_ptr<scalar_t>() : nullptr;

  constexpr size_t cache_line_size = 64;
  constexpr int64_t cache_line_aligned_task_unit = cache_line_size / sizeof(scalar_t);
  size_t num_units = divup(param.numel(), cache_line_aligned_task_unit);

  auto sgd_fn = [&](int64_t begin, int64_t end) {
        // local pointers
        begin *= cache_line_aligned_task_unit;
        end = std::min(end * cache_line_aligned_task_unit, param.numel());
        scalar_t* param_ptr = param_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* momentum_buffer_ptr = has_momentum_buffer ? momentum_buffer_data + begin : nullptr;

        const int64_t size = end - begin;
        sgd_math<scalar_t, opmath_t>(
          param_ptr,
          grad_ptr,
          momentum_buffer_ptr,
          weight_decay,
          momentum,
          lr,
          dampening,
          nesterov,
          maximize,
          is_first_step,
          grad_scale_ptr,
          size
        );
      };
  at::parallel_for(
      0, num_units, 0, sgd_fn);
}

void fused_sgd_kernel(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& momentum_buffer,
    const double weight_decay,
    const double momentum,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr
  ) {
  Tensor grad_contiguous = grad.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, param.scalar_type(), "fused_sgd_kernel", [&] {
    sgd_fused_step_impl<scalar_t>(
      param,
      grad,
      momentum_buffer,
      weight_decay,
      momentum,
      lr,
      dampening,
      nesterov,
      maximize,
      is_first_step,
      grad_scale_ptr);
  });
}

}

REGISTER_DISPATCH(fused_sgd_stub, &fused_sgd_kernel);
} // namespace at::native
