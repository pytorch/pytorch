#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

namespace at::native {

namespace {

constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kStateSumIdx = 2;

template <typename scalar_t, typename opmath_t>
C10_DEVICE inline void adagrad_math(
    scalar_t r_args[3][kILP],
    const double& corrected_lr,
    const double& weight_decay,
    const double& eps,
    const bool& maximize,
    const float* grad_scale_ptr,
    const float* found_inf_ptr) {
#pragma unroll
  for (int ii = 0; ii < kILP; ++ii) {
    opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
    opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);
    opmath_t state_sum = static_cast<opmath_t>(r_args[kStateSumIdx][ii]);

    if (grad_scale_ptr) {
      grad /= (static_cast<double>(*grad_scale_ptr));
    }
    const opmath_t grad_to_store = grad;
    if (maximize) {
      grad = -grad;
    }
    if (weight_decay != 0) {
      grad += param * weight_decay; // Can I change this to use std::fma?
    }
    state_sum += grad * grad; // Can I change this to use std::fma?
    param = param - corrected_lr * grad / (std::sqrt(state_sum) + eps);

    r_args[kParamIdx][ii] = param;
    if (grad_scale_ptr) {
      r_args[kGradIdx][ii] = grad_to_store;
    }
    r_args[kStateSumIdx][ii] = state_sum;
  }
}

template <typename scalar_t>
struct FusedAdagradMathFunctor {
  using opmath_t = at::opmath_type<scalar_t>;

  C10_DEVICE __forceinline__ void operator()(
      int chunk_size,
      FusedOptimizerTensorListMetadata<3>& tl,
      const float* lr_ptr,
      const double& lr,
      const double& lr_decay,
      const double& weight_decay,
      const double& eps,
      const bool& maximize,
      const float* grad_scale_ptr,
      const float* found_inf_ptr) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    const double lr_double = lr_ptr ? *lr_ptr : lr;

    if (found_inf_ptr && *found_inf_ptr == 1) {
      return;
    }

    const auto corrected_lr = [&]() -> double {
      auto* step_count =
          reinterpret_cast<const float*>(tl.state_steps_addresses[tensor_loc]);
      const auto denom = 1 + (*step_count - 1) * lr_decay;
      const auto corrected_lr = lr_double / denom;
      return corrected_lr;
    }();

    scalar_t* args[3];
    scalar_t r_args[3][kILP];
    const auto n = tl.numel_for_tensor[tensor_loc] -
        static_cast<int64_t>(chunk_idx * chunk_size);

    const bool all_aligned{
        init_args<3>(args, tl, chunk_idx, chunk_size, tensor_loc)};

    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        load_store(r_args[kParamIdx], args[kParamIdx], 0, i_start);
        load_store(r_args[kGradIdx], args[kGradIdx], 0, i_start);
        load_store(r_args[kStateSumIdx], args[kStateSumIdx], 0, i_start);

        adagrad_math<scalar_t, opmath_t>(
            r_args,
            corrected_lr,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);

        load_store(args[kParamIdx], r_args[kParamIdx], i_start, 0);
        load_store(args[kStateSumIdx], r_args[kStateSumIdx], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<3>(r_args, args, i_start, chunk_size, n);

        adagrad_math<scalar_t, opmath_t>(
            r_args,
            corrected_lr,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);

#pragma unroll
        for (int i = 0; i < 3; i++) {
          if (i != kGradIdx || grad_scale_ptr) {
            store_args(args[i], r_args[i], i_start, chunk_size, n);
          }
        }
      }
    }
  }
};

} // namespace

} // namespace at::native