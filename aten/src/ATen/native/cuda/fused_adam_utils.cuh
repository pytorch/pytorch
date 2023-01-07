#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/Pow.cuh>


namespace at { namespace native {

namespace {

constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kExpAvgIdx = 2;
constexpr uint8_t kExpAvgSqIdx = 3;
constexpr uint8_t kMaxExpAvgSqIdx = 4;

template <typename scalar_type, typename opmath_t, int depth=4>
C10_DEVICE __forceinline__ void adam_math(
    scalar_type r_args[depth][kILP],
    const float* step_count,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const bool amsgrad,
    const float* grad_scale_ptr,
    const float* found_inf_ptr
) {
#pragma unroll
    for (int ii = 0; ii < kILP; ii++) {
        // Load values.
        opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
        opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);
        if (grad_scale_ptr) {
            grad /= (static_cast<double>(*grad_scale_ptr));
        }
        const opmath_t grad_to_store = grad;
        if (maximize) {
            grad = -grad;
        }
        opmath_t exp_avg = static_cast<opmath_t>(r_args[kExpAvgIdx][ii]);
        opmath_t exp_avg_sq = static_cast<opmath_t>(r_args[kExpAvgSqIdx][ii]);
        opmath_t max_exp_avg_sq;
        if (amsgrad) {
            max_exp_avg_sq = static_cast<opmath_t>(r_args[kMaxExpAvgSqIdx][ii]);
        }

        // Update param, grad, 1st and 2nd order momentum.
        if (weight_decay != 0) {
            grad += param * weight_decay;
        }
        // todo(crcrpar): use lerp
        // ref: https://developer.nvidia.com/blog/lerp-faster-cuda/
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad;
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad;

        if (amsgrad) {
            max_exp_avg_sq = std::max(max_exp_avg_sq, exp_avg_sq);
        }

        const opmath_t bias_correction1 = 1 - at::native::pow_(beta1, *step_count);
        const opmath_t bias_correction2 = 1 - at::native::pow_(beta2, *step_count);

        const opmath_t step_size = lr / bias_correction1;

        const opmath_t bias_correction2_sqrt = std::sqrt(bias_correction2);

        opmath_t denom;
        if (amsgrad) {
            denom = (std::sqrt(max_exp_avg_sq) / bias_correction2_sqrt) + eps;
        } else {
            denom = (std::sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps;
        }

        param -= step_size * exp_avg / denom;

        // Store results.
        r_args[kParamIdx][ii] = param;
        if (grad_scale_ptr) {
          r_args[kGradIdx][ii] = grad_to_store;
        }
        r_args[kExpAvgIdx][ii] = exp_avg;
        r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
        if (amsgrad) {
            r_args[kMaxExpAvgSqIdx][ii] = max_exp_avg_sq;
        }
    }
}

// [note: Conditional Gradient Store when `optimizer.step` is called by GradScaler]
// When a user is training their model(s) with an FP16 AMP recipe,
// parameter updates are done via `grad_scaler.step(optimizer)` instead of `optimizer.step()`.
// For most optimizers, GradScaler unscales gradients on behalf of those optimizers.
// Also, before `.step`, it makes sure that all the gradients involved are finite, which incurs a device sync.
// On the other hand, fused optimizers set their member variable of `_step_supports_amp_scaling` to `True`
// in order to remove the device sync above. This means that fused optimizers have to have
// their CUDA kernels (a) unscale gradients and (b) skip parameter updates accordingly.
// To be functionally on par with `torch.optim` optimizers and `_multi_tensor` ones,
// the kernel below writes out gradients only when `grad_scale_ptr != nullptr.
template <typename scalar_type, int depth=4>
struct FusedAdamMathFunctor {
    static_assert(depth == 4 || depth == 5, "depth of 4 for Adam, depth of 5 for Adam with AMSGrad.");
    using opmath_t = at::opmath_type<scalar_type>;
    C10_DEVICE __forceinline__ void operator()(
            int chunk_size,
            FusedOptimizerTensorListMetadata<depth>& tl,
            const double lr,
            const double beta1,
            const double beta2,
            const double weight_decay,
            const double eps,
            const bool maximize,
            const bool amsgrad,
            const float* grad_scale_ptr,
            const float* found_inf_ptr
  ) {
        int tensor_loc = tl.block_to_tensor[blockIdx.x];
        int chunk_idx = tl.block_to_chunk[blockIdx.x];
        int n = tl.numel_for_tensor[tensor_loc];

        if (found_inf_ptr && *found_inf_ptr == 1) {
            return;
        }
        float *step_count = reinterpret_cast<float*>(tl.state_steps_addresses[tensor_loc]);

        scalar_type* args[depth];
        const bool all_aligned{init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
        n -= chunk_idx * chunk_size;
        scalar_type r_args[depth][kILP];

        if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
            for (int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
#pragma unroll
                for (int i = 0; i < depth; i++) {
                    load_store(r_args[i], args[i], 0, i_start);
                }
                adam_math<scalar_type, opmath_t, depth>(
                    r_args, step_count, lr, beta1, beta2, weight_decay, eps, maximize, amsgrad, grad_scale_ptr, found_inf_ptr);
#pragma unroll
                for (int i = 0; i < depth; i++) {
                  if (i != kGradIdx || grad_scale_ptr) {
                    load_store(args[i], r_args[i], i_start, 0);
                  }
                }
            }
        } else {
            for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
              load_args<depth>(r_args, args, i_start, chunk_size, n);
              adam_math<scalar_type, opmath_t, depth>(
                  r_args, step_count, lr, beta1, beta2, weight_decay, eps, maximize, amsgrad, grad_scale_ptr, found_inf_ptr);
#pragma unroll
              for (int i = 0; i < depth; i++) {
                  if (i != kGradIdx || grad_scale_ptr) {
                    store_args(args[i], r_args[i], i_start, chunk_size, n);
                  }
              }
            }
        }
    }
};
} // namespace

}} // namespace at::native
