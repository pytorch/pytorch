#include <ATen/ATen.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>


namespace at { namespace native {

namespace {
constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kExpAvgIdx = 2;
constexpr uint8_t kExpAvgSqIdx = 3;
constexpr uint8_t kMaxExpAvgSqIdx = 4;

template <typename scalar_type, int Depth=4>
struct FusedAdamMathFunctor {
    static_assert(Depth == 4 || Depth == 5, "");
    using opmath_t = at::opmath_type<scalar_type>;
    C10_DEVICE __forceinline__ void operator()(
            int chunk_size,
            FusedOptimizerTensorListMetadata<Depth>& tl,
            const double lr,
            const double beta1,
            const double beta2,
            const double weight_decay,
            const double eps,
            const bool maximize,
            const bool amsgrad,
            const float* inv_grad_scale_ptr,
            const float* found_inf_ptr
  ) {
        int tensor_loc = tl.block_to_tensor[blockIdx.x];
        int chunk_idx = tl.block_to_chunk[blockIdx.x];
        int n = tl.numel_for_tensor[tensor_loc];

        if (found_inf_ptr && *found_inf_ptr == 1) {
            return;
        }
        float *step_count = reinterpret_cast<float*>(tl.state_steps_addresses[tensor_loc]);

        scalar_type* args[Depth];
        const bool all_aligned{init_args<Depth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
        n -= chunk_idx * chunk_size;
        scalar_type r_args[Depth][kILP];

        if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
            for (int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                // Store values into register.
#pragma unroll
                for (int i = 0; i < Depth; i++) {
                    load_store(r_args[i], args[i], 0, i_start);
                }
            // TODO(crcrpar): Dissect this into a `__device__` function.
#pragma unroll
            for (int ii = 0; ii < kILP; ii++) {
                opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
                opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);
                if (inv_grad_scale_ptr) {
                    grad *= (*inv_grad_scale_ptr);
                }
                opmath_t exp_avg = static_cast<opmath_t>(r_args[kExpAvgIdx][ii]);
                opmath_t exp_avg_sq = static_cast<opmath_t>(r_args[kExpAvgSqIdx][ii]);
                opmath_t max_exp_avg_sq;
                if (Depth == kMaxExpAvgSqIdx + 1) {
                    max_exp_avg_sq = static_cast<opmath_t>(r_args[kMaxExpAvgSqIdx][ii]);
                }
                if (weight_decay != 0) {
                    grad += param * weight_decay;
                }
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad;
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad;

                const opmath_t bias_correction1 = 1 - ::pow(beta1, *step_count);
                const opmath_t bias_correction2 = 1 - ::pow(beta2, *step_count);

                opmath_t step_size = lr / bias_correction1;
                const opmath_t step_size_neg = -step_size;
                const opmath_t denom = ::sqrt(exp_avg_sq) / (::sqrt(bias_correction2) * step_size_neg + eps / step_size_neg);
                step_size *= exp_avg / denom;
                if (maximize) {
                    param += step_size * exp_avg / denom;
                } else {
                    param -= step_size * exp_avg / denom;
                }
                r_args[kParamIdx][ii] = param;
                r_args[kGradIdx][ii] = grad;
                r_args[kExpAvgIdx][ii] = exp_avg;
                r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
                if (Depth == kMaxExpAvgSqIdx + 1) {
                    r_args[kMaxExpAvgSqIdx][ii] = max_exp_avg_sq;
                }
            }
#pragma unroll
                for (int i = 0; i < Depth; i++) {
                    store_args(args[i], r_args[i], i_start, chunk_size, n);
                }
            }
        }
    }
};
} // namespace

// TODO(crcrpar): Support complex dtypes
void _fused_adam_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const bool capturable,
    const c10::optional<at::Tensor>& inv_grad_scale,
    const c10::optional<at::Tensor>& found_inf
) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(params.vec());
    tensor_lists.emplace_back(grads.vec());
    tensor_lists.emplace_back(exp_avgs.vec());
    tensor_lists.emplace_back(exp_avg_sqs.vec());
    auto state_steps_vec = state_steps.vec();
    if (amsgrad) {
        tensor_lists.emplace_back(max_exp_avg_sqs.vec());
    }
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, tensor_lists[0][0].scalar_type(), "fused_adam_kernel_cuda",
        [&]() {
            float* inv_grad_scale_ptr = inv_grad_scale.has_value() ? inv_grad_scale->data_ptr<float>() : nullptr;
            float* found_inf_ptr = found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
            if (amsgrad) {
                multi_tensor_apply_for_fused_optimizer<5>(
                    tensor_lists,
                    state_steps_vec,
                    FusedAdamMathFunctor<scalar_t, 5>(),
                    lr,
                    beta1,
                    beta2,
                    weight_decay,
                    eps,
                    maximize,
                    amsgrad,
                    inv_grad_scale_ptr,
                    found_inf_ptr
                );
            } else {
                multi_tensor_apply_for_fused_optimizer<4>(
                    tensor_lists,
                    state_steps_vec,
                    FusedAdamMathFunctor<scalar_t, 4>(),
                    lr,
                    beta1,
                    beta2,
                    weight_decay,
                    eps,
                    maximize,
                    amsgrad,
                    inv_grad_scale_ptr,
                    found_inf_ptr
                );
            }
        }
    );
}

}} // namespace at::native
