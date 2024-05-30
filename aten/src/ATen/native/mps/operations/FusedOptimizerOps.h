#pragma once
#include <ATen/native/mps/OperationUtils.h>

namespace at::native {
namespace mps {

static const char* FUSED_ADAM_OPS = R"METAL(
#include <metal_stdlib>

#define kmaxThreadGroups 32
#define kmaxTensors 32
#define chunk_size 65536

constexpr constant uint kParamIdx = 0;
constexpr constant uint kGradIdx = kParamIdx + kmaxTensors;
constexpr constant uint kExpAvgIdx = kGradIdx + kmaxTensors;
constexpr constant uint kExpAvgSqIdx = kExpAvgIdx + kmaxTensors;
constexpr constant uint kMaxExpAvgSqIdx = kExpAvgSqIdx + kmaxTensors;
constexpr constant uint kStateStepsIdx = kExpAvgSqIdx + kmaxTensors;
constexpr constant uint kStateStepsIdxForAmsgrad = kMaxExpAvgSqIdx + kmaxTensors;

template<typename T, typename state_steps_t>
struct AdamArguments {
    metal::array<device T *,  kmaxTensors>   params        [[ id(kParamIdx) ]];
    metal::array<device T *,  kmaxTensors>   grads         [[ id(kGradIdx) ]];
    metal::array<device T *,  kmaxTensors>   exp_avgs      [[ id(kExpAvgIdx) ]];
    metal::array<device T *,  kmaxTensors>   exp_avg_sqs   [[ id(kExpAvgSqIdx) ]];
    metal::array<device state_steps_t *,  kmaxTensors>   state_steps   [[ id(kStateStepsIdx) ]];
};

template<typename T, typename state_steps_t>
struct AdamAmsgradArguments {
    metal::array<device T *,  kmaxTensors>   params        [[ id(kParamIdx) ]];
    metal::array<device T *,  kmaxTensors>   grads         [[ id(kGradIdx) ]];
    metal::array<device T *,  kmaxTensors>   exp_avgs      [[ id(kExpAvgIdx) ]];
    metal::array<device T *,  kmaxTensors>   exp_avg_sqs   [[ id(kExpAvgSqIdx) ]];
    metal::array<device T *,  kmaxTensors>   max_exp_avg_sqs   [[ id(kMaxExpAvgSqIdx) ]];
    metal::array<device state_steps_t *,  kmaxTensors>   state_steps   [[ id(kStateStepsIdxForAmsgrad) ]];
};

struct MetadataArguments {
    uint32_t numels[kmaxTensors];
    uint32_t threadgroup_to_tensor[kmaxThreadGroups];
    uint32_t threadgroup_to_chunk[kmaxThreadGroups];
};

enum ADAM_MODE : uint8_t {
  ORIGINAL = 0,
  ADAMW = 1
};

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
inline void adam_math_amsgrad(
    device T & param,
    device T & grad,
    device T & exp_avg,
    device T & exp_avg_sq,
    device T & max_exp_avg_sq,
    device state_steps_t & state_steps,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const uint8_t maximize
) {
  T grad_ = grad;

  if (maximize) {
    grad = -grad;
  }

  // Update param, grad, 1st and 2nd order momentum.
  if (weight_decay != 0) {
    switch (adam_mode) {
      case ADAM_MODE::ORIGINAL:
        grad += param * weight_decay;
        break;
      case ADAM_MODE::ADAMW:
        param -= lr * weight_decay * param;
        break;
    }
  }

  exp_avg = beta1 * exp_avg + (1 - beta1) * grad;
  exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad;
  const float casted_state_steps = static_cast<float>(state_steps);
  const T bias_correction1 = 1 - metal::pow(beta1, casted_state_steps);
  const T step_size = lr / bias_correction1;
  const T bias_correction2 = 1 - metal::pow(beta2, casted_state_steps);
  const T bias_correction2_sqrt = metal::sqrt(bias_correction2);
  max_exp_avg_sq = metal::max(max_exp_avg_sq, exp_avg_sq);

  const T denom = (metal::sqrt(max_exp_avg_sq) / bias_correction2_sqrt) + eps;
  param -= step_size * exp_avg / denom;
  grad = grad_;
}

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
inline void adam_math(
    device T & param,
    device T & grad,
    device T & exp_avg,
    device T & exp_avg_sq,
    device state_steps_t & state_steps,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const uint8_t maximize
) {
  T grad_ = grad;

  if (maximize) {
    grad = -grad;
  }

  // Update param, grad, 1st and 2nd order momentum.
  if (weight_decay != 0) {
    switch (adam_mode) {
      case ADAM_MODE::ORIGINAL:
        grad += param * weight_decay;
        break;
      case ADAM_MODE::ADAMW:
        param -= lr * weight_decay * param;
        break;
    }
  }

  exp_avg = beta1 * exp_avg + (1 - beta1) * grad;
  exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad;
  const float casted_state_steps = static_cast<float>(state_steps);
  const T bias_correction1 = 1 - metal::pow(beta1, casted_state_steps);
  const T step_size = lr / bias_correction1;
  const T bias_correction2 = 1 - metal::pow(beta2, casted_state_steps);
  const T bias_correction2_sqrt = metal::sqrt(bias_correction2);
  const T denom = (metal::sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps;
  param -= step_size * exp_avg / denom;
  grad = grad_;
}

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
kernel void fused_adam_amsgrad(
    device   AdamAmsgradArguments<T, state_steps_t> & args    [[buffer(0)]],
    constant MetadataArguments & metadata_args [[buffer(1)]],
    constant float & lr             [[buffer(2)]],
    constant float & beta1          [[buffer(3)]],
    constant float & beta2          [[buffer(4)]],
    constant float & weight_decay   [[buffer(5)]],
    constant float & eps            [[buffer(6)]],
    constant uint8_t   & maximize       [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {

    const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
    const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
    const uint32_t chunk_offset = chunk_idx * chunk_size;
    const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

    const auto step_count = args.state_steps[tensor_loc];

    // each chunk is a threadgroup
    auto param = args.params[tensor_loc] + chunk_offset;
    auto grad = args.grads[tensor_loc] + chunk_offset;
    auto exp_avg = args.exp_avgs[tensor_loc] + chunk_offset;
    auto exp_avg_sq = args.exp_avg_sqs[tensor_loc] + chunk_offset;
    auto max_exp_avg_sq = args.max_exp_avg_sqs[tensor_loc] + chunk_offset;

    for (uint32_t i_start = tid; i_start < numel && i_start < chunk_size; i_start += tptg) {
      adam_math_amsgrad<T, state_steps_t, adam_mode>(
        *(param + i_start),
        *(grad + i_start),
        *(exp_avg + i_start),
        *(exp_avg_sq + i_start),
        *(max_exp_avg_sq + i_start),
        *step_count,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize
      );
    }
}

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
kernel void fused_adam(
    device   AdamArguments<T, state_steps_t> & args    [[buffer(0)]],
    constant MetadataArguments & metadata_args [[buffer(1)]],
    constant float & lr             [[buffer(2)]],
    constant float & beta1          [[buffer(3)]],
    constant float & beta2          [[buffer(4)]],
    constant float & weight_decay   [[buffer(5)]],
    constant float & eps            [[buffer(6)]],
    constant uint8_t   & maximize       [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {

    const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
    const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
    const uint32_t chunk_offset = chunk_idx * chunk_size;
    const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

    const auto step_count = args.state_steps[tensor_loc];

    // each chunk is a threadgroup
    auto param = args.params[tensor_loc] + chunk_offset;
    auto grad = args.grads[tensor_loc] + chunk_offset;
    auto exp_avg = args.exp_avgs[tensor_loc] + chunk_offset;
    auto exp_avg_sq = args.exp_avg_sqs[tensor_loc] + chunk_offset;

    for (uint32_t i_start = tid; i_start < numel && i_start < chunk_size; i_start += tptg) {
      adam_math<T, state_steps_t, adam_mode>(
        *(param + i_start),
        *(grad + i_start),
        *(exp_avg + i_start),
        *(exp_avg_sq + i_start),
        *step_count,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize
      );
    }
}

#define REGISTER_FUSED_ADAM_OP(DTYPE, STATE_STEPS_DTYPE, ADAM_MODE_DTYPE, HOST_NAME, KERNEL_NAME, ARGUMENTS_STRUCT)       \
template                                    \
[[host_name(#HOST_NAME "_" #DTYPE "_" #STATE_STEPS_DTYPE)]]        \
kernel void KERNEL_NAME<DTYPE, STATE_STEPS_DTYPE, ADAM_MODE_DTYPE>(             \
    device   ARGUMENTS_STRUCT<DTYPE, STATE_STEPS_DTYPE> & args    [[buffer(0)]],\
    constant MetadataArguments & metadata_args [[buffer(1)]],\
    constant float & lr             [[buffer(2)]],\
    constant float & beta1          [[buffer(3)]],\
    constant float & beta2          [[buffer(4)]],\
    constant float & weight_decay   [[buffer(5)]],\
    constant float & eps            [[buffer(6)]],\
    constant uint8_t   & maximize       [[buffer(7)]],\
    uint tid [[thread_position_in_threadgroup]],\
    uint tgid [[threadgroup_position_in_grid]],\
    uint tptg [[threads_per_threadgroup]])

REGISTER_FUSED_ADAM_OP(float, float, ADAM_MODE::ORIGINAL, fused_adam, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(float, half, ADAM_MODE::ORIGINAL, fused_adam, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(half, float, ADAM_MODE::ORIGINAL, fused_adam, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(half, half, ADAM_MODE::ORIGINAL, fused_adam, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(float, float, ADAM_MODE::ADAMW, fused_adamw, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(float, half, ADAM_MODE::ADAMW, fused_adamw, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(half, float, ADAM_MODE::ADAMW, fused_adamw, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(half, half, ADAM_MODE::ADAMW, fused_adamw, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(float, float, ADAM_MODE::ORIGINAL, fused_adam_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(float, half, ADAM_MODE::ORIGINAL, fused_adam_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(half, float, ADAM_MODE::ORIGINAL, fused_adam_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(half, half, ADAM_MODE::ORIGINAL, fused_adam_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(float, float, ADAM_MODE::ADAMW, fused_adamw_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(float, half, ADAM_MODE::ADAMW, fused_adamw_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(half, float, ADAM_MODE::ADAMW, fused_adamw_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(half, half, ADAM_MODE::ADAMW, fused_adamw_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);

)METAL";

static id<MTLLibrary> compileFusedOptimizerOpsLibrary(const id<MTLDevice>& device, const char* kernel) {
  static id<MTLLibrary> fusedOptimizerLibrary = nil;
  if (fusedOptimizerLibrary) {
    return fusedOptimizerLibrary;
  }

  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  fusedOptimizerLibrary = [device newLibraryWithSource:[NSString stringWithCString:kernel
                                                                         encoding:NSASCIIStringEncoding]
                                              options:options
                                                error:&error];
  TORCH_CHECK(
      fusedOptimizerLibrary, "Failed to create metal fused optimizer library, error: ", [[error description] UTF8String]);
  return fusedOptimizerLibrary;
}


static std::tuple<id<MTLComputePipelineState>, id<MTLFunction>> getPipelineState(const id<MTLDevice>& device, const char* kernel, const std::string& kernel_name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  static std::unordered_map<std::string, id<MTLFunction>> funcCache;

  id<MTLComputePipelineState> pso = psoCache[kernel_name];
  if (pso) {
    return std::make_tuple(pso, funcCache[kernel_name]);
  }

  NSError* error = nil;
  id<MTLLibrary> fusedOptimizerLib = compileFusedOptimizerOpsLibrary(device, kernel);
  id<MTLFunction> fusedOptimizerFunc =
      [fusedOptimizerLib newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
  TORCH_CHECK(fusedOptimizerFunc, "Failed to create function state object for: ", kernel_name);
  pso = [device newComputePipelineStateWithFunction:fusedOptimizerFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel_name] = pso;
  funcCache[kernel_name] = fusedOptimizerFunc;
  return std::make_tuple(pso, fusedOptimizerFunc);
}

} //namespace mps
} // namespace at::native