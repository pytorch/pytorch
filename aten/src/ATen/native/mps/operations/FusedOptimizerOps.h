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
constexpr constant uint kMomentumBufferListIdx = kGradIdx + kmaxTensors;
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

template<typename T>
struct SgdArguments {
    metal::array<device T *,  kmaxTensors>   params        [[ id(kParamIdx) ]];
    metal::array<device T *,  kmaxTensors>   grads         [[ id(kGradIdx) ]];
};

template<typename T>
struct SgdMomentumArguments {
    metal::array<device T *,  kmaxTensors>   params        [[ id(kParamIdx) ]];
    metal::array<device T *,  kmaxTensors>   grads         [[ id(kGradIdx) ]];
    metal::array<device T *,  kmaxTensors>   momentum_buffer_list       [[ id(kMomentumBufferListIdx) ]];
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
  const T bias_correction1 = 1 - metal::precise::pow(beta1, casted_state_steps);
  const T step_size = lr / bias_correction1;
  const T bias_correction2 = 1 - metal::precise::pow(beta2, casted_state_steps);
  const T bias_correction2_sqrt = metal::precise::sqrt(bias_correction2);
  max_exp_avg_sq = metal::max(max_exp_avg_sq, exp_avg_sq);

  const T denom = (metal::precise::sqrt(max_exp_avg_sq) / bias_correction2_sqrt) + eps;
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
  const T bias_correction1 = 1 - metal::precise::pow(beta1, casted_state_steps);
  const T step_size = lr / bias_correction1;
  const T bias_correction2 = 1 - metal::precise::pow(beta2, casted_state_steps);
  const T bias_correction2_sqrt = metal::precise::sqrt(bias_correction2);
  const T denom = (metal::precise::sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps;
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

template <typename T>
inline void sgd_momentum_math(
    device T & param,
    device T & grad,
    device T & momentum_buffer,
    const float weight_decay,
    const float momentum,
    const float lr,
    const float dampening,
    const uint8_t nesterov,
    const uint8_t maximize,
    const uint8_t is_first_step
) {
  auto grad_ = grad;
  if (maximize) {
      grad_ *= -1.0;
  }
  if (weight_decay != 0) {
      grad_ += weight_decay * param;
  }

  momentum_buffer = is_first_step ? grad_ : (momentum * momentum_buffer + (1 - dampening) * grad_);
  if (nesterov) {
      grad_ += momentum * momentum_buffer;
  } else {
      grad_ = momentum_buffer;
  }

  param -= lr * grad_;
}

template <typename T>
inline void sgd_math(
    device T & param,
    device T & grad,
    const float weight_decay,
    const float lr,
    const uint8_t maximize
) {
  auto grad_ = grad;
  if (maximize) {
      grad_ *= -1.0;
  }
  if (weight_decay != 0) {
      grad_ += weight_decay * param;
  }

  param -= lr * grad_;
}

template <typename T>
kernel void fused_sgd(
    device   SgdArguments<T> & args    [[buffer(0)]],
    constant MetadataArguments & metadata_args [[buffer(1)]],
    constant float & weight_decay   [[buffer(2)]],
    constant float & lr             [[buffer(3)]],
    constant uint8_t & maximize       [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {

    const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
    const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
    const uint32_t chunk_offset = chunk_idx * chunk_size;
    const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

    // each chunk is a threadgroup
    auto param = args.params[tensor_loc] + chunk_offset;
    auto grad = args.grads[tensor_loc] + chunk_offset;

    for (uint32_t i_start = tid; i_start < numel && i_start < chunk_size; i_start += tptg) {
      sgd_math<T>(
        *(param + i_start),
        *(grad + i_start),
        weight_decay,
        lr,
        maximize
      );
    }
}

template <typename T>
kernel void fused_sgd(
    device   SgdMomentumArguments<T> & args    [[buffer(0)]],
    constant MetadataArguments & metadata_args [[buffer(1)]],
    constant float & weight_decay   [[buffer(2)]],
    constant float & momentum       [[buffer(3)]],
    constant float & lr             [[buffer(4)]],
    constant float & dampening          [[buffer(5)]],
    constant uint8_t & nesterov          [[buffer(6)]],
    constant uint8_t   & maximize       [[buffer(7)]],
    constant uint8_t   & is_first_step       [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {

    const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
    const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
    const uint32_t chunk_offset = chunk_idx * chunk_size;
    const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

    // each chunk is a threadgroup
    auto param = args.params[tensor_loc] + chunk_offset;
    auto grad = args.grads[tensor_loc] + chunk_offset;
    auto momentum_buffer_list = args.momentum_buffer_list[tensor_loc] + chunk_offset;

    for (uint32_t i_start = tid; i_start < numel && i_start < chunk_size; i_start += tptg) {
      sgd_momentum_math<T>(
        *(param + i_start),
        *(grad + i_start),
        *(momentum_buffer_list + i_start),
        weight_decay,
        momentum,
        lr,
        dampening,
        nesterov,
        maximize,
        is_first_step
      );
    }
}

#define REGISTER_FUSED_SGD_OP(DTYPE) \
template                                                            \
[[host_name("fused_sgd_" #DTYPE)]]                                  \
kernel void fused_sgd<DTYPE>(                                       \
    device   SgdArguments<DTYPE>     & args          [[buffer(0)]], \
    constant MetadataArguments       & metadata_args [[buffer(1)]], \
    constant float                   & weight_decay  [[buffer(2)]], \
    constant float                   & lr            [[buffer(3)]], \
    constant uint8_t                 & maximize      [[buffer(4)]], \
    uint tid  [[thread_position_in_threadgroup]], \
    uint tgid [[threadgroup_position_in_grid]],   \
    uint tptg [[threads_per_threadgroup]])

#define REGISTER_FUSED_SGD_MOMENTUM_OP(DTYPE) \
template                                                            \
[[host_name("fused_sgd_momentum_" #DTYPE)]]                         \
kernel void fused_sgd<DTYPE>(                                       \
    device   SgdMomentumArguments<DTYPE> & args      [[buffer(0)]], \
    constant MetadataArguments       & metadata_args [[buffer(1)]], \
    constant float                   & weight_decay  [[buffer(2)]], \
    constant float                   & momentum      [[buffer(3)]], \
    constant float                   & lr            [[buffer(4)]], \
    constant float                   & dampening     [[buffer(5)]], \
    constant uint8_t                 & nesterov      [[buffer(6)]], \
    constant uint8_t                 & maximize      [[buffer(7)]], \
    constant uint8_t                 & is_first_step [[buffer(8)]], \
    uint tid  [[thread_position_in_threadgroup]], \
    uint tgid [[threadgroup_position_in_grid]],   \
    uint tptg [[threads_per_threadgroup]])

REGISTER_FUSED_SGD_OP(float);
REGISTER_FUSED_SGD_OP(half);
REGISTER_FUSED_SGD_MOMENTUM_OP(float);
REGISTER_FUSED_SGD_MOMENTUM_OP(half);

)METAL";

static std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getCPLState(const std::string& fname) {
  static MetalShaderLibrary lib(FUSED_ADAM_OPS, 0);
  return std::make_pair(lib.getPipelineStateForFunc(fname), lib.getMTLFunction(fname));
}

} //namespace mps
} // namespace at::native