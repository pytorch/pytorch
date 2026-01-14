#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/miopen_ctc_loss.h>
#include <ATen/ops/miopen_ctc_loss_native.h>
#endif

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at::native {

bool _use_miopen_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK) {
  return false;
}

bool _use_miopen_ctc_loss_tensor(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK) {
  return false;
}

std::tuple<Tensor, Tensor> miopen_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t blank,
    bool deterministic,
    bool zero_infinity) {
  TORCH_CHECK(false, "miopen_ctc_loss: ATen not compiled with MIOpen support");
}

std::tuple<Tensor, Tensor> miopen_ctc_loss_tensor(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t blank,
    bool deterministic,
    bool zero_infinity) {
  TORCH_CHECK(false, "miopen_ctc_loss: ATen not compiled with MIOpen support");
}

} // namespace at::native

#else // AT_ROCM_ENABLED()

#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Handle.h>

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPException.h>
#include <c10/util/irange.h>

namespace at::native {

bool _use_miopen_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK) {
  auto& ctx = at::globalContext();

  bool use_miopen = ctx.userEnabledCuDNN() && (BLANK == 0) &&
      (targets.dim() == 1) && (log_probs.scalar_type() == at::kFloat) &&
      (targets.scalar_type() == at::kInt) &&
      (targets.device().type() == at::kCPU) && (targets.is_contiguous()) &&
      (log_probs.device().type() == at::kCUDA) && (log_probs.dim() == 3);

  if (use_miopen) {
    // we don't know that input_lengths and target_lengths have the same size
    // (they should, but we didn't check yet)
    int64_t max_input_length = log_probs.size(0);
    for (const auto input_length : input_lengths) {
      use_miopen = use_miopen && ((input_length == max_input_length) ? 1 : 0);
    }
    for (const auto b : c10::irange(target_lengths.size())) {
      // target length < 256 is documented, but we see illegal memory accesses
      // when target lengths > input lengths for MIOpen (same as cuDNN)
      use_miopen = use_miopen && (target_lengths[b] < 256) &&
          (target_lengths[b] <= input_lengths[b]);
    }
  }
  return use_miopen;
}

bool _use_miopen_ctc_loss_tensor(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK) {
  auto& ctx = at::globalContext();

  bool use_miopen = ctx.userEnabledCuDNN() && (BLANK == 0) &&
      (targets.dim() == 1) && (log_probs.scalar_type() == at::kFloat) &&
      (targets.scalar_type() == at::kInt) &&
      (log_probs.device().type() == at::kCUDA) && (targets.is_contiguous()) &&
      (log_probs.dim() == 3) && (input_lengths.scalar_type() == at::kInt) &&
      (target_lengths.scalar_type() == at::kInt);

  if (use_miopen) {
    Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
    Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
    IntArrayRef il(ilc.const_data_ptr<int64_t>(), ilc.numel());
    IntArrayRef tl(tlc.const_data_ptr<int64_t>(), tlc.numel());
    for (const auto b : c10::irange(tl.size())) {
      // target length < 256 is documented, but we see illegal memory accesses
      // when target lengths > input lengths for MIOpen (same as cuDNN)
      use_miopen = use_miopen && (tl[b] < 256) && (tl[b] <= il[b]);
      if (!use_miopen) {
        return use_miopen;
      }
    }
  }
  return use_miopen;
}

std::tuple<Tensor, Tensor> miopen_ctc_loss(
    const Tensor& log_probs_t,
    const Tensor& targets_t,
    IntArrayRef input_lengths_,
    IntArrayRef target_lengths_,
    int64_t BLANK,
    bool deterministic,
    bool zero_infinity) {
  (void)zero_infinity; // only used for backward

  // Validate non-empty tensor before MIOpen call
  TORCH_CHECK(log_probs_t.numel() > 0, "log_probs tensor must not be empty");

  const CheckedFrom c = "miopen_ctc_loss";
  const TensorArg log_probs{log_probs_t, "log_probs", 1};
  const TensorArg targets{targets_t, "targets", 2};

  checkDim(c, log_probs, 3);
  checkScalarType(c, log_probs, kFloat);
  checkDim(c, targets, 1);
  checkScalarType(c, targets, kInt);
  checkContiguous(c, targets);
  checkBackend(c, {*log_probs}, Backend::CUDA);
  checkBackend(c, {*targets}, Backend::CPU);

  const auto batch_size = log_probs->size(1);
  const auto input_length = log_probs->size(0);
  const auto num_labels = log_probs->size(2);

  TORCH_CHECK(
      static_cast<int64_t>(input_lengths_.size()) == batch_size,
      "input_lengths needs to have size to match batch_size");
  TORCH_CHECK(
      static_cast<int64_t>(target_lengths_.size()) == batch_size,
      "target_lengths needs to have size to match batch_size");
  TORCH_CHECK(BLANK == 0, "blank must be label 0 for miopen_ctc_loss");

  std::vector<int> input_lengths(input_lengths_.begin(), input_lengths_.end());
  std::vector<int> target_lengths(target_lengths_.begin(), target_lengths_.end());

  miopenHandle_t handle = getMiopenHandle();
  miopenCTCLossDescriptor_t ctc_desc;
  MIOPEN_CHECK(miopenCreateCTCLossDescriptor(&ctc_desc));

  // MIOpen expects probabilities, apply_softmax=true converts log_probs via exp()
  MIOPEN_CHECK(miopenSetCTCLossDescriptor(
      ctc_desc, miopenFloat, static_cast<int>(BLANK), /*apply_softmax=*/true));

  int dims[3] = {
      static_cast<int>(input_length),
      static_cast<int>(batch_size),
      static_cast<int>(num_labels)};
  int strides[3] = {
      static_cast<int>(batch_size * num_labels),
      static_cast<int>(num_labels),
      1};

  miopenTensorDescriptor_t probs_desc, grads_desc;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&probs_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&grads_desc));
  MIOPEN_CHECK(miopenSetTensorDescriptor(probs_desc, miopenFloat, 3, dims, strides));
  MIOPEN_CHECK(miopenSetTensorDescriptor(grads_desc, miopenFloat, 3, dims, strides));

  Tensor costs = at::empty({batch_size}, log_probs->options());
  Tensor grad = at::empty_like(log_probs_t, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // MIOpen requires labels and lengths on GPU
  Tensor labels_gpu = targets_t.to(Device(at::kCUDA), at::kInt);
  Tensor label_lengths_gpu = at::empty(
      {static_cast<int64_t>(target_lengths.size())},
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  Tensor input_lengths_gpu = at::empty(
      {static_cast<int64_t>(input_lengths.size())},
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

  C10_HIP_CHECK(hipMemcpy(
      label_lengths_gpu.data_ptr<int>(),
      target_lengths.data(),
      target_lengths.size() * sizeof(int),
      hipMemcpyHostToDevice));
  C10_HIP_CHECK(hipMemcpy(
      input_lengths_gpu.data_ptr<int>(),
      input_lengths.data(),
      input_lengths.size() * sizeof(int),
      hipMemcpyHostToDevice));

  size_t workspace_size;
  (void)deterministic; // MIOpen only supports deterministic algorithm
  MIOPEN_CHECK(miopenGetCTCLossWorkspaceSize(
      handle,
      probs_desc,
      grads_desc,
      labels_gpu.data_ptr<int>(),
      label_lengths_gpu.data_ptr<int>(),
      input_lengths_gpu.data_ptr<int>(),
      MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC,
      ctc_desc,
      &workspace_size));

  Tensor workspace = at::empty(workspace_size, log_probs->options().dtype(kByte));

  MIOPEN_CHECK(miopenCTCLoss(
      handle,
      probs_desc,
      log_probs_t.data_ptr(),
      labels_gpu.data_ptr<int>(),
      label_lengths_gpu.data_ptr<int>(),
      input_lengths_gpu.data_ptr<int>(),
      costs.data_ptr(),
      grads_desc,
      grad.data_ptr(),
      MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC,
      ctc_desc,
      workspace.data_ptr(),
      workspace_size));

  MIOPEN_CHECK(miopenDestroyTensorDescriptor(probs_desc));
  MIOPEN_CHECK(miopenDestroyTensorDescriptor(grads_desc));
  MIOPEN_CHECK(miopenDestroyCTCLossDescriptor(ctc_desc));

  return std::make_tuple(costs, grad);
}

std::tuple<Tensor, Tensor> miopen_ctc_loss_tensor(
    const Tensor& log_probs_t,
    const Tensor& targets_t,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK,
    bool deterministic,
    bool zero_infinity) {
  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  IntArrayRef il(ilc.const_data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.const_data_ptr<int64_t>(), tlc.numel());

  Tensor targets_cpu = targets_t.device().type() == at::kCPU
      ? targets_t
      : targets_t.to(Device(at::kCPU));

  return at::native::miopen_ctc_loss(
      log_probs_t, targets_cpu, il, tl, BLANK, deterministic, zero_infinity);
}

} // namespace at::native

#endif // AT_ROCM_ENABLED()
