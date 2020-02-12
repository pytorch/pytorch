#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/export_c10_op_to_caffe2.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;

namespace caffe2 {
namespace {
inline float sigmoid_partition(float lgt) {
  // computes log(1 + exp(lgt)) with only exp(x) function when x >= 0
  return lgt * (lgt >= 0) + log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

inline float sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * (tgt - (lgt >= 0)) - log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

inline float sigmoid_xent_forward_with_log_d_trick(float lgt, float tgt) {
  return (2 * tgt - 1.) * (lgt - sigmoid_partition(lgt));
}

inline float unjoined_sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * tgt + (tgt - 1) * lgt * (lgt >= 0) -
      (1 - tgt) * log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

void sigmoid_cross_entropy_with_logits_op_cpu_impl(
    const at::Tensor& logits_,
    const at::Tensor& targets_,
    const at::Tensor& out_,
    bool log_D_trick,
    bool unjoined_lr_loss) {
  Tensor logits(logits_);
  Tensor targets(targets_);
  Tensor out(out_);

  CAFFE_ENFORCE_EQ(logits.sizes(), targets.sizes());
  const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto outer_size = logits.numel() / inner_size;

  if (logits.dim() == 0) {
    out.Resize(std::vector<int64_t>{});
  } else {
    std::vector<int64_t> dims(logits.sizes().begin(), logits.sizes().end() - 1);
    out.Resize(dims);
  }
  auto* out_ptr = out.mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();

  auto in_idx = 0;
  for (int i = 0; i < outer_size; ++i) {
    float value = 0;
    for (int j = 0; j < inner_size; ++j) {
      if (unjoined_lr_loss) {
        value += unjoined_sigmoid_xent_forward(
            logits_ptr[in_idx], targets_ptr[in_idx]);
      } else {
        value +=
            (log_D_trick ? sigmoid_xent_forward_with_log_d_trick(
                               logits_ptr[in_idx], targets_ptr[in_idx])
                         : sigmoid_xent_forward(
                               logits_ptr[in_idx], targets_ptr[in_idx]));
      }
      ++in_idx;
    }
    out_ptr[i] = -value / inner_size;
  }
}

static auto registry = c10::RegisterOperators().op(
    "_c10_experimental::SigmoidCrossEntropyWithLogits",
    c10::RegisterOperators::options()
      .kernel<
        decltype(sigmoid_cross_entropy_with_logits_op_cpu_impl),
        &sigmoid_cross_entropy_with_logits_op_cpu_impl>(DispatchKey::CPUTensorId));

} // namespace

C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    "_c10_experimental::SigmoidCrossEntropyWithLogits",
    C10SigmoidCrossEntropyWithLogits_DontUseThisOpYet)

} // namespace caffe2
