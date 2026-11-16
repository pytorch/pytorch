#include <ATen/autocast_mode.h>

using at::Tensor;

Tensor binary_cross_entropy_banned(
    const Tensor&,
    const Tensor&,
    const std::optional<Tensor>&,
    int64_t) {
  TORCH_CHECK(
      false,
      "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
      "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
      "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
      "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
      "safe to autocast.");
}

// LITERALINCLUDE START: AMP FALLTHROUTH
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
// LITERALINCLUDE END: AMP FALLTHROUTH

// LITERALINCLUDE START: AMP IMPL
TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // lower_precision_fp
  KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)

  // fp32
  KERNEL_PRIVATEUSEONE(asin, fp32)

  m.impl(
      TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
      TORCH_FN((&binary_cross_entropy_banned)));
}
// LITERALINCLUDE END: AMP IMPL
