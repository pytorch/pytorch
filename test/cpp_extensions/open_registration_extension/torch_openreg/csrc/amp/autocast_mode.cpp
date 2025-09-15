#include <ATen/autocast_mode.h>

// LITERALINCLUDE START: AMP_REGISTER_FALLBACK
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}
// LITERALINCLUDE END: AMP_REGISTER_FALLBACK

// LITERALINCLUDE START: AMP_REGISTER_KERNEL
at::Tensor binary_cross_entropy_not_supported(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, int64_t)
{
    AT_ERROR("torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are not supported.");
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
    // lower_precision_fp
    KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
    // fp32
    KERNEL_PRIVATEUSEONE(asin, fp32)

    m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
           TORCH_FN((&binary_cross_entropy_not_supported)));
}
// LITERALINCLUDE END: AMP_REGISTER_KERNEL