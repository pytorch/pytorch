#include <ATen/autocast_mode.h>

// LITERALINCLUDE START: AMP_REGISTER_FALLBACK
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}
// LITERALINCLUDE END: AMP_REGISTER_FALLBACK

// LITERALINCLUDE START: AMP_REGISTER_KERNEL
TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
    // lower_precision_fp
    KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
    // fp32
    KERNEL_PRIVATEUSEONE(dot, fp32)
}
// LITERALINCLUDE END: AMP_REGISTER_KERNEL