#include <ATen/autocast_mode.h>

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
// lower_precision_fp
KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
// fp32
KERNEL_PRIVATEUSEONE(dot, fp32)
}