#include <ATen/autocast_mode.h>


TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
}