#include <ATen/autocast_mode.h>
#include <torch/library.h>

// Autocast policies for Vulkan backend (PrivateUse1)
//
// Ops are cast to float16 for compute-bound ops (mm, bmm, linear, conv)
// and kept in float32 for precision-sensitive ops (norms, softmax, loss).
// All other ops fall through to PrivateUse1 without casting.
//
// The actual f16 computation is done via widen-compute-narrow:
// f16 inputs are cast to f32 for shader dispatch, then output is cast back to f16.
// This gives correct AMP behavior: reduced memory for activations/weights,
// while compute stays in f32 for numerical stability.

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
    // Lower precision (f16) for compute-bound BLAS ops
    KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(bmm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(addmm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(linear, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(convolution_overrideable, lower_precision_fp)

    // Keep in f32 for precision-sensitive ops
    KERNEL_PRIVATEUSEONE(native_layer_norm, fp32)
    KERNEL_PRIVATEUSEONE(native_batch_norm, fp32)
    KERNEL_PRIVATEUSEONE(native_group_norm, fp32)
    KERNEL_PRIVATEUSEONE(_softmax, fp32)
    KERNEL_PRIVATEUSEONE(_log_softmax, fp32)
    KERNEL_PRIVATEUSEONE(binary_cross_entropy, fp32)
    KERNEL_PRIVATEUSEONE(nll_loss_forward, fp32)
    KERNEL_PRIVATEUSEONE(mse_loss, fp32)
}

// Fallback: all other ops pass through without casting.
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}
