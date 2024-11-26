#ifndef AOTI_TORCH_SHIM_XPU
#define AOTI_TORCH_SHIM_XPU

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef USE_XPU
#ifdef __cplusplus
extern "C" {
#endif

struct XPUGuardOpaque;
using XPUGuardHandle = XPUGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_xpu_guard(
    int32_t device_index,
    XPUGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_xpu_guard(XPUGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_xpu_guard_set_index(XPUGuardHandle guard, int32_t device_index);

struct XPUStreamGuardOpaque;
using XPUStreamGuardHandle = XPUStreamGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_xpu_stream(int32_t device_index, void** ret_stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // USE_XPU
#endif // AOTI_TORCH_SHIM_XPU
