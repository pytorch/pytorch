
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUStream.h>

AOTITorchError aoti_torch_create_xpu_guard(
    int32_t device_index,
    XPUGuardHandle* ret_guard // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::DeviceGuard* guard =
        new at::DeviceGuard(at::Device(at::DeviceType::XPU, device_index));
    *ret_guard = reinterpret_cast<XPUGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_xpu_guard(XPUGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::DeviceGuard*>(guard); });
}

AOTITorchError aoti_torch_xpu_guard_set_index(
    XPUGuardHandle guard,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { reinterpret_cast<at::DeviceGuard*>(guard)->set_index(device_index); });
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_xpu_stream(int32_t device_index, void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_stream = &(at::xpu::getCurrentXPUStream(device_index).queue()); });
}
