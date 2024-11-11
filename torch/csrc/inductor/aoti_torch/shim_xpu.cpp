
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUStreamGuard.h>

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

AOTITorchError aoti_torch_create_xpu_stream_guard(
    void* stream,
    int32_t device_index,
    XPUStreamGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (stream == nullptr) {
      stream = &(at::xpu::getCurrentXPUStream(device_index).queue());
    }
    at::xpu::XPUStreamGuard* guard =
        new at::xpu::XPUStreamGuard(at::xpu::getStreamFromExternal(
            static_cast<sycl::queue*>(stream), device_index));
    *ret_guard = reinterpret_cast<XPUStreamGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_xpu_stream_guard(XPUStreamGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::xpu::XPUStreamGuard*>(guard); });
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_xpu_stream(int32_t device_index, void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_stream = &(at::xpu::getCurrentXPUStream(device_index).queue()); });
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_xpu_device(int32_t* device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *device_index = static_cast<int32_t>(c10::xpu::current_device()); });
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_set_current_xpu_device(const int32_t& device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { c10::xpu::set_device(static_cast<int8_t>(device_index)); });
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_current_sycl_queue(void** ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    int32_t device_index = static_cast<int32_t>(c10::xpu::current_device());
    *ret = &(at::xpu::getCurrentXPUStream(device_index).queue());
  });
}
