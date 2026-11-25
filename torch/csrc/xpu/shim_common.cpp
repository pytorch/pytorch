#include <c10/xpu/XPUStream.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>

AOTITorchError torch_get_xpu_stream_from_pool(
    bool is_high_priority,
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret_stream =
        &c10::xpu::getStreamFromPool(is_high_priority, device_index).queue();
  });
}
