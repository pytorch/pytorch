#pragma once

// This header is used by model.so and must depend only on the stable C ABI,
// not on ATen/C10 or MTIA-private headers.
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/stable/accelerator.h>

#include <cstdint>

// The kernel is owned by the MTIA runtime; model code uses its opaque handle.
typedef struct TorchInductorMTIAKernel TorchInductorMTIAKernel;

namespace torch::aot_inductor {

using MTIAInductorDeviceGuard = torch::stable::accelerator::DeviceGuard;

inline int32_t mtia_check_stream(void* stream, int32_t device_index) {
  AOTI_RUNTIME_CHECK(
      stream == nullptr,
      "MTIA AOTInductor supports only the current stream; pass a null stream");
  return device_index;
}

class MTIAInductorStreamGuard {
 public:
  MTIAInductorStreamGuard(void* stream, int32_t device_index)
      : device_guard_(mtia_check_stream(stream, device_index)) {}

 private:
  MTIAInductorDeviceGuard device_guard_;
};

} // namespace torch::aot_inductor
