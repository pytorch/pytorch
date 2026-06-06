#include <cstdint>
#include <limits>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/util/Exception.h>

int64_t test_stream_native_handle(int32_t device_index) {
  STD_TORCH_CHECK(
      device_index >= std::numeric_limits<int32_t>::min() &&
          device_index <= std::numeric_limits<int32_t>::max(),
      "Device index is out of range of DeviceIndex (int32_t).");

  void* native_handle =
      torch::stable::accelerator::getCurrentStream(device_index).nativeHandle();
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(native_handle));
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("test_stream_native_handle(int device_index) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("test_stream_native_handle", TORCH_BOX(&test_stream_native_handle));
}
