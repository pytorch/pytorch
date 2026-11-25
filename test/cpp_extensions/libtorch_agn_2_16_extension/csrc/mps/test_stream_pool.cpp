#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>

namespace {
void* get_mps_stream_from_pool(int32_t device_index) {
  return torch::stable::accelerator::getStreamFromPool<
      torch::headeronly::DeviceType::MPS>(device_index);
}
} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("get_mps_stream_from_pool(int device_index) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("get_mps_stream_from_pool", TORCH_BOX(&get_mps_stream_from_pool));
}
