#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

// Wrapper for torch::stable::from_blob with all parameters
// Note: We pass data_ptr as int64_t since we can't pass void* through the
// dispatcher
Tensor my_from_blob(
    int64_t data_ptr,
    torch::headeronly::HeaderOnlyArrayRef<int64_t> sizes,
    torch::headeronly::HeaderOnlyArrayRef<int64_t> strides,
    torch::stable::Device device,
    torch::headeronly::ScalarType dtype) {
  void* data = reinterpret_cast<void*>(data_ptr);
  return torch::stable::from_blob(
      data, sizes, strides, device, dtype);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def(
      "my_from_blob(int data_ptr, int[] sizes, int[] strides, Device device, ScalarType dtype) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_from_blob", TORCH_BOX(&my_from_blob));
}
