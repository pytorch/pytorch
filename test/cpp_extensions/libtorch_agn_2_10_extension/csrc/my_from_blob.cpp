#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

// Global counter to track deleter calls for testing
static int64_t g_deleter_call_count = 0;

static void test_deleter(void* /*data*/) {
  g_deleter_call_count++;
}

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

// Wrapper for from_blob with deleter - uses a test deleter that increments
// a global counter
Tensor my_from_blob_with_deleter(
    int64_t data_ptr,
    torch::headeronly::HeaderOnlyArrayRef<int64_t> sizes,
    torch::headeronly::HeaderOnlyArrayRef<int64_t> strides,
    torch::stable::Device device,
    torch::headeronly::ScalarType dtype) {
  void* data = reinterpret_cast<void*>(data_ptr);
  return torch::stable::from_blob(
      data, sizes, strides, device, dtype, test_deleter);
}

int64_t get_deleter_call_count() {
  return g_deleter_call_count;
}

void reset_deleter_call_count() {
  g_deleter_call_count = 0;
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def(
      "my_from_blob(int data_ptr, int[] sizes, int[] strides, Device device, ScalarType dtype) -> Tensor");
  m.def(
      "my_from_blob_with_deleter(int data_ptr, int[] sizes, int[] strides, Device device, ScalarType dtype) -> Tensor");
  m.def("get_deleter_call_count() -> int");
  m.def("reset_deleter_call_count() -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_from_blob", TORCH_BOX(&my_from_blob));
  m.impl("my_from_blob_with_deleter", TORCH_BOX(&my_from_blob_with_deleter));
  m.impl("get_deleter_call_count", TORCH_BOX(&get_deleter_call_count));
  m.impl("reset_deleter_call_count", TORCH_BOX(&reset_deleter_call_count));
}
