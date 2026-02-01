#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/c/shim.h>

void* my_get_current_cuda_stream(int32_t device_index) {
  void* ret_stream;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &ret_stream));
  return ret_stream;
}

void my_set_current_cuda_stream(void* stream, int32_t device_index) {
  TORCH_ERROR_CODE_CHECK(torch_set_current_cuda_stream(stream, device_index));
}

void* my_get_cuda_stream_from_pool(bool isHighPriority, int32_t device_index) {
  void* ret_stream;
  TORCH_ERROR_CODE_CHECK(torch_get_cuda_stream_from_pool(isHighPriority, device_index, &ret_stream));
  return ret_stream;
}

void my_cuda_stream_synchronize(void* stream, int32_t device_index) {
  TORCH_ERROR_CODE_CHECK(torch_cuda_stream_synchronize(stream, device_index));
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_get_current_cuda_stream(int device_index) -> int");
  m.def("my_set_current_cuda_stream(int stream, int device_index) -> ()");
  m.def("my_get_cuda_stream_from_pool(bool isHighPriority, int device_index) -> int");
  m.def("my_cuda_stream_synchronize(int stream, int device_index) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_get_current_cuda_stream", TORCH_BOX(&my_get_current_cuda_stream));
  m.impl("my_set_current_cuda_stream", TORCH_BOX(&my_set_current_cuda_stream));
  m.impl("my_get_cuda_stream_from_pool", TORCH_BOX(&my_get_cuda_stream_from_pool));
  m.impl("my_cuda_stream_synchronize", TORCH_BOX(&my_cuda_stream_synchronize));
}
