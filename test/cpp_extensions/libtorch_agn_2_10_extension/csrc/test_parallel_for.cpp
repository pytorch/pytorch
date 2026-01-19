#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>

using torch::stable::Tensor;

Tensor test_parallel_for(int64_t size, int64_t grain_size) {
  AtenTensorHandle tensor_handle;
  int64_t stride = 1;

  aoti_torch_empty_strided(
      1,
      &size,
      &stride,
      aoti_torch_dtype_int64(),
      aoti_torch_device_type_cpu(),
      0,
      &tensor_handle);

  Tensor tensor(tensor_handle);
  int64_t* data_ptr = reinterpret_cast<int64_t*>(tensor.data_ptr());

  torch::stable::zero_(tensor);

  // Use parallel_for to fill each element with its index
  // If using a parallel path, the thread id is encoded in the upper 32 bits
  torch::stable::parallel_for(
      0, size, grain_size, [data_ptr](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
          STD_TORCH_CHECK(i <= UINT32_MAX);
          uint32_t thread_id;
          torch_get_thread_idx(&thread_id);
          data_ptr[i] = i | (static_cast<int64_t>(thread_id) << 32);
        }
      });

  return tensor;
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("test_parallel_for(int size, int grain_size) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_parallel_for", TORCH_BOX(&test_parallel_for));
}
