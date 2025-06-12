#pragma once
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <cpuinfo.h>

#if AT_ZENDNN_ENABLED()
#include <zendnnl.hpp>
namespace at::native {
using namespace zendnnl::interface;
inline bool zendnn_bf16_device_check() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512bf16();
}
inline void set_zendnn_tensor_attributes(
    const at::Tensor& at_tensor,
    tensor_t& zendnn_tensor,
    const std::string& tensor_name,
    const data_type_t& tensor_datatype) {
  std::vector<long unsigned int> at_tensor_sizes = {
      static_cast<long unsigned int>(at_tensor.sizes()[0]),
      static_cast<long unsigned int>(at_tensor.sizes()[1])};
  if (at_tensor.dim() == 1) {
    at_tensor_sizes = {static_cast<long unsigned int>(at_tensor.sizes()[0])};
  }
  void* at_tensor_ptr = at_tensor.data_ptr();

  zendnn_tensor.set_name(tensor_name)
      .set_size(at_tensor_sizes)
      .set_data_type(tensor_datatype)
      .set_storage(at_tensor_ptr, at_tensor.nbytes());
  if (tensor_name == "weights") {
    zendnn_tensor.set_order("ba");
  }
}

} // namespace at::native
#endif // AT_ZENDNN_ENABLED()
