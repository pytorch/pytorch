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

inline data_type_t get_zendnn_dtype(const at::Tensor& tensor) {
  if (tensor.scalar_type() == c10::ScalarType::Float) {
    return data_type_t::f32;
  } else if (tensor.scalar_type() == c10::ScalarType::BFloat16) {
    return data_type_t::bf16;
  }
  TORCH_CHECK(false, "ZenDNN only supports Float32 and BFloat16.");
}

inline bool is_tensor_2d_and_transposed(const at::Tensor& t) {
  if (t.dim() == 2) {
    return t.strides()[0] == 1 && t.strides()[1] == t.sizes()[0];
  }
  return false;
}

inline void set_zendnn_tensor_attributes(
    const at::Tensor& at_tensor,
    tensor_t& zendnn_tensor,
    const std::string& tensor_name,
    const data_type_t& tensor_datatype,
    const bool is_tensor_prepacked = false) {
  std::vector<long unsigned int> at_tensor_sizes_vec(
      at_tensor.sizes().begin(), at_tensor.sizes().end());
  void* at_tensor_ptr = at_tensor.data_ptr();
  zendnn_tensor.set_name(tensor_name)
      .set_size(at_tensor_sizes_vec)
      .set_data_type(tensor_datatype)
      .set_storage(at_tensor_ptr, at_tensor.nbytes());
  if (is_tensor_2d_and_transposed(at_tensor)) {
    zendnn_tensor.set_order("ba");
  }
  if (is_tensor_prepacked && tensor_name == "weights") {
    zendnn_tensor.set_layout(tensor_layout_t::blocked);
  }
}

inline void create_zendnn_tensor(
    const at::Tensor& source_tensor,
    tensor_t& target_tensor,
    const std::string& tensor_name,
    const data_type_t datatype,
    const bool is_tensor_prepacked = false) {
  set_zendnn_tensor_attributes(
      source_tensor, target_tensor, tensor_name, datatype, is_tensor_prepacked);
  target_tensor.create();
  TORCH_CHECK(
      target_tensor.check(),
      "tensor creation of ",
      target_tensor.get_name(),
      " failed.");
}

} // namespace at::native
#endif // AT_ZENDNN_ENABLED()
