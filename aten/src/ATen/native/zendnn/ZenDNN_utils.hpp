#pragma once
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <cpuinfo.h>
#include <torch/library.h>

#if AT_ZENDNN_ENABLED()
#include <zendnnl.hpp>
namespace at::native {
using namespace zendnnl::interface;
inline bool zendnn_bf16_device_check() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512bf16();
}

inline bool is_tensor_2d_n_transposed(const at::Tensor& t) {
  const auto sizes = t.sizes();
  const auto strides = t.strides();
  // check for transposed tensors
  if (t.dim() == 2) {
    return strides[0] == 1 && strides[1] == sizes[0];
  } else {
    return false;
  }
}

inline void set_zendnn_tensor_attributes(
    const at::Tensor& at_tensor,
    tensor_t& zendnn_tensor,
    const std::string& tensor_name,
    const data_type_t& tensor_datatype,
    const bool& is_tensor_prepacked = false) {
  std::vector<long unsigned int> at_tensor_sizes_vec;
  auto at_tensor_sizes = at_tensor.sizes();
  for (auto val : at_tensor_sizes) {
    at_tensor_sizes_vec.emplace_back(static_cast<long unsigned int>(val));
  }

  void* at_tensor_ptr = at_tensor.data_ptr();

  zendnn_tensor.set_name(tensor_name)
      .set_size(at_tensor_sizes_vec)
      .set_data_type(tensor_datatype)
      .set_storage(at_tensor_ptr, at_tensor.nbytes());

  if (is_tensor_2d_n_transposed(at_tensor)) {
    zendnn_tensor.set_order("ba");
  }

  if (is_tensor_prepacked && tensor_name == "weights") {
    zendnn_tensor.set_layout(tensor_layout_t::blocked);
  }
}

} // namespace at::native
#endif // AT_ZENDNN_ENABLED()
