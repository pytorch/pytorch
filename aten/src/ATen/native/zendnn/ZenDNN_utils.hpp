#pragma once
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <cpuinfo.h>
#if AT_ZENDNN_ENABLED()
#include <zendnnl.hpp>
namespace at::native {
using namespace zendnnl::interface;

inline data_type_t get_zendnn_dtype(const at::Tensor& tensor) {
  if (tensor.scalar_type() == c10::ScalarType::Float) {
    return data_type_t::f32;
  } else if (tensor.scalar_type() == c10::ScalarType::BFloat16) {
    return data_type_t::bf16;
  }
  TORCH_CHECK(false, "ZenDNN only supports Float32 and BFloat16.");
}
} // namespace at::native
#endif // AT_ZENDNN_ENABLED()
