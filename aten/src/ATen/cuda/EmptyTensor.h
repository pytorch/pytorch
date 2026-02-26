#pragma once
#include <ATen/core/TensorBase.h>

namespace at::detail {

TORCH_CUDA_CPP_API TensorBase empty_cuda(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<Device> device_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

TORCH_CUDA_CPP_API TensorBase empty_cuda(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

TORCH_CUDA_CPP_API TensorBase empty_cuda(
    IntArrayRef size,
    const TensorOptions &options);

TORCH_CUDA_CPP_API TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<Device> device_opt);

TORCH_CUDA_CPP_API TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt);

TORCH_CUDA_CPP_API TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options);


}  // namespace at::detail
