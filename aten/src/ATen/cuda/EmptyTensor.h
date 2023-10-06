#pragma once
#include <ATen/core/TensorBase.h>

namespace at::detail {

TORCH_CUDA_CPP_API TensorBase empty_cuda(
    IntArrayRef size,
    ScalarType dtype,
    c10::optional<Device> device_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

TORCH_CUDA_CPP_API TensorBase empty_cuda(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

TORCH_CUDA_CPP_API TensorBase empty_cuda(
    IntArrayRef size,
    const TensorOptions &options);

TORCH_CUDA_CPP_API TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    c10::optional<Device> device_opt);

TORCH_CUDA_CPP_API TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt);

TORCH_CUDA_CPP_API TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options);


}  // namespace at::detail
