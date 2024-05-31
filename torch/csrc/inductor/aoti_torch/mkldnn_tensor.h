#pragma once

#include <ATen/Tensor.h>

namespace torch {
namespace aot_inductor {

void* data_ptr_from_mkldnn(at::Tensor* mkldnn_tensor);

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size);

} // namespace aot_inductor
} // namespace torch
