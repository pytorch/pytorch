#pragma once

#include <ATen/Tensor.h>
#include <hipdnn_frontend.hpp>
#include <c10/macros/Export.h>

namespace at::native {

TORCH_CUDA_CPP_API hipdnn_frontend::DataType getHipdnnDataType(
    const at::Tensor& tensor);

} // namespace at::native
