#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <hipdnn_frontend.hpp>
#include <memory>

namespace at::native {

inline hipdnn_frontend::DataType getHipdnnDataType(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case at::kFloat:
      return hipdnn_frontend::DataType::FLOAT;
    case at::kHalf:
      return hipdnn_frontend::DataType::HALF;
    case at::kBFloat16:
      return hipdnn_frontend::DataType::BFLOAT16;
    default:
      TORCH_CHECK(
          false,
          "getHipdnnDataType() not supported for ",
          toString(tensor.scalar_type()));
  }
}

inline std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
createTensorAttributes(const Tensor& t) {
  auto tensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  tensor->set_dim(t.sizes().vec()).set_data_type(getHipdnnDataType(t));
  tensor->set_stride(t.strides().vec());
  return tensor;
}

} // namespace at::native
