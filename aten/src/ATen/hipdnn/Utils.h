#pragma once

#include <ATen/Tensor.h>
#include <ATen/hipdnn/Types.h>
#include <hipdnn_frontend.hpp>
#include <memory>

namespace at::native {

inline std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
createTensorAttributes(const Tensor& t) {
  auto tensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  tensor->set_dim(t.sizes().vec()).set_data_type(getHipdnnDataType(t));
  tensor->set_stride(t.strides().vec());
  return tensor;
}

} // namespace at::native
