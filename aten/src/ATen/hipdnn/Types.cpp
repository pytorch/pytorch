#include <ATen/hipdnn/Types.h>
#include <hipdnn_frontend/Types.hpp>

#include <ATen/ATen.h>

namespace at { namespace native {

hipdnn_frontend::DataType getHipdnnDataType(const at::Tensor& tensor) {
  switch (tensor.scalar_type())
  {
    case at::kFloat:
      return hipdnn_frontend::DataType::FLOAT;
    case at::kHalf:
      return hipdnn_frontend::DataType::HALF;
    case at::kBFloat16:
      return hipdnn_frontend::DataType::BFLOAT16;
    default:
      std::string msg("getHipdnnDataType() not supported for ");
      msg += toString(tensor.scalar_type());
      throw std::runtime_error(msg);
  }
}

}}  // namespace at::native
