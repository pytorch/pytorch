#include "lazy_tensor_core/csrc/ops/get_dimensions_size.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

GetDimensionsSize::GetDimensionsSize(const torch::lazy::Value& input,
                                     std::vector<int64_t> dimensions)
    : TsNode(ltc_get_dimensions_size, {input},
             {torch::lazy::Shape(c10::ScalarType::Int, {})},
             /*num_outputs=*/1, torch::lazy::MHash(dimensions)),
      dimensions_(std::move(dimensions)) {}

std::string GetDimensionsSize::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", dimensions=("
     << c10::Join(", ", dimensions_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
