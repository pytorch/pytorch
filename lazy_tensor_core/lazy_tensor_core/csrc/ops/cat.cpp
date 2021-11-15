#include "lazy_tensor_core/csrc/ops/cat.h"
#include <torch/csrc/lazy/core/shape.h>

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensors/computation_client/util.h"
namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Cat::Cat(std::vector<torch::lazy::Value> values, int64_t dim,
        std::vector<torch::lazy::Shape>&& shapes)
    : TsNode(torch::lazy::OpKind(at::aten::cat), values,
             std::move(shapes), /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

std::string Cat::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
