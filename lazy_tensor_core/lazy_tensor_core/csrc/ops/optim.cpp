#include "lazy_tensor_core/csrc/ops/optim.h"

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/util.h>

#include "lazy/core/ir.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyLazyIr.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Optim::Optim(const torch::lazy::Value& input, const torch::lazy::Value& grad)
    : torch::lazy::TsNode(torch::lazy::OpKind(c10::Symbol::fromQualString("lazy_cuda::optim")),
                          {input, grad},
                          /*num_outputs=*/1) {
  SetShapeDeferred(
    [&]() { 
      auto& shape = GetShapeFromTsValue(input);
      std::vector<int64_t> sizes(shape.sizes().begin(), shape.sizes().end());
      return Shape(shape.scalar_type(), sizes); 
    }
  );
}

std::string Optim::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString();
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
