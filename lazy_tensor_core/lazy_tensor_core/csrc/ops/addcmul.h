
#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include "torch/csrc/lazy/ts_backend/ts_node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Addcmul : public torch::lazy::TsNode {
 public:

    Addcmul(const torch::lazy::Value& self, const torch::lazy::Value& tensor1, const torch::lazy::Value& tensor2, const at::Scalar& value, std::vector<torch::lazy::Shape>&& shapes);

    std::string ToString() const override;
    torch::lazy::TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                    torch::lazy::TSLoweringContext* loctx) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
