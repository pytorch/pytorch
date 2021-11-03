#include "lazy_tensor_core/csrc/ops/bernoulli.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Bernoulli::Bernoulli(const torch::lazy::Value& probability, const torch::lazy::Value& seed,
                     lazy_tensors::Shape shape)
    : TsNode(torch::lazy::OpKind(at::aten::bernoulli), {probability, seed},
           {std::move(shape)}) {}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
