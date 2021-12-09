#include "lazy_tensor_core/csrc/ops/addcmul.h"

#include <sstream>


#include "lazy_tensor_core/csrc/lazy_graph_executor.h"
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

  std::string Addcmul::ToString() const {
    std::stringstream ss;
    ss << TsNode::ToString();
    return ss.str();
  }

  Addcmul::Addcmul(const torch::lazy::Value& self, const torch::lazy::Value& tensor1, const torch::lazy::Value& tensor2, const at::Scalar& value, std::vector<torch::lazy::Shape>&& shapes)
    : TsNode(torch::lazy::OpKind(at::aten::addcmul),
            {self, tensor1, tensor2, LazyGraphExecutor::Get()->GetIrValueForScalar(value, torch::lazy::getBackend()->GetBackendDevice(c10::Device(c10::kCPU, 0)))}, std::move(shapes),
            /* num_outputs */ 1) 
  {}

  torch::lazy::TSOpVector Addcmul::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                    torch::lazy::TSLoweringContext* loctx) const {
        std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve(3);
    kwarguments.reserve(1);
    size_t i = 0;
    arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
    arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
    arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
    kwarguments.emplace_back("value", loctx->GetOutputOp(operand(i++)));
    torch::lazy::TSOpVector addcmul_out = torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    CHECK_EQ(addcmul_out.size(), 1);

    // TODO: need to call GenerateClone sometimes? Or else return LowerBuiltIn() directly
    return addcmul_out;
  }

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors

