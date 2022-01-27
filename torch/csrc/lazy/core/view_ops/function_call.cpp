#include <torch/csrc/lazy/core/view_ops/function_call.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include "jit/api/function_impl.h"

namespace torch {
namespace lazy {

FunctionCall::FunctionCall(torch::lazy::OpList values, c10::ArrayRef<c10::IValue> consts, torch::jit::Function* f, std::vector<Shape> shapes)
    : torch::lazy::TsNode(torch::lazy::OpKind(c10::prim::CallFunction), values, std::move(shapes),
                          f->getSchema().returns().size(), torch::lazy::MHash(reinterpret_cast<int64_t>(f))),
      consts_(consts.cbegin(), consts.cend()),
      function_(f) {
        TORCH_CHECK(function_->isGraphFunction(), "Not a Graph Function!");
      }

std::string FunctionCall::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", function=" << function_;
  return ss.str();
}


torch::lazy::TSOpVector FunctionCall::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  
  auto gfunc = dynamic_cast<torch::jit::GraphFunction*>(function_);
  
  auto graph = function->graph();
  auto fn_constant = graph->insertNode(graph->create(c10::prim::Constant))
                           ->s_(c10::attr::name, function_->name())
                           ->output()
                           ->setType(c10::FunctionType::create(function_));
  std::vector<torch::jit::Value*> func_call_inputs = {fn_constant};
  for (auto op: operands()) {
    func_call_inputs.push_back(loctx->GetOutputOp(op));
  }

  for (auto cnst: consts_) {
    auto cnst_val = graph->insertConstant(cnst);
    std::cerr << "adding const " << cnst_val << std::endl; 
    func_call_inputs.push_back(cnst_val);
  }

  auto result =
      graph->insertNode(graph->create(c10::prim::CallFunction, func_call_inputs))
          ->output()
          ->setType(gfunc->graph()->outputs()[0]->type());

  return {result};
}

}  // namespace ops
}  // namespace ir
