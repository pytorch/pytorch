#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

StaticRuntime::StaticRuntime(
    const torch::jit::Module& m,
    std::shared_ptr<torch::jit::Graph> g)
    : graph_(std::move(g)), module_(m.deepcopy()) {
  Inline(*graph_);
  ConstantPropagation(graph_);
  for (auto n : graph_->nodes()) {
    if (n->kind() == c10::Symbol::fromQualString("prim::GetAttr")) {
      throw std::runtime_error("Cannot accelerate unfrozen graphs");
    }
  }
}

std::vector<at::Tensor> StaticRuntime::run(
    const std::vector<at::Tensor>& inps) const {
  std::vector<torch::jit::IValue> stack;
  if (graph_->inputs().at(0)->type()->is_module()) {
    stack.emplace_back(module_._ivalue());
  }
  for (const auto& inp : inps) {
    stack.emplace_back(inp);
  }
  torch::jit::Code code(graph_, "");
  torch::jit::InterpreterState interp(code);
  interp.run(stack);
  std::vector<at::Tensor> out;
  for (const auto& v : stack) {
    if (v.isTuple()) {
      auto t = v.toTuple();
      for (const auto& el : t->elements()) {
        out.emplace_back(el.toTensor());
      }
    } else {
      out.emplace_back(v.toTensor());
    }
  }
  return out;
}
} // namespace jit
} // namespace torch
