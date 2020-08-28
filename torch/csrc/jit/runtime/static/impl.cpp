#include <torch/csrc/jit/runtime/static/impl.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

using c10::DispatchKey;
using c10::RegisterOperators;

static auto reg =
    RegisterOperators()
        .op("static::add(Tensor a, Tensor b) -> Tensor",
            RegisterOperators::options().kernel(
                DispatchKey::CPU,
                [](at::Tensor a, at::Tensor b) -> at::Tensor { return a + b; }))
        .op("static::mul.a(Tensor a, Tensor b) -> Tensor",
            RegisterOperators::options().kernel(
                DispatchKey::CPU,
                [](at::Tensor a, at::Tensor b) -> at::Tensor { return a * b; }))
        .op("static::mul.b(Tensor a, int b) -> Tensor",
            RegisterOperators::options().kernel(
                DispatchKey::CPU,
                [](at::Tensor a, int64_t b) -> at::Tensor { return a * b; }));

#define SUPPORTED_OPS(F) \
  F(aten::__getitem__)   \
  F(aten::add)           \
  F(aten::addmm)         \
  F(aten::bmm)           \
  F(aten::cat)           \
  F(aten::clamp)         \
  F(aten::contiguous)    \
  F(aten::div)           \
  F(aten::flatten)       \
  F(aten::index_put_)    \
  F(aten::isnan)         \
  F(aten::matmul)        \
  F(aten::mul)           \
  F(aten::permute)       \
  F(aten::relu)          \
  F(aten::sigmoid)       \
  F(aten::size)          \
  F(aten::softmax)       \
  F(aten::t)             \
  F(aten::to)            \
  F(aten::transpose)     \
  F(aten::view)          \
  F(prim::Constant)      \
  F(prim::ListConstruct) \
  F(prim::TupleConstruct)

StaticRuntime::StaticRuntime(const torch::jit::Module& m)
    : module_(m.copy()), graph_(nullptr) {
  module_.eval();
  module_ = freeze_module(module_);
  graph_ = module_.get_method("forward").graph();

  Inline(*graph_);
  ConstantPropagation(graph_);
  Canonicalize(graph_);
  ConstantPropagation(graph_);
  RemoveTensorMutation(graph_);
  ConstantPropagation(graph_);

  for (auto n : graph_->nodes()) {
    if (n->kind() == c10::Symbol::fromQualString("prim::GetAttr")) {
      throw std::runtime_error("Cannot accelerate unfrozen graphs");
    }
    bool supported = false;
#define X(_)                                          \
  if (n->kind() == c10::Symbol::fromQualString(#_)) { \
    supported = true;                                 \
  }
    SUPPORTED_OPS(X)
#undef X
    if (!supported) {
      throw std::runtime_error(
          std::string("Unsupported operation: ") + n->kind().toQualString());
    }
  }

  SubgraphRewriter sr;
  sr.RegisterRewritePattern(
      R"IR(
  graph(%x, %w, %s):
    %r = aten::add(%x, %w, %s)
    return (%r))IR",
      R"IR(
  graph(%x, %w, %s):
    %y = static::add(%x, %w)
    %r = static::mul(%y, %s)
    return (%r))IR");
  sr.runOnGraph(graph_);
  code_ = std::make_unique<Code>(graph_, "");
  interp_ = std::make_unique<InterpreterState>(*code_);
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

  interp_->run(stack);
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
