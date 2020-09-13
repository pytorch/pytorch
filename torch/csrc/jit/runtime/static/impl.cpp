#include <torch/csrc/jit/runtime/static/impl.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

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

  // remove unused input 0 from graph
  if (graph_->inputs().at(0)->type()->is_module()) {
    if (!graph_->inputs().at(0)->hasUses()) {
      graph_->eraseInput(0);
    }
  }

  // fill constant_table_ and operator_table_
  for (Node* node : graph_->nodes()) {
    switch (node->kind()) {
      case prim::Constant:
        CHECK(node->output()->type()->kind() != FunctionType::Kind);
        constant_table_[node->output()] = toIValue(node->output()).value();
        break;
      case prim::ListConstruct:
        nodes_.emplace_back(node, nullptr);
        break;
      case prim::TupleConstruct:
        nodes_.emplace_back(node, nullptr);
        break;
      default: {
        const Operator& op = node->getOperator();
        CHECK(op.hasOperation());
        nodes_.emplace_back(node, op.getOperation(node));
      }
    }
  }
}

void StaticRuntime::getInputIValues(
    Node* node,
    const ConstantMap& ws,
    std::vector<IValue>& stack) const {
  const size_t size = node->inputs().size();
  stack.reserve(size);
  for (size_t i = 0; i < size; i++) {
    Value* v = node->inputs()[i];
    auto f = constant_table_.find(v);
    if (f == constant_table_.end()) {
      auto f_ws = ws.find(v);
      TORCH_CHECK(
          f_ws != ws.end(),
          "Workspace does not contain Value ",
          v->debugName());
      stack.emplace_back(f_ws->second);
    } else {
      stack.emplace_back(f->second);
    }
  }
}

void StaticRuntime::runNodes(ConstantMap& workspace) const {
  std::vector<IValue> stack;
  for (const auto& p : nodes_) {
    Node* node = p.first;
    const Operation& op = p.second;
    getInputIValues(node, workspace, stack);
    VLOG(1) << node->kind().toDisplayString();

    switch (node->kind()) {
      case prim::ListConstruct: {
        listConstruct(
            stack,
            node->output()->type()->expect<ListType>(),
            node->inputs().size());
      } break;
      case prim::TupleConstruct: {
        bool named =
            node->output()->type()->expect<TupleType>()->name().has_value();
        if (named) {
          namedTupleConstruct(
              stack,
              node->output()->type()->expect<TupleType>(),
              node->inputs().size());
        } else {
          tupleConstruct(stack, node->inputs().size());
        }
      } break;
      default: {
        DCHECK(op);
        op(&stack);
        break;
      }
    }

    DCHECK_EQ(stack.size(), node->outputs().size());
    for (auto i = 0; i < node->outputs().size(); i++) {
      workspace[node->outputs()[i]] = stack[i];
    }
    stack.clear();
  }
}

std::vector<at::Tensor> StaticRuntime::run(
    const std::vector<at::Tensor>& inps) const {
  // Container for inputs, outputs, and activations (excluding parameters)
  ConstantMap workspace_;

  int start = 0;
  if (graph_->inputs().size() != inps.size()) {
    start = 1;
    CHECK_EQ(graph_->inputs().size(), inps.size() + 1);
    CHECK((graph_->inputs().at(0)->type()->is_module()));
    workspace_.emplace(graph_->inputs()[0], module_._ivalue());
  }

  for (size_t i = 0; i < inps.size(); i++) {
    workspace_.emplace(graph_->inputs()[i + start], inps[i]);
  }

  runNodes(workspace_);

  std::vector<at::Tensor> out;
  for (Value* output : graph_->outputs()) {
    const IValue& v = workspace_[output];
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
