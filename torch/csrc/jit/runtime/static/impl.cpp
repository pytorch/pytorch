#include <torch/csrc/jit/runtime/static/impl.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace torch {
namespace jit {

using c10::DispatchKey;
using c10::RegisterOperators;

std::shared_ptr<torch::jit::Graph> PrepareForStaticRuntime(
    std::shared_ptr<torch::jit::Graph> g) {
  Inline(*g);
  ConstantPropagation(g);
  Canonicalize(g);
  ConstantPropagation(g);
  RemoveTensorMutation(g);
  ConstantPropagation(g);

  for (auto n : g->nodes()) {
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

  // remove unused input 0 from graph
  if (g->inputs().at(0)->type()->is_module()) {
    if (!g->inputs().at(0)->hasUses()) {
      g->eraseInput(0);
    }
  }

  return g;
}

std::shared_ptr<torch::jit::Graph> PrepareForStaticRuntime(
    const torch::jit::Module& m) {
  auto module = m.copy();
  module.eval();
  module = freeze_module(module);
  auto g = module.get_method("forward").graph();
  return PrepareForStaticRuntime(g);
}

StaticRuntime::StaticRuntime(std::shared_ptr<torch::jit::Graph> g) : graph_(g) {
  // fill workspace_ with constants
  for (Node* node : graph_->nodes()) {
    if (node->kind() == prim::Constant) {
      CHECK(node->output()->type()->kind() != FunctionType::Kind);
      workspace_[node->output()] = toIValue(node->output()).value();
    } else {
      nodes_.emplace_back(node);
    }
  }
}

std::vector<at::Tensor> StaticRuntime::run(
    const std::vector<at::Tensor>& inps) const {
  // Container for inputs, outputs, and activations (excluding parameters)

  TORCH_INTERNAL_ASSERT(graph_->inputs().size() == inps.size());

  for (size_t i = 0; i < inps.size(); i++) {
    workspace_[graph_->inputs()[i]] = inps[i];
  }

  for (const auto& n : nodes_) {
    n.run(workspace_);
  }

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

ProcessedNode::ProcessedNode(Node* node) : node_(node) {
  if (node->kind() != prim::ListConstruct &&
      node->kind() != prim::TupleConstruct) {
    const Operator& op = node->getOperator();
    CHECK(op.hasOperation());
    op_ = op.getOperation(node);
  }
  if (canRunOutOfPlace(node)) {
    fn_ = getOutOfPlaceOperation(node);
  }
}

void ProcessedNode::run(StaticRuntime::ConstantMap& workspace) const {
  if (!fn_) {
    std::vector<IValue> stack;
    const size_t size = node_->inputs().size();
    stack.reserve(size);
    for (size_t i = 0; i < size; i++) {
      Value* v = node_->inputs()[i];
      auto f = workspace.find(v);
      TORCH_CHECK(
          f != workspace.end(),
          "Workspace does not contain Value ",
          v->debugName());
      stack.emplace_back(f->second);
    }
    if (op_) {
      op_->operator()(&stack);
    } else {
      if (node_->kind() == prim::ListConstruct) {
        listConstruct(
            stack,
            node_->output()->type()->expect<ListType>(),
            node_->inputs().size());
      } else if (node_->kind() == prim::TupleConstruct) {
        bool named =
            node_->output()->type()->expect<TupleType>()->name().has_value();
        if (named) {
          namedTupleConstruct(
              stack,
              node_->output()->type()->expect<TupleType>(),
              node_->inputs().size());
        } else {
          tupleConstruct(stack, node_->inputs().size());
        }
      } else {
        TORCH_CHECK(0, "Unhandled operation!", node_->kind().toQualString());
      }
    }
    DCHECK_EQ(stack.size(), node_->outputs().size());
    for (auto i = 0; i < node_->outputs().size(); i++) {
      workspace[node_->outputs()[i]] = stack[i];
    }
  } else {
    fn_->operator()(workspace);
  }
}

} // namespace jit
} // namespace torch
