#include "torch/csrc/jit/passes/init_pass.h"
#include <unordered_map>

namespace torch { namespace jit {

namespace {

using constructor_type = std::function<Node*(Graph*, PythonOp*)>;

Node * trivial_ctor(Graph *graph, PythonOp *p) {
  return graph->create(stringToSymbol(p->name()), p->inputs());
}

Node * chunk_ctor(Graph * graph, PythonOp * p) {
  auto num_chunks = PyLong_AsLong(p->scalar_args[0]);
  auto dim = PyLong_AsLong(p->scalar_args[1]);
  return graph->createChunk(p->input(),num_chunks,dim);
}

std::unordered_map<std::string, constructor_type> constructors = {
  {"Add",         trivial_ctor},
  {"Mul",         trivial_ctor},
  {"Sigmoid",     trivial_ctor},
  {"Tanh",        trivial_ctor},
  {"Chunk",       chunk_ctor},
  {"Negate",      trivial_ctor},
};

} // anonymous namespace

void MatchJITOps(std::shared_ptr<Graph>& graph) {
  auto nodes = graph->nodes();
  for(auto it = nodes.begin(); it != nodes.end(); ++it) {
    PythonOp *p = (*it)->cast<PythonOp>();
    if (!p) {
      continue;
    }

    auto ctor_it = constructors.find(p->name());
    if (ctor_it == constructors.end()) {
      continue;
    }
    auto& constructor = ctor_it->second;

    // Set up the Node that will replace p
    auto new_op = constructor(graph.get(), p);
    new_op->insertAfter(p);

    if (new_op->hasMultipleOutputs()) {
      auto uses = p->uses();
      for (auto & use : uses) {
        // Invariant: Node replacements never make use of opaque handles,
        // so we drop them before we do the node replacement.  E.g., if
        // an op previously returned %1 : Tensor, %2 : Handle, after
        // replacement it will only return %1 : Tensor.
        // NB: This code implies that init_pass doesn't work with backwards.
        if (use.user->type()->kind() != TypeKind::HandleType) continue;
        JIT_ASSERT(use.user->uses().size() == 0);
        use.user->destroy();
      }
      p->replaceAllUsesWith(new_op);
    } else {
      // PythonOps are always multireturn.
      // We need to replace the tensor Select node and remove the handle Select.
      JIT_ASSERT(p->uses().size() == 2);
      auto tensor_select = p->uses()[0].user;
      auto handle_select = p->uses()[1].user;
      JIT_ASSERT(tensor_select->type()->kind() == TypeKind::TensorType);
      JIT_ASSERT(handle_select->type()->kind() == TypeKind::HandleType);
      JIT_ASSERT(tensor_select->kind() == kSelect);
      JIT_ASSERT(handle_select->kind() == kSelect);
      JIT_ASSERT(handle_select->uses().size() == 0);
      new_op->setType(tensor_select->type());
      tensor_select->replaceAllUsesWith(new_op);
      tensor_select->destroy();
      handle_select->destroy();
    }

    it.destroyCurrent();
  }
}

}}
