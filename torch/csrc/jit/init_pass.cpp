#include "torch/csrc/jit/init_pass.h"
#include <unordered_map>

namespace torch { namespace jit {

namespace {

using constructor_type = std::function<Node*(Graph*, PythonOp*)>;

template<typename T>
Node * trivial_ctor(Graph *graph, PythonOp *p) {
  return graph->create<T>();
}

Node * chunk_ctor(Graph * graph, PythonOp * p) {
  auto num_chunks = PyLong_AsLong(p->scalar_args[0]);
  auto dim = PyLong_AsLong(p->scalar_args[1]);
  return graph->create<Chunk>(num_chunks,dim);
}

std::unordered_map<std::string, constructor_type> constructors = {
  {"Add",         trivial_ctor<Add>},
  {"Mul",         trivial_ctor<Mul>},
  {"Sigmoid",     trivial_ctor<Sigmoid>},
  {"Tanh",        trivial_ctor<Tanh>},
  {"Chunk",       chunk_ctor},
};

} // anonymous namespace

void MatchJITOps(std::unique_ptr<Graph>& graph) {
  auto & nodes = graph->nodes();
  auto it = nodes.begin();
  while (it != nodes.end()) {
    PythonOp *p = (*it)->cast<PythonOp>();
    if (!p) {
      ++it;
      continue;
    }

    auto ctor_it = constructors.find(p->name());
    if (ctor_it == constructors.end()) {
      ++it;
      continue;
    }
    auto& constructor = ctor_it->second;

    // Set up the Node that will replace p
    auto new_op = constructor(graph.get(), p);
    for (Node *input : p->inputs()) {
      new_op->addInput(input);
    }
    new_op->insertAfter(p);

    if(new_op->hasMultipleOutputs()) {
      new_op->setType(p->type());
      p->replaceAllUsesWith(new_op);
    } else {
      // PythonOps are always multireturn. We need to remove the Select node.
      JIT_ASSERT(p->uses().size() == 1);
      auto single_select = p->uses()[0].user;
      JIT_ASSERT(single_select->kind() == NodeKind::Select);
      new_op->setType(single_select->type());
      single_select->replaceAllUsesWith(new_op);
      single_select->destroy();
    }

    // Erasing p directly would invalidate iterator
    ++it;
    p->destroy();
  }
}

}}
