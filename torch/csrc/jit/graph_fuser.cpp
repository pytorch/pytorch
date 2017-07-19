#include "torch/csrc/jit/graph_fuser.h"

namespace torch { namespace jit {

std::unordered_set<std::string> simple_mappable = {
  "Sigmoid",
  "Tanh",
  "Mul",
  "Add",
};

struct GraphFuser {
  std::unique_ptr<Graph> graph;
  GraphFuser(std::unique_ptr<Graph> graph)
  : graph(std::move(graph)) {}
  void replacePythonOps() {
    auto nodes = graph->nodes();
    for(auto it = nodes.begin(), end = nodes.end(); it != end; ++it) {
      if(auto p = it->cast<PythonOp>()) {
        std::string name = p->name();
        if(simple_mappable.count(name) > 0) {
          auto new_op = graph->create<SimpleMap>(name,p->inputs());
          new_op->insertAfter(p);
          JIT_ASSERT(1 == p->uses().size());
          auto single_select = p->uses()[0].user;
          single_select->replaceAllUsesWith(new_op);
          single_select->eraseFromParent();
          it.eraseCurrentFromParent(); //erasing p directly would invalidate iterator
        }
      }
    }

  }
  std::unique_ptr<Graph> run() {
    replacePythonOps();
    return std::move(graph);
  }
};

std::unique_ptr<Graph> FuseGraph(std::unique_ptr<Graph> graph) {
  GraphFuser gf(std::move(graph));
  return gf.run();
}

}}
