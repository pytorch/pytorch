#include <torch/csrc/jit/passes/lower_grad_of.h>

#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

void LowerGradOf(Graph& g) {
  for (auto it = g.nodes().begin(); it != g.nodes().end(); ++it) {
    if (it->kind() == prim::GradOf) {
      // if any_defined(inputs):
      //  outputs = <original_computation>
      // else:
      //  outputs = autograd zero tensors
      WithInsertPoint guard(*it);
      auto cond = g.insertNode(g.create(prim::AutogradAnyNonZero, it->inputs()))
                      ->output()
                      ->setType(IntType::get());
      auto if_stat =
          g.insertNode(g.create(prim::If, {cond}, it->outputs().size()));
      if_stat->addBlock()->cloneFrom(
          it->blocks().at(0), [](Value* v) { return v; });
      auto else_block = if_stat->addBlock();
      auto undef = g.createAutogradZero()
                       ->insertBefore(else_block->return_node())
                       ->output();
      for (size_t i = 0; i < it->outputs().size(); ++i) {
        else_block->registerOutput(undef);
        if_stat->outputs().at(i)->copyMetadata(it->outputs().at(i));
      }
      GRAPH_UPDATE("Replacing ", getHeader(*it), " with ", getHeader(if_stat));
      it->replaceAllUsesWith(if_stat);
      it.destroyCurrent();
    }
  }
}

} // namespace jit
} // namespace torch
