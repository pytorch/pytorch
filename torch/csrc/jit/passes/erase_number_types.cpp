#include "torch/csrc/jit/passes/erase_number_types.h"

namespace torch { namespace jit {

static void EraseNumberTypesOnBlock(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      EraseNumberTypesOnBlock(sub);
    }
    switch (it->kind()) {
      case prim::Constant: {
        it->output()->inferTypeFrom(it->t(attr::value));
      } break;
      case prim::TensorToNum:
      case prim::NumToTensor: {
        it->output()->replaceAllUsesWith(it->inputs()[0]);
        it.destroyCurrent();
      } break;
      default: {
        for(auto o : it->outputs()) {
          if (o->type()->isSubtypeOf(*NumberType::get())) {
            o->setType(DynamicType::get());
          }
        }
      } break;
    }
  }
}

void EraseNumberTypes(const std::shared_ptr<Graph>& graph) {
  EraseNumberTypesOnBlock(graph->block());
}

}}
