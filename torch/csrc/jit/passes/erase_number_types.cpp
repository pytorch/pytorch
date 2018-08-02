#include "torch/csrc/jit/passes/erase_number_types.h"
#include "torch/csrc/jit/constants.h"

namespace torch { namespace jit {

static void EraseNumberTypesOnBlock(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      EraseNumberTypesOnBlock(sub);
    }
    switch (it->kind()) {
      case prim::Constant: {
        // remove primitive constants, replacing with tensor equivalent
        // ONNX does not support non-tensor constants
        if(it->output()->type()->isSubtypeOf(NumberType::get())) {
          auto s = *constant_as<at::Scalar>(it->output());
          WithInsertPoint guard(*it);
          Value* r = insertConstant(*block->owningGraph(), s.toTensor());
          it->output()->replaceAllUsesWith(r);
        }
      } break;
      case prim::TensorToNum:
      case prim::NumToTensor: {
        it->output()->replaceAllUsesWith(it->inputs()[0]);
        // Let DCE cleanup
      } break;
      default: {
        for(auto o : it->outputs()) {
          if (o->type()->isSubtypeOf(NumberType::get())) {
            o->setType(TensorType::fromNumberType(o->type()));
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
