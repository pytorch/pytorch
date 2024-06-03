#include <torch/csrc/jit/passes/erase_number_types.h>

#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <ATen/ScalarOps.h>

namespace torch {
namespace jit {

static void SetNumTypeToTensorType(Value* v) {
  if (v->type()->isSubtypeOf(*NumberType::get())) {
    v->setType(TensorType::fromNumberType(*v->type()));
  } else if (v->type()->isSubtypeOf(*BoolType::get())) {
    v->setType(TensorType::fromBoolType());
  }
}

void EraseNumberTypesOnBlock(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto inp : it->inputs()) {
      SetNumTypeToTensorType(inp);
    }
    for (auto sub : it->blocks()) {
      EraseNumberTypesOnBlock(sub);
    }
    switch (it->kind()) {
      case prim::Constant: {
        // remove primitive constants, replacing with tensor equivalent
        // ONNX does not support non-tensor constants
        if (it->output()->type()->isSubtypeOf(*NumberType::get()) ||
            it->output()->type()->isSubtypeOf(*BoolType::get())) {
          at::Scalar s;
          if (it->output()->type()->isSubtypeOf(*BoolType::get())) {
            s = *constant_as<bool>(it->output());
          } else {
            s = *constant_as<at::Scalar>(it->output());
          }

          WithInsertPoint guard(*it);
          Value* r = block->owningGraph()->insertConstant(
              scalar_to_tensor(s), c10::nullopt, it->scope());
          r->copyMetadata(it->output());
          it->output()->replaceAllUsesWith(r);
          it.destroyCurrent();
        }
      } break;
      case aten::Bool:
      case aten::Float:
      case aten::Int:
      case aten::FloatImplicit:
      case aten::IntImplicit:
      case aten::ScalarImplicit:
      case prim::NumToTensor: {
        it->output()->replaceAllUsesWith(it->inputs()[0]);
        it.destroyCurrent();
      } break;
      default: {
        for (auto o : it->outputs()) {
          SetNumTypeToTensorType(o);
        }
      } break;
    }
  }
}

void EraseNumberTypes(const std::shared_ptr<Graph>& graph) {
  for (auto inp : graph->inputs()) {
    SetNumTypeToTensorType(inp);
  }
  EraseNumberTypesOnBlock(graph->block());
  GRAPH_DUMP("After EraseNumberTypes: ", graph);
}
} // namespace jit
} // namespace torch
