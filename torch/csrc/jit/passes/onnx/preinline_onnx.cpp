#include <torch/csrc/jit/passes/onnx/preinline_onnx.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <c10/util/Exception.h>

#include <c10/util/Optional.h>
#include <algorithm>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

void replaceFunctions(Node* to_replace, Function* callee) {
  if (callee->name() == "interpolate") {
    to_replace->removeInput(0);
    Node* interpolate_node =  to_replace->owningGraph()->create(Symbol::fromQualString("aten::__interpolate"), {to_replace->inputs()}, to_replace->outputs().size());
    interpolate_node->output()->copyMetadata(to_replace->output());
    interpolate_node->insertAfter(to_replace);
    to_replace->replaceAllUsesWith(interpolate_node);
    to_replace->removeAllInputs();
    to_replace->destroy();
    return;
  }
}


void PreInlineCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        replaceFunctions(cur, fun_type->function());
      } break;
      default: {
        for (auto b : cur->blocks()) {
          PreInlineCalls(b);
        }
      } break;
    }
  }
}

void PreInlineONNX(Graph& graph) {
  GRAPH_DUMP("Before PreInlining: ", &graph);
  PreInlineCalls(graph.block());
  GRAPH_DUMP("After PreInlining: ", &graph);
}

} // namespace jit
} // namespace torch