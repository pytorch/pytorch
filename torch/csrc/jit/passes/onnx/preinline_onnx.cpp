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

				if (fun_type->function()->name() == "interpolate") {
          cur->removeInput(0);
//          cur->input(0)->node()->destroy();
			    Node* interpolate_node =  block->owningGraph()->create(Symbol::fromQualString("aten::__interpolate"), {cur->inputs()}, cur->outputs().size());
		      interpolate_node->output()->copyMetadata(cur->output());
		      interpolate_node->insertAfter(cur);
		      cur->replaceAllUsesWith(interpolate_node);
		      cur->removeAllInputs();
		      cur->destroy();
          return;
	      }
				Block* block = fun_type->function()->graph()->block();
				PreInlineCalls(block);
//        std::cout<<"callee->optimized_graph()->toString() ============= " << fun_type->function()->graph()->toString() << "\n";
      } break;
//      case prim::CallMethod: {
//        const std::string& name = cur->s(attr::name);
//        std::cout<<"method name ============== " << name << "\n";
//        if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
//          auto function = class_type->getMethod(name);
//          if (!function->isGraphFunction()) {
//            continue;
//          }
//          GRAPH_UPDATE("Inlining method '", function->name(), "' to ", *cur);
//          GRAPH_UPDATE("Function body: ", *function->optimized_graph());
//          preinlineCallTo(cur, function);
//        }
//      } break;
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