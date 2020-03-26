#include <torch/csrc/jit/passes/onnx/pre_inline_onnx.h>
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


std::vector<Value*> preinlineCallTo(Node* to_replace, Function* callee) {
  WithInsertPoint guard(to_replace);
  TORCH_INTERNAL_ASSERT(callee->isGraphFunction());
  std::unordered_map<Value*, Value*> value_map;

  std::cout << "iiiiiiiiiincoming " << callee->optimized_graph()->inputs().size() << "\n";

  auto new_outputs = insertGraph(
      *to_replace->owningGraph(),
      *(callee->optimized_graph()),
      to_replace->inputs());

	std::cout << "siiiiiiiiiize " << to_replace->input()->node()->kind().toQualString();
  return new_outputs;
}


void preinlineCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        cur->removeInput(0);

        std::cout<<"func name ============= " << fun_type->function()->name() << "\n";

				if (fun_type->function()->name() == "interpolate") {
			    Node* interpolate_node =  block->owningGraph()->create(Symbol::fromQualString("aten::__interpolate"), {cur->inputs()}, cur->outputs().size());
		      interpolate_node->output()->copyMetadata(cur->output());
		      interpolate_node->insertAfter(cur);
		      cur->replaceAllUsesWith(interpolate_node);
//		      cur->input(0)->node()->destroy();
		      cur->removeAllInputs();
		      cur->destroy();
	      }

        GRAPH_UPDATE(
            "Inlining function '", fun_type->function()->name(), "' to ", *cur);
        GRAPH_UPDATE(
            "Function body: ", *fun_type->function()->optimized_graph());

//        preinlineCalls(fun_type->function()->optimized_graph()->block());

//        preinlineCallTo(cur, fun_type->function());

        auto to_replace = cur;
        auto callee = fun_type->function();
        WithInsertPoint guard(to_replace);
			  TORCH_INTERNAL_ASSERT(callee->isGraphFunction());
			  std::unordered_map<Value*, Value*> value_map;

			  std::cout << "iiiiiiiiiincoming " << callee->optimized_graph()->toString() << "\n";

				int i = 0;
		    for (const auto& node : callee->optimized_graph()->nodes()) {
		      if (i == 0){
            std::cout<<"staaaaaaaaaaaaaaaaaaaaaart ^666666666666666666&&&&&&&&& " << i << "\n";
            Node* cur = node;
		        AT_ASSERT(cur->kind() == prim::Constant);
		        auto fun_type =
		            cur->output()->type()->expect<FunctionType>();
//						cur->removeInput(0);

	          std::cout<<"func name ============= " << fun_type->function()->name() << "\n";

						if (fun_type->function()->name() == "interpolate") {
					    Node* interpolate_node =  block->owningGraph()->create(Symbol::fromQualString("aten::__interpolate"), {cur->inputs()}, cur->outputs().size());
				      interpolate_node->output()->copyMetadata(cur->output());
				      interpolate_node->insertAfter(cur);
				      cur->replaceAllUsesWith(interpolate_node);
		//		      cur->input(0)->node()->destroy();
				      cur->removeAllInputs();
				      cur->destroy();
			      }
		      }
		      i+=1;
		    }
//
//				preinlineCalls(callee->optimized_graph()->block());
//			  auto new_outputs = insertGraph(
//			      *to_replace->owningGraph(),
//			      *(callee->optimized_graph()),
//			      to_replace->inputs());
//
//				std::cout << "siiiiiiiiiize " << to_replace->input()->node()->kind().toQualString();
//			  return new_outputs;

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
//        for (auto b : cur->blocks()) {
//          preinlineCalls(b);
//        }
      } break;
    }
  }
}

void PreInlineONNX(Graph& graph) {
  std::cout<<"start ------------ " << "\n";

  GRAPH_DUMP("Before PreInlining: ", &graph);
  preinlineCalls(graph.block());
  GRAPH_DUMP("After PreInlining: ", &graph);
}


} // namespace jit
} // namespace torch