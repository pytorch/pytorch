#include <torch/csrc/jit/passes/onnx/nezha_helper.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <c10/util/Optional.h>
#include <algorithm>

#include <ATen/core/qualified_name.h>


namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}


void NeZha_TryUpdateModule(
    Module& dst_module,
    std::shared_ptr<Graph>& src_graph) {

    printf("\n------ Start NeZha_TryUpdateModule ------");
    auto my_graph = dst_module.get_method("forward").graph();

    for (auto node : my_graph->block()->nodes()) {
        printf("\n------ NeZha_TryUpdateModule:appendNode %s------", node->kind().toQualString());
    }

    auto temporary_nodes = my_graph->block()->nodes();
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
            ++it) {
        printf("\n------ NeZha_TryUpdateModule:removeAllInputs - %s------", it->kind().toQualString());
        it->removeAllInputs();
    }

    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
            ++it) {

        IValue ival(true);
        auto new_constant = it->owningGraph()->insertConstant(ival);

        it->output()->replaceAllUsesWith(new_constant);
        it.destroyCurrent();
    }

    const auto method_name = QualifiedName(*dst_module.type()->name(), "forward");
    dst_module.type()->unsafeRemoveMethod("forward");
    dst_module._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = dst_module._ivalue()->compilation_unit()->create_function(
        method_name, my_graph);
    dst_module.type()->addMethod(fn);
}

void NeZha_TryUpdateGraph(
    std::shared_ptr<Graph>& dst_graph,
    std::shared_ptr<Graph>& src_graph) {
    
    printf("\n------ Start NeZha_TryUpdateGraph ------");
    // auto env = [](Value* v) -> Value* {
    // AT_ERROR(
    //     "Graph::copy() encountered a use of a value " + v->debugName() +
    //     " not in scope. Run lint!");
    // };
    // printf("\n------ NeZha_TryUpdateGraph: Clone block to dst_graph ------");
    // dst_graph->block()->cloneFrom(src_graph->block(), env);

    for (auto node : src_graph->block()->nodes()) {
        // auto new_node = dst_graph->block()->appendNode(dst_graph->createClone(node, env));
        printf("\n------ NeZha_TryUpdateGraph:appendNode %s------", node->kind().toQualString());
        // for (size_t i = 0; i < node->outputs().size(); ++i) {
        // auto oo = node->outputs()[i];
        // auto no = new_node->outputs()[i];
        // local_map[oo] = no;
        // no->copyMetadata(oo);
        // }
    }

    // for (auto node : dst_graph->block()->nodes()) {
    auto temporary_nodes = dst_graph->block()->nodes();
    // auto temp_node = src_graph->block()->nodes().rend();
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
            ++it) {
        // auto new_node = dst_graph->block()->appendNode(dst_graph->createClone(node, env));
        // Node* node = *it;
        printf("\n------ NeZha_TryUpdateGraph:removeAllInputs - %s------", it->kind().toQualString());
        it->removeAllInputs();
    }

    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
            ++it) {

        IValue ival(true);
        auto new_constant = it->owningGraph()->insertConstant(ival);

        printf("\n------ NeZha_TryUpdateGraph:replaceAllUsesWith - %s------", it->kind().toQualString());
        it->output()->replaceAllUsesWith(new_constant);                
        printf("\n------ NeZha_TryUpdateGraph:destroy - %s------", it->kind().toQualString());
        // it->destroy();
        it.destroyCurrent();
    }

    for (auto node : dst_graph->block()->nodes()) {
        // auto new_node = dst_graph->block()->appendNode(dst_graph->createClone(node, env));
        printf("\n------ NeZha_TryUpdateGraph: after destroyed: %s------", node->kind().toQualString());
        // for (size_t i = 0; i < node->outputs().size(); ++i) {
        // auto oo = node->outputs()[i];
        // auto no = new_node->outputs()[i];
        // local_map[oo] = no;
        // no->copyMetadata(oo);
        // }
    }

    printf("\n------ End NeZha_TryUpdateGraph ------");
}


} // namespace jit
} // namespace torch
