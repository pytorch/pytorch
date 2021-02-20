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

static void releaseGraph(std::shared_ptr<Graph> graph){
    if (graph == nullptr){
        return;
    }

    for (auto node : graph->block()->nodes()) {
        printf("\n------ NeZha_TryUpdateModule:releaseGraph %s, scopeName: %s, debugName: %s ------", node->kind().toQualString(), node->scopeName().c_str(), node->inputs()[0]->debugName().c_str());
        auto tempVal = node->get(Symbol::fromQualString("prim::GetAttr"));
        if (tempVal != nullptr){
            printf("\n------ more: %s", node->get(Symbol::prim("name")).value().toString()->string().c_str());
        }
    }

    auto temporary_nodes = graph->block()->nodes();
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
            ++it) {
        printf("\n------ NeZha_TryUpdateModule:removeAllInputs - %s------", it->kind().toQualString());
        //it->removeAllInputs();
    }

    // for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
    //         ++it) {
    //     IValue ival(true);
    //     auto new_constant = it->owningGraph()->insertConstant(ival);
        // if (std::strcmp(it->kind().toQualString(), "prim::CallMethod") == 0){
        //     printf("\n------ updateGraph:removeAllInputs - %s------", it->kind().toQualString());
        //     it->removeAllInputs();
        // }
    //     it->output()->replaceAllUsesWith(new_constant);
    //     it.destroyCurrent();
    // }


    printf("\n------ Release graph ------");
} 

static void updateGraph(std::shared_ptr<Graph> graph){
    if (graph == nullptr){
        return;
    }
    
    auto temporary_nodes = graph->block()->nodes();
    generic_graph_node_list_iterator<Node> keyNode;

    int delNodesCount = 0;
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
            it++) {
        printf("\n------ delNodesCount: %d ------\n", delNodesCount);

        if (delNodesCount < 3) {
            keyNode = it;
            keyNode++;

            it->output()->replaceAllUsesWith(keyNode->outputs()[0]);
            it.destroyCurrent();
            delNodesCount++;
        }

        if (std::strcmp(it->kind().toQualString(), "prim::GetAttr") == 0){
            if (!it->output()->hasUses()){
                it.destroyCurrent();
            }
        }

        printf("\n------ check graph: ------\n");
        graph->dump();
    }
}

static void update2ndGraph(Module& module){
    // auto allGraphInputs = graph->inputs();
    // printf("\n------ all inputs: %ld ------\n", allGraphInputs.size());
    // for (int i = 0; i < allGraphInputs.size(); i++){
    //     printf("\n------ input: %d, %s ------\n", i, allGraphInputs[i]->debugName().c_str());
    // }

    // auto graph_input_node = allGraphInputs[1]->node();
    // printf("\n------ input.1 node: ------\n");
    // allGraphInputs[1]->node()->dump();

    std::shared_ptr<Graph> graph = module.get_method("forward").graph()->copy();
    auto temporary_nodes = graph->block()->nodes();

    int delNodesCount = 0;

    for (auto it = temporary_nodes.begin(); it != temporary_nodes.end();
            it++) {
        printf("\n------ delNodesCount: %d ------\n", delNodesCount);

        if (std::strcmp(it->kind().toQualString(), "prim::GetAttr") == 0){
            continue;
        }

        if (delNodesCount < 2) {
            it->output()->replaceAllUsesWith(graph->inputs()[1]);
            it.destroyCurrent();
            delNodesCount++;
        }

        printf("\n------ check graph: ------\n");
        graph->dump();
    }

    for (auto it = temporary_nodes.begin(); it != temporary_nodes.end();
        it++) {
        if (std::strcmp(it->kind().toQualString(), "prim::GetAttr") == 0){
            if (!it->output()->hasUses()){
                it.destroyCurrent();
            }
        }
    }

    printf("\n------ check graph: ------\n");
    graph->dump();

    const auto method_name = QualifiedName(*module.type()->name(), "forward");
    module.type()->unsafeRemoveMethod("forward");
    module._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = module._ivalue()->compilation_unit()->create_function(method_name, graph);
    module.type()->addMethod(fn);
    printf("\n------ Print second module after: \n%s ------", module.dump_to_str(true, false, false, 0).c_str());
    printf("\n------ Print second module details: %s ------\n", module.type()->name()->qualifiedName().c_str());    
}

void NeZha_TrySplitModule(
    Module& module_1st,
    Module& module_2nd) {

    printf("\n------ Start NeZha_TrySplitModule ------");

    // printf("\n------ Print first module before: \n%s ------", module_1st.dump_to_str(true, false, false, 0).c_str());
    auto graph_bak = module_1st.get_method("forward").graph()->copy();
    updateGraph(graph_bak);
    const auto method_name = QualifiedName(*module_1st.type()->name(), "forward");
    module_1st.type()->unsafeRemoveMethod("forward");
    module_1st._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = module_1st._ivalue()->compilation_unit()->create_function(
        method_name, graph_bak);
    module_1st.type()->addMethod(fn);
    // printf("\n------ Print first module after: \n%s ------", module_1st.dump_to_str(true, true, true, 0).c_str());

    update2ndGraph(module_2nd);
}

void NeZha_TryUpdateModule(
    Module& dst_module,
    std::shared_ptr<Graph>& src_graph) {

    printf("\n------ Start NeZha_TryUpdateModule ------");

    // printf("\n------ Print module before: \n%s ------", dst_module.dump_to_str(true, false, false, 0).c_str());
    auto graph_bak = dst_module.get_method("forward").graph()->copy();
    updateGraph(graph_bak);
    
    // releaseGraph(dst_module.get_method("forward").graph());
    
    const auto method_name = QualifiedName(*dst_module.type()->name(), "forward");
    dst_module.type()->unsafeRemoveMethod("forward");
    dst_module._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = dst_module._ivalue()->compilation_unit()->create_function(
        method_name, graph_bak);
    dst_module.type()->addMethod(fn);
    printf("\n------ Print module after: \n%s ------", dst_module.dump_to_str(true, false, false, 0).c_str());
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
