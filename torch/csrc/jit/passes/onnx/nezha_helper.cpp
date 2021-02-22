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
    printf("\n------ Print first module before: \n%s ------", module_1st.dump_to_str(true, false, true, 0).c_str());
    auto graph_bak = module_1st.get_method("forward").graph()->copy();
    updateGraph(graph_bak);
    const auto method_name = QualifiedName(*module_1st.type()->name(), "forward");
    module_1st.type()->unsafeRemoveMethod("forward");
    module_1st._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = module_1st._ivalue()->compilation_unit()->create_function(
        method_name, graph_bak);
    module_1st.type()->addMethod(fn);
    printf("\n------ Print first module after: \n%s ------", module_1st.dump_to_str(true, false, true, 0).c_str());

    update2ndGraph(module_2nd);
}


} // namespace jit
} // namespace torch
