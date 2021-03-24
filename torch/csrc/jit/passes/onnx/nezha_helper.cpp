#include <torch/csrc/jit/passes/onnx/nezha_helper.h>

#include <torch/script.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <c10/util/Optional.h>
#include <algorithm>

#include <ATen/core/qualified_name.h>

#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

// static py::object* py_onnx = nullptr;


static void export_to_onnx(Module module, std::string file_name, torch::Tensor inputs){
    // if (py_onnx == nullptr) {
    //     *py_onnx = py::module::import("torch.onnx");
    // }
    py::object py_onnx = py::module::import("torch.onnx");
    py_onnx.attr("export_c_module")(module, file_name, inputs);
}

torch::jit::Module NeZha_ConvertModule(Module& module, torch::Tensor input) {

    Module new_final_module = module.clone(); // we will generate the final graph here and return it.
    Module new_module = module.clone();
    std::shared_ptr<Graph> graph = new_module.get_method("forward").graph()->copy();

    auto temporary_nodes = graph->block()->nodes();
    generic_graph_node_list_iterator<Node> keyNode;

    // Construct the first ONNX.
    // for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
    //         it++) {
        
    //     auto current_kind = it->kind().toQualString();
    //     // if (std::strcmp(current_kind, "onnx_ops::dummy_ops") == 0){
    //     //     // insert a new node before this to return a tuple, which contains inputs of dummy_ops.
    //     //     auto selectedNode = *it;
    //     //     WithInsertPoint guard(selectedNode);
    //     //     auto* new_output = selectedNode->owningGraph()->insertBefore(selectedNode->owningGraph()->create(
    //     //                 Symbol::fromQualString("onnx_ops::fake_ops"),
    //     //                 selectedNode->inputs(),
    //     //                 selectedNode->outputs().size()));

    //     //     for (size_t i = 0; i < it->outputs().size(); ++i) {
    //     //         it->output(i)->replaceAllUsesWith(new_output->output(i));
    //     //     }
    //     // }

        
    //     keyNode = it;
    //     keyNode++;

    //     it->output()->replaceAllUsesWith(keyNode->outputs()[0]);
    //     it.destroyCurrent();

    //     if (std::strcmp(current_kind, "onnx_ops::dummy_ops") == 0) {
    //         break;
    //     }

    //     printf("\n------ check graph: ------\n");
    //     graph->dump();        
    // }

    // for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
    //             it++) {
    //     if (std::strcmp(it->kind().toQualString(), "prim::GetAttr") == 0){
    //         if (!it->output()->hasUses()){
    //             it.destroyCurrent();
    //         }
    //     }
    // }

    printf("\n------ check final graph: ------\n");
    graph->dump();

    // auto method_name = QualifiedName(*new_module.type()->name(), "forward");
    // new_module.type()->unsafeRemoveMethod("forward");
    // new_module._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    // auto fn = new_module._ivalue()->compilation_unit()->create_function(method_name, graph);
    // new_module.type()->addMethod(fn);

    // printf("------ ready to export: ------\n");
    // // Export current new_module to ONNX file
    // export_to_onnx(new_module, "first_part.onnx", input);

    // Now, let's update the module code to the operator which handles the onnx file.
    // Remove all operators before the given dummy ops
    std::shared_ptr<Graph> new_final_graph = new_final_module.get_method("forward").graph()->copy();
    temporary_nodes = new_final_graph->block()->nodes();

    for (auto it = temporary_nodes.begin(); it != temporary_nodes.end();
            it++) {
        if (std::strcmp(it->kind().toQualString(), "prim::GetAttr") == 0){
            auto allNames = it->attributeNames();

            for (c10::Symbol sy : allNames){
                printf("\n------ Attr: %s ------\n", sy.toQualString());
            }
            continue;
        }

        if (std::strcmp(it->kind().toQualString(), "onnx_ops::dummy_ops") != 0) {
            it->output()->replaceAllUsesWith(new_final_graph->inputs()[1]);
            it.destroyCurrent();
        } else {
            // Insert the new operator for ort inferencing
            Node* selectedNode = *it;
            WithInsertPoint guard(selectedNode);
            auto* new_node = new_final_graph->create(
                    Symbol::fromQualString("onnx_ops::ort_inference_ops"),
                    selectedNode->owningGraph()->inputs(),
                    selectedNode->inputs().size());
            printf("\n------ Create Node: ------\n");

            new_node->insertBefore(selectedNode);
            printf("\n------ Insert Node: ------\n");
            selectedNode->replaceInput(0, new_node->outputs()[0]);

            // Construct a new string Constant for onnx file name and update it to input of ort_inference_ops
            auto* new_str = new_final_graph->create(
                    Symbol::prim("Constant"),
                    1);
            new_str->s_(Symbol::attr("value"), "first_part.onnx");
            new_str->outputs()[0]->setType(jit::StringType::get());
            new_str->insertBefore(new_node);
            new_node->replaceInput(0, new_str->outputs()[0]);
            
            // new_node->values_
            break;
        }

        printf("\n------ check fina graph during iteration: ------\n");
        new_final_graph->dump();
    }

    printf("\n------ check fina graph: ------\n");
    new_final_graph->dump();

    auto method_name = QualifiedName(*new_final_module.type()->name(), "forward");
    new_final_module.type()->unsafeRemoveMethod("forward");
    new_final_module._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = new_final_module._ivalue()->compilation_unit()->create_function(method_name, new_final_graph);
    new_final_module.type()->addMethod(fn);

    return new_final_module;
}

Module NeZha_UpdateOps(
    Module& module) {
    
    std::shared_ptr<Graph> graph = module.get_method("forward").graph()->copy();
    auto temporary_nodes = graph->block()->nodes();

    torch::Tensor inferred_input;
    auto all_inputs = graph->inputs();
    for (auto local_input : all_inputs) {
        printf("----- input name: %s,  input type: %s-----\n", local_input->debugName().c_str(), local_input->type()->str().c_str());

        if (local_input->type()->kind() == c10::TypeKind::TensorType) {
            printf("------ This is a tensor. ------\n");
        }

        // if (local_input->isTensor()) {
        //     inferred_input = local_input->toTensor();
        // }
        printf("------ finished one. ------\n");
    }

    Node* selectedNode;    
    for (auto it = temporary_nodes.begin(); it != temporary_nodes.end();
            it++) {

        if (std::strcmp(it->kind().toQualString(), "prim::GetAttr") == 0){
            continue;
        }

        if (std::strcmp(it->kind().toQualString(), "onnx_ops::dummy_ops") == 0){
            selectedNode = *it;
            break;
        }

        printf("------ check node: ------\n");
        it->dump();
    }

    WithInsertPoint guard(selectedNode);
    auto* new_node =
        selectedNode->owningGraph()->insertNode(selectedNode->owningGraph()->create(
            Symbol::fromQualString("onnx_ops::fake_ops"),
            selectedNode->inputs(),
            selectedNode->outputs().size()));
    for (size_t i = 0; i < selectedNode->outputs().size(); ++i) {
        selectedNode->output(i)->replaceAllUsesWith(new_node->output(i));
    }
    // new_node->s_(Symbol::fromQualString("attr::operator"), "expand");

    // selectedNode->replaceAllUsesWith(new_node);
    selectedNode->destroy();

    const auto method_name = QualifiedName(*module.type()->name(), "forward");
    module.type()->unsafeRemoveMethod("forward");
    module._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = module._ivalue()->compilation_unit()->create_function(method_name, graph);
    module.type()->addMethod(fn);
    printf("\n------ Print module after updated: \n%s ------", module.dump_to_str(true, false, false).c_str());

    return module;
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
    printf("\n------ Print second module after: \n%s ------", module.dump_to_str(true, false, false).c_str());
}

void NeZha_TrySplitModule(
    Module& module_1st,
    Module& module_2nd) {

    printf("\n------ Start NeZha_TrySplitModule ------");
    printf("\n------ Print first module before: \n%s ------", module_1st.dump_to_str(true, false, true).c_str());
    auto graph_bak = module_1st.get_method("forward").graph()->copy();
    updateGraph(graph_bak);
    const auto method_name = QualifiedName(*module_1st.type()->name(), "forward");
    module_1st.type()->unsafeRemoveMethod("forward");
    module_1st._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = module_1st._ivalue()->compilation_unit()->create_function(
        method_name, graph_bak);
    module_1st.type()->addMethod(fn);
    printf("\n------ Print first module after: \n%s ------", module_1st.dump_to_str(true, false, true).c_str());

    update2ndGraph(module_2nd);
}

std::vector<Module> NeZha_GetSplitModules(
    Module& module) {
    
    auto splitModules = std::vector<Module>{};

    auto module_1st = module.clone();
    auto graph_bak = module_1st.get_method("forward").graph()->copy();
    updateGraph(graph_bak);
    const auto method_name = QualifiedName(*module_1st.type()->name(), "forward");
    module_1st.type()->unsafeRemoveMethod("forward");
    module_1st._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = module_1st._ivalue()->compilation_unit()->create_function(
        method_name, graph_bak);
    module_1st.type()->addMethod(fn);
    splitModules.push_back(module_1st);

    auto module_2nd = module.clone();
    update2ndGraph(module_2nd);
    splitModules.push_back(module_2nd);

    return splitModules;
}

} // namespace jit
} // namespace torch
