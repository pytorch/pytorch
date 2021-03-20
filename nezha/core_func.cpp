//#include<core_func.h>
#include <torch/script.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/extension.h>

#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>

static void updateGraph(std::shared_ptr<torch::jit::Graph> graph){
    if (graph == nullptr){
        return;
    }
    
    auto temporary_nodes = graph->block()->nodes();
    torch::jit::generic_graph_node_list_iterator<torch::jit::Node> keyNode;

    int delNodesCount = 0;
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
            it++) {
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

static void update2ndGraph(torch::jit::Module& module){
    std::shared_ptr<torch::jit::Graph> graph = module.get_method("forward").graph()->copy();
    auto temporary_nodes = graph->block()->nodes();

    int delNodesCount = 0;

    for (auto it = temporary_nodes.begin(); it != temporary_nodes.end();
            it++) {

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

    const auto method_name = c10::QualifiedName(*module.type()->name(), "forward");
    module.type()->unsafeRemoveMethod("forward");
    module._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = module._ivalue()->compilation_unit()->create_function(method_name, graph);
    module.type()->addMethod(fn);
    printf("\n------ Print second module after: \n%s ------", module.dump_to_str(true, false, false).c_str());
}

std::vector<torch::jit::Module> split_modules(
    const torch::jit::Module& origModule) {
    
    auto splitModules = std::vector<torch::jit::Module>{};

    auto module_1st = origModule.clone();
    auto graph_bak = module_1st.get_method("forward").graph()->copy();
    updateGraph(graph_bak);
    const auto method_name = c10::QualifiedName(*module_1st.type()->name(), "forward");
    module_1st.type()->unsafeRemoveMethod("forward");
    module_1st._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
    auto fn = module_1st._ivalue()->compilation_unit()->create_function(
        method_name, graph_bak);
    module_1st.type()->addMethod(fn);
    splitModules.push_back(module_1st);

    auto module_2nd = origModule.clone();
    update2ndGraph(module_2nd);
    splitModules.push_back(module_2nd);

    printf("\n------ in core_func.cpp ------\n");
    return splitModules;
}

py::object ort_inference(std::string file_name, torch::Tensor inputs, torch::Tensor outputs) {
// std::string ort_inference(std::string file_name, torch::Tensor input) {
    torch::jit::Module test_module = torch::jit::load(file_name);

    py::object onnx = py::module::import("torch.onnx");

    onnx.attr("export_c_module")(test_module, inputs, outputs, "/home/jay/work/test_nezha.onnx");

    torch::Tensor new_input = torch::rand_like(inputs);
    py::object my_output = onnx.attr("try_ort_inference")("/home/jay/work/test_nezha.onnx", new_input);
    
    return my_output;
  }

//pybind11 binding
PYBIND11_MODULE (TORCH_EXTENSION_NAME, m) {
    m.def ("split_modules", &split_modules, "split a module into several ones.");
    m.def ("ort_inference", &ort_inference, "inference a module by ORT.");
}


// Test code in /home/jay/repos/fatcat-z/pytorch/torch/onnx/__init__.py file.

// class DummyModule(torch.nn.Module):
//     def forward(self, x):
//         return x

// def export_c_module(m, inputs, outputs, file_name):
//     local_module = torch.jit.trace(DummyModule(), torch.ones(1))
//     local_module._c = m
//     torch.onnx.export(local_module, inputs, file_name, example_outputs=outputs)

// def try_ort_inference(file_name, inputs):
//     ort_sess = ort.InferenceSession(file_name)
//     input_name = ort_sess.get_inputs()[0].name
//     label_name = ort_sess.get_outputs()[0].name

//     my_input, _ = torch.jit._flatten(inputs)
//     my_inputs = [to_numpy(inp) for inp in my_input]

//     ort_outs = ort_sess.run([label_name], {input_name: my_inputs[0]})
//     return torch.from_numpy(ort_outs[0])