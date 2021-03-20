// This header is all you need to do the C++ portions of this
// tutorial
#include <torch/script.h>

// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <torch/custom_class.h>

#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>

#include <string>
#include <vector>

struct ONNXRuntimeClass : torch::CustomClassHolder {
  ONNXRuntimeClass() {}

  std::string inference(std::string file_name, torch::Tensor inputs, torch::Tensor outputs) {
    torch::jit::Module test_module = torch::jit::load(file_name);
    printf("\n ======== Finished load.");
    try{
      py::object onnx = py::module::import("torch.onnx");
    // py::object onnx_symbolic = py::module::import("torch.onnx.symbolic_helper");
    // py::object onnx_registry = py::module::import("torch.onnx.symbolic_registry");   
      printf("\n ======== Finished import.");
      onnx.attr("export_c_module")(test_module, inputs, outputs, "/home/jay/work/test_nezha.onnx");

    }catch (std::exception& e) {
      printf("\n ======== Caught exception.");
      printf("\n ======== Ex: %s", e.what());
    }

    printf("\n ======== Finished inference.");

    // torch.onnx._export(model, input_copy, f,
    //                    opset_version=opset_version,
    //                    example_outputs=example_outputs,
    //                    do_constant_folding=do_constant_folding,
    //                    keep_initializers_as_inputs=keep_initializers_as_inputs,
    //                    dynamic_axes=dynamic_axes,
    //                    input_names=input_names, output_names=output_names,
    //                    fixed_batch_size=fixed_batch_size, training=training,
    //                    onnx_shape_inference=onnx_shape_inference)


          // "_jit_pass_complete_shape_analysis",
          // [](const std::shared_ptr<Graph>& graph,
          //    const py::tuple& inputs,
          //    bool with_grad) {
          //   ArgumentSpecCreator arg_spec_creator(*graph);
          //   Stack stack;
          //   stack.reserve(inputs.size()); // captures?
          //   for (auto& obj : inputs) {
          //     stack.push_back(toTypeInferredIValue(obj));
          //   }
          //   ArgumentSpec spec = arg_spec_creator.create(with_grad, stack);
          //   arg_spec_creator.specializeTypes(*graph, spec);
          //   // We only get partial specialization from the arg_spec_creator, but
          //   // we want full shape specialization. The alternative would be to
          //   // have a "complete type inference" function in ArguemntSpecCreator.
          //   auto g_inputs = graph->inputs();
          //   for (size_t i = 0; i < inputs.size(); ++i) {
          //     if (stack[i].isTensor()) {
          //       g_inputs[i]->setType(stack[i].type());
          //     }
          //   }

    return test_module.dump_to_str(false, false, false);
  }

};


TORCH_LIBRARY(nezha_classes, m) {
m.class_<ONNXRuntimeClass>("ONNXRuntimeClass")
    .def(torch::init())
    .def("inference", &ONNXRuntimeClass::inference);
}