
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/jit/passes/onnx/nezha_helper.h>


torch::Tensor dummyOps(torch::Tensor testData) {
    printf(" ===== from dummyOps =====\n");
    auto tempValue = torch::ones_like(testData);
    auto output = torch::add(testData, tempValue);
    return output.clone();
}

torch::Tensor fake_ops(torch::Tensor testData, int64_t value) {
    printf(" ===== from fake_ops =====\n");
    auto tempValue = torch::ones_like(testData);
    auto output = torch::add(testData, tempValue);
    return output.clone();
}

torch::Tensor ort_inference_ops(std::string file_name, torch::Tensor inputs) {
// torch::Tensor ort_inference_ops(torch::Tensor test, torch::Tensor inputs) {    
    PyObject *jit_module = PyImport_ImportModule("torch.onnx");
    printf(" ===== from ort_inference_ops:start =====\n");
    Py_InitializeEx(0);

    printf(" ===== from ort_inference_ops:init 0 =====\n");
    try{
        py::object py_onnx = py::module::import("io");
        printf(" ===== ort_inference_ops: import successfully =====\n");        
        py::object my_output = py_onnx.attr("try_ort_inference")(file_name, inputs);
        // torch::jit::InferredType my_type = torch::jit::tryToInferType(my_output);
        // auto final_output = torch::jit::toIValue(my_output, my_type.type()); // toIValue() failed to be exported by pybind11.
        auto final_output = py::cast<torch::autograd::Variable>(my_output);    
    } catch (py::error_already_set& e) {
        printf(" ===== Got an exception \n");
        printf(" ===== Exception: %s \n", e.what());
    }

    printf(" ===== ort_inference_ops: import successfully =====\n");
    
    return inputs;
}

TORCH_LIBRARY(onnx_ops, m) {
  m.def("dummy_ops", dummyOps);
  m.def("fake_ops", fake_ops);
  m.def("ort_inference_ops", ort_inference_ops);
}