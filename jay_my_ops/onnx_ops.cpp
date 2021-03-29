
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/jit/passes/onnx/nezha_helper.h>

#include <pybind11/embed.h>

#include <unistd.h>
#include <sys/wait.h>

torch::Tensor dummyOps(torch::Tensor testData) {
    printf(" ===== from dummyOps =====\n");
    // auto tempValue = torch::ones_like(testData);
    // auto output = torch::add(testData, tempValue);
    return testData.clone();
}

torch::Tensor fake_ops(torch::Tensor testData, int64_t value) {
    printf(" ===== from fake_ops =====\n");
    auto tempValue = torch::ones_like(testData);
    auto output = torch::add(testData, tempValue);
    return output.clone();
}

torch::Tensor ort_inference_ops(std::string file_name, torch::Tensor inputs) {
    torch::Tensor final_output;

    PyEval_InitThreads();
    printf(" ===== from ort_inference_ops:init =====\n");
    pybind11::gil_scoped_acquire acquire;

    py::object py_onnx = py::module::import("torch.onnx");
    printf(" ===== ort_inference_ops: import successfully =====\n");
    py::object my_output = py_onnx.attr("try_ort_inference")(file_name, inputs);
    printf(" ===== ort_inference_ops: finish running method. =====\n");        
    final_output = py::cast<torch::autograd::Variable>(my_output);    
    pybind11::gil_scoped_release release;
    return final_output;
}

TORCH_LIBRARY(onnx_ops, m) {
  m.def("dummy_ops", dummyOps);
  m.def("fake_ops", fake_ops);
  m.def("ort_inference_ops", ort_inference_ops);
}