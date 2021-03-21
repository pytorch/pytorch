
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <torch/csrc/autograd/function_hook.h>

torch::Tensor dummyOps(torch::Tensor testData, int64_t value) {
    printf(" ===== from dummyOps =====\n");
    auto tempValue = torch::ones(value);
    auto output = torch::add(testData, tempValue);
    return output.clone();
}

torch::Tensor fake_ops(torch::Tensor testData, int64_t value) {
    printf(" ===== from fake_ops =====\n");
    auto tempValue = torch::ones(value);
    auto output = torch::add(testData, tempValue);
    return output.clone();
}

torch::Tensor ort_inference_ops(std::string file_name, torch::Tensor inputs) {
    py::object onnx = py::module::import("torch.onnx");
    py::object my_output = onnx.attr("try_ort_inference")(file_name, inputs);
    torch::jit::InferredType my_type = torch::jit::tryToInferType(my_output);
    // auto final_output = torch::jit::toIValue(my_output, my_type.type()); // toIValue() failed to be exported by pybind11.
    auto final_output = py::cast<torch::autograd::Variable>(my_output);

    return final_output;
}

TORCH_LIBRARY(onnx_ops, m) {
  m.def("dummy_ops", dummyOps);
  m.def("fake_ops", fake_ops);
  m.def("ort_inference_ops", ort_inference_ops);
}