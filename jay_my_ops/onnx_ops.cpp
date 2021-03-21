
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/python/pybind_utils.h>

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
    // torch::Tensor new_input = torch::rand_like(inputs);
    py::object my_output = onnx.attr("try_ort_inference")(file_name, inputs);
    auto final_output = torch::jit::toIValue(my_output, c10::TensorType::get());

    return final_output.toTensor();
}

TORCH_LIBRARY(onnx_ops, m) {
  m.def("dummy_ops", dummyOps);
  m.def("fake_ops", fake_ops);
  m.def("ort_inference_ops", ort_inference_ops);
}