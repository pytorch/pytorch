
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>


torch::Tensor dummyOps(torch::Tensor testData, int64_t value) {
    printf(" ===== from dummyOps =====\n");
    auto tempValue = torch::ones(value);
    auto output = torch::add(testData, tempValue);
    return output.clone();
}

torch::Tensor ort_inference_ops(torch::Tensor testData, int64_t value) {
    printf(" ===== from ort_inference_ops =====\n");
    auto tempValue = torch::ones(value);
    auto output = torch::add(testData, tempValue);
    return output.clone();
}

TORCH_LIBRARY(onnx_ops, m) {
  m.def("dummy_ops", dummyOps);
  m.def("ort_inference_ops", ort_inference_ops);
}