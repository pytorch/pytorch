// inductor_module.cpp
#include <torch/extension.h>
#include <torch/script.h>
#include <vector>

extern std::vector<at::Tensor> inductor_entry_cpp(const std::vector<at::Tensor>& args);

struct InductorModule : torch::nn::Module {
  InductorModule() {}

  torch::Tensor forward(torch::Tensor x, torch::Tensor y) {
    std::vector<at::Tensor> inputs = {x, y};
    std::vector<at::Tensor> outputs = inductor_entry_cpp(inputs);
    return outputs[0];
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<InductorModule, std::shared_ptr<InductorModule>>(m, "InductorModule")
    .def(py::init<>())
    .def("forward", &InductorModule::forward);
}
