#include <stdexcept>

#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif

#include "aoti_custom_class.h"

namespace torch::aot_inductor {

static auto registerMyAOTIClass =
    torch::class_<MyAOTIClass>("aoti", "MyAOTIClass")
        .def(torch::init<std::string, std::string>())
        .def("forward", &MyAOTIClass::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<MyAOTIClass>& self)
                -> std::vector<std::string> {
              std::vector<std::string> v;
              v.push_back(self->lib_path());
              v.push_back(self->device());
              return v;
            },
            [](std::vector<std::string> params) {
              return c10::make_intrusive<MyAOTIClass>(params[0], params[1]);
            });

MyAOTIClass::MyAOTIClass(
    const std::string& model_path,
    const std::string& device)
    : lib_path_(model_path), device_(device) {
  if (device_ == "cuda") {
    runner_ = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_path.c_str());
  } else if (device_ == "cpu") {
    runner_ = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        model_path.c_str());
  } else {
    throw std::runtime_error("invalid device: " + device);
  }
}

std::vector<torch::Tensor> MyAOTIClass::forward(
    std::vector<torch::Tensor> inputs) {
  return runner_->run(inputs);
}

} // namespace torch::aot_inductor
