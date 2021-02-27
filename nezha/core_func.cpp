//#include<core_func.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/extension.h>

std::vector<torch::jit::Module> split_modules(const torch::jit::Module& origModule) {
    auto splitModules = std::vector<torch::jit::Module>{};
    splitModules.push_back(origModule);

    return splitModules;
}


//pybind11 binding
PYBIND11_MODULE (TORCH_EXTENSION_NAME, m) {
    m.def ("split_modules",&split_modules, "split a module into several ones.");
}