#include <torch/csrc/jit/ir/ir.h>
//#include <torch/custom_class.h>

std::vector<torch::jit::Module> split_modules(const torch::jit::Module& origModule);