#pragma once
#include <torch/csrc/jit/api/module.h>
#include <iostream>
#include <vector>
namespace torchpy {
void init();
std::string hello();
torch::jit::Module load(const std::string& filename);
std::vector<torch::jit::IValue> inputs(std::vector<int64_t> shape);
}
