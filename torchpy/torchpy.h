#pragma once
#include <vector>

namespace torchpy {
void init();
std::vector<torch::jit::IValue> inputs();
}
