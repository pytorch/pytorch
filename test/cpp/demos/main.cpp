#include <iostream>
#include <string>
#include <torch/csrc/jit/mobile/import.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

int main(int argc, char* argv[]) {
    std::cout << "test start" << std::endl;
    std::string file_path = argv[1];
    std::vector<c10::IValue> inputs;

    torch::jit::mobile::Module bc = torch::jit::_load_for_mobile(file_path);
    // inputs.push_back(c10::IValue(1));
    inputs.push_back(torch::ones({1, 1, 4, 4}));

    at::Tensor output = bc.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    return 0;
}
