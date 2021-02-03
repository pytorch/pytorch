#include <iostream>
#include <string>
#include <torch/csrc/jit/mobile/import.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

int main(int argc, char* argv[]) {
    std::cout << "lite_interpreter_demo is starting..." << std::endl;

    if (argc == 2) {
        std::string file_path = argv[1];
        std::vector<c10::IValue> inputs;

        std::cout << "Loading model from: " << file_path << std::endl;
        torch::jit::mobile::Module bc = torch::jit::_load_for_mobile(file_path);

        inputs.push_back(torch::ones({1, 1, 4, 4}));

        at::Tensor output = bc.forward(inputs).toTensor();
        std::cout << "Model output is: " << std::endl;
        std::cout << output << std::endl;;
    } else {
        throw std::invalid_argument( "Incorrect input. "
            "Example usage: lite_interpreter_demo model.ptl" );
    }
    return 0;
}
