#include <iostream>
#include <string>
#include "torch/csrc/jit/mobile/import.h"

int main(int argc, char* argv[]) {
    std::cout << "test start" << std::endl;
    std::string file_path = argv[1];
    torch::jit::mobile::Module bc = torch::jit::_load_for_mobile(file_path);
    return 0;
}
