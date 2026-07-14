#include <torch/csrc/jit/api/module.h>
#include <torch/jit.h>
#include <torch/script.h>

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  std::string input_file_path{argv[1]};
  std::string output_file_path{argv[2]};

  std::ifstream ifs(input_file_path);
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  torch::jit::Module m("TestModule");

  m.define(buffer.str());
  m.save(output_file_path);
}
