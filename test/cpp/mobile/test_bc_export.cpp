#include <torch/script.h> // One-stop header.
#include <torch/csrc/jit/export_bytecode.h>
#include <torch/csrc/jit/export.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-of-script-module> <path-of-bytecode-file>";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  torch::jit::script::Module module = torch::jit::load(argv[1]);
//  //Save in new format with code and pkl files.
//  std::string nfile(argv[1]);
//  nfile += "1.pt";
//  module.save(nfile);
//  module.save("/Users/myuan/data/lenet/lenet_traced1.pt");
  torch::jit::mobile::SaveBytecode(module, argv[2]);

//  // Resnet
//  std::vector<torch::jit::IValue> inputs;
//  inputs.push_back(torch::ones({1, 3, 224, 224}));
//  at::Tensor output = module.forward(inputs).toTensor();
//  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;

//  // pytext
//  auto options = torch::TensorOptions().dtype(torch::kI64);
//  int length = 5;
//  inputs.push_back(torch::ones({1, length}, options));
//  auto stensor = length * torch::ones({1}, options);
//  inputs.push_back(stensor);
//  auto output = module->forward(inputs);
//  std::cout << output;
}
