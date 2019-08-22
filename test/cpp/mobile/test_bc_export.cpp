#include <torch/script.h> // One-stop header.
//#include <aten/src/ATen/core/ivalue.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-of-script-module> <path-of-bytecode-file>";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  torch::jit::script::Module module = torch::jit::load(argv[1]);

//  torch::jit::mobile::save(module, argv[2]);
  auto compUnit = module.class_compilation_unit();
  auto funcList = compUnit->get_functions();
  for (auto func : funcList) {
    auto funcName = func->name();
    torch::jit::Code code(func->graph());
  }
  std::vector<torch::jit::IValue> inputs;

  //  inputs.push_back(torch::ones({1, 2}));

  //  inputs.push_back(torch::ones({1, 10}));

    inputs.push_back(torch::ones({1, 3, 224, 224}));
  //  module->run_method("forward", inputs);
  //  module->save_method("forward", inputs, "/Users/myuan/data/resnet18_eval.bc");

//  inputs.push_back(torch::ones({1, 1, 28, 28}));

  at::Tensor output = module.forward(inputs).toTensor();
//  module.save("/Users/myuan/data/resnet18/exported.pt");
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;

  //  // pytext
  //  auto options = torch::TensorOptions().dtype(torch::kI64);
  //  int length = 5;
  //  inputs.push_back(torch::ones({1, length}, options));
  //  auto stensor = length * torch::ones({1}, options);
  //  inputs.push_back(stensor);
  //  auto output = module->forward(inputs);
  //  std::cout << output;
  //  module->save_method("forward", inputs, "/Users/myuan/data/pytext/model_traced.bc");
}
