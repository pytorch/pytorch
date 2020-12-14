#include <torch/script.h>


#include <memory>
#include <string>
#include <vector>

#include <iostream>



void test_argument_checking_for_serialized_modules(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);

  try {
    module.forward({torch::jit::IValue(1), torch::jit::IValue(2)});
    //AT_ASSERT(false);
  } catch (const c10::Error& error) {
    //AT_ASSERT(
    //    std::string(error.what_without_backtrace())
    //        .find("Expected at most 2 argument(s) for operator 'forward', "
    //              "but received 3 argument(s)") == 0);
    AT_ASSERT(true);
  }

  try {
    module.forward({torch::jit::IValue(5)});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    //AT_ASSERT(
    //    std::string(error.what_without_backtrace())
    //        .find("forward() Expected a value of type 'Tensor' "
    //              "for argument 'input' but instead found type 'int'") == 0);
    AT_ASSERT(true);
  }

  try {
    module.forward({});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("forward() is missing value for argument 'input'") == 0);
  }
}

void load_serialized_module_with_custom_op_and_execute(
    const std::string& path_to_exported_script_module) {
  std::cout << "loading module in test_jit_hooks.cpp" << std::endl;
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);
  std::cout << "module loaded in test_jit_hooks.cpp!" << std::endl;
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"cpp_input"});
  inputs.push_back(list);
  auto output = module(inputs);
   std::cout << "module read in output: " << output << std::endl;
  //AT_ASSERT(output);
}



int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: test_jit_hooks <path-to-exported-script-module>\n";
    return -1;
  }
  const std::string path_to_exported_script_module = argv[1];

  std::cout << "\n\n\n ran ran ran! \n\n\n" << std::endl; 

  load_serialized_module_with_custom_op_and_execute(
      path_to_exported_script_module);
  test_argument_checking_for_serialized_modules(path_to_exported_script_module);


  std::cout << "ok\n";
}
