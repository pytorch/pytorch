#include <torch/script.h>


#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include <iostream>

void test_submodule_multiple_hooks_single_input(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_multiple_hooks_single_input" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("pre_hook_override_name2_inner_mod_fwh1" == output);
}

void test_submodule_hook_return_nothing(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_hook_return_nothing" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("a_outermod_inner_mod" == output);
}

void test_submodule_same_hook_repeated(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_same_hook_repeated" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("a_outermod_ph_ph_inner_mod_fh_fh" == output);
}

void test_module_no_forward_input(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_no_forward_input" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT(torch::jit::IValue() == output);
}

void test_module_same_hook_repeated(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_same_hook_repeated" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("a_ph_ph_outermod_inner_mod_fh_fh" == output);
}

void test_module_forward_single_input(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_forward_single_input" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("pre_hook_override_name_outermod_inner_mod_fh" == output);
}

void test_module_hook_return_nothing(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_hook_return_nothing" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("a_outermod_inner_mod" == output);
}


void test_submodule_forward_single_input(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_forward_single_input" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("pre_hook_override_name_inner_mod" == output);
}

void test_module_multiple_hooks_single_input(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_multiple_hooks_single_input" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("pre_hook_override_name2_outermod_inner_mod_fh1_fh2" == output);
}


void test_submodule_multiple_hooks_multiple_inputs(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_multiple_hooks_multiple_inputs" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"a"});
  inputs.push_back(list);
  inputs.push_back("no_pre_hook");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::ostringstream stream;
  stream << output;
  std::string output_str =  stream.str();
  AT_ASSERT("([pre_hook_override_name, inner_mod_name], pre_hook_override2_fh1_fh2)" == output_str);
}

void test_module_multiple_hooks_multiple_inputs(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_multiple_hooks_multiple_inputs" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"a"});
  inputs.push_back(list);
  inputs.push_back("no_pre_hook");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::ostringstream stream;
  stream << output;
  std::string output_str =  stream.str();
  AT_ASSERT("([pre_hook_override_name2, outer_mod_name, inner_mod_name], pre_hook_override_fh1_fh2)" == output_str);
}

void test_forward_tuple_input(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_forward_tuple_input" + ".pt");
  std::vector<torch::jit::IValue> inputs;

  std::tuple<int> input_tuple (11);
  inputs.push_back(input_tuple);

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::ostringstream stream;
  stream << output;
  std::string output_str =  stream.str();
  AT_ASSERT("(11,)" == output_str);;
}


void test_module_forward_multiple_inputs(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_forward_multiple_inputs" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"a"});
  inputs.push_back(list);
  inputs.push_back("no_pre_hook");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::ostringstream stream;
  stream << output;
  std::string output_str =  stream.str();
  AT_ASSERT("([pre_hook_override_name, outer_mod_name, inner_mod_name], pre_hook_override_fh)" == output_str);
}

void test_module_forward_invocation_no_hooks_run(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_forward_multiple_inputs" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"a"});
  inputs.push_back(list);
  inputs.push_back("no_pre_hook");

  auto output = module(inputs);
  auto output_forward = module.forward(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::cout << "----- module forward output: " << output_forward << std::endl;
  std::ostringstream stream;
  stream << output_forward;
  std::string output_forward_str =  stream.str();
  AT_ASSERT("([a, outer_mod_name, inner_mod_name], no_pre_hook_)" == output_forward_str);
}


void test_submodule_forward_multiple_inputs(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_forward_multiple_inputs" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"a"});
  inputs.push_back(list);
  inputs.push_back("no_pre_hook");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::ostringstream stream;
  stream << output;
  std::string output_str =  stream.str();
  AT_ASSERT("([pre_hook_override_name, inner_mod_name], pre_hook_override_fh)" == output_str);
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: test_jit_hooks <path-to-exported-script-module>\n";
    return -1;
  }
  const std::string path_to_exported_script_module = argv[1];

  // Note: Modules loaded in this file are produced in /test/jit_hooks/model.py

  std::cout << "Tesing JIT Hooks in CPP" << std::endl; 
  std::cout << "testing: test_submodule_multiple_hooks_single_input" << std::endl;
  test_submodule_multiple_hooks_single_input(path_to_exported_script_module);
  std::cout << "testing: test_submodule_hook_return_nothing" << std::endl;
  test_submodule_hook_return_nothing(path_to_exported_script_module);
  std::cout << "testing: test_submodule_same_hook_repeated" << std::endl;
  test_submodule_same_hook_repeated(path_to_exported_script_module);
  std::cout << "testing: test_submodule_forward_single_input" << std::endl;
  test_submodule_forward_single_input(path_to_exported_script_module);
  std::cout << "testing: test_submodule_multiple_hooks_multiple_inputs" << std::endl;
  test_submodule_multiple_hooks_multiple_inputs(path_to_exported_script_module);
  std::cout << "testing: test_submodule_forward_multiple_inputs" << std::endl;
  test_submodule_forward_multiple_inputs(path_to_exported_script_module);
  
  std::cout << "testing: test_module_forward_single_input" << std::endl;
  test_module_forward_single_input(path_to_exported_script_module);
  std::cout << "testing: test_module_multiple_hooks_single_input" << std::endl;
  test_module_multiple_hooks_single_input(path_to_exported_script_module);
  std::cout << "testing: test_module_hook_return_nothing" << std::endl;
  test_module_hook_return_nothing(path_to_exported_script_module);
  std::cout << "testing: test_module_same_hook_repeated" << std::endl;
  test_module_same_hook_repeated(path_to_exported_script_module);
  std::cout << "testing: test_module_forward_multiple_inputs" << std::endl;
  test_module_forward_multiple_inputs(path_to_exported_script_module);
  std::cout << "testing: test_module_multiple_hooks_multiple_inputs" << std::endl;
  test_module_multiple_hooks_multiple_inputs(path_to_exported_script_module);
  
  std::cout << "testing: test_module_no_forward_input" << std::endl;
  test_module_no_forward_input(path_to_exported_script_module);
  std::cout << "testing: test_forward_tuple_input" << std::endl;
  test_forward_tuple_input(path_to_exported_script_module);

  std::cout << "testing: test_module_forward_invocation_no_hooks_run" << std::endl;
  test_module_forward_invocation_no_hooks_run(path_to_exported_script_module);

  std::cout << "JIT CPP Hooks okay!" << std::endl;
}
