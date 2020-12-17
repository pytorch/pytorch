#include <torch/script.h>


#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include <iostream>

void test_submodule_forward_and_pre_hooks_single_IO_multiple_hooks(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_forward_and_pre_hooks_single_IO_multiple_hooks" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("pre_hook_overrid_name2_inner_mod_fwh1" == output);
}

void test_submodule_forward_and_pre_hooks_single_IO_no_change(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_forward_and_pre_hooks_single_IO_no_change" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("a_outermod_inner_mod" == output);
}

void test_submodule_forward_and_pre_hook_single_IO_same_hook_twice(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_forward_and_pre_hook_single_IO_same_hook_twice" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("a_outermod_ph_ph_inner_mod_fh_fh" == output);
}

void test_module_hook_and_pre_hook_no_IO(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_hook_and_pre_hook_no_IO" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT(torch::jit::IValue() == output);
}

void test_module_forward_and_pre_hook_single_IO_same_hook_twice(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_forward_and_pre_hook_single_IO_same_hook_twice" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("a_ph_ph_outermod_inner_mod_fh_fh" == output);
}

void test_module_forward_and_pre_hook_single_IO(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_forward_and_pre_hook_single_IO" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("pre_hook_overrid_name_outermod_inner_mod_fh" == output);
}

void test_module_forward_and_pre_hook_single_IO_no_change(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_forward_and_pre_hook_single_IO_no_change" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("a_outermod_inner_mod" == output);
}


void test_submodule_forward_and_pre_hooks_single_IO(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_forward_and_pre_hooks_single_IO" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("pre_hook_overrid_name_inner_mod" == output);
}

void test_module_forward_and_pre_hook_single_IO_multiple_hooks(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_forward_and_pre_hook_single_IO_multiple_hooks" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("a");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  AT_ASSERT("pre_hook_overrid_name2_outermod_inner_mod_fh1_fh2" == output);
}


void test_submodule_hook_and_pre_hook_multiple_IO_multiple_hooks(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_hook_and_pre_hook_multiple_IO_multiple_hooks" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"a"});
  inputs.push_back(list);
  inputs.push_back("no_pre_hook");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::ostringstream stream;
  stream << output;
  std::string output_str =  stream.str();
  AT_ASSERT("([pre_hook_overrid_name, inner_mod_name], pre_hook_override2_fh1_fh2)" == output_str);
}

void test_module_hook_and_pre_hook_multiple_IO_multiple_hooks(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_hook_and_pre_hook_multiple_IO_multiple_hooks" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"a"});
  inputs.push_back(list);
  inputs.push_back("no_pre_hook");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::ostringstream stream;
  stream << output;
  std::string output_str =  stream.str();
  AT_ASSERT("([pre_hook_overrid_name2, outer_mod_name, inner_mod_name], pre_hook_override_fh1_fh2)" == output_str);
}

void test_module_hook_and_pre_hook_multiple_IO(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_module_hook_and_pre_hook_multiple_IO" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"a"});
  inputs.push_back(list);
  inputs.push_back("no_pre_hook");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::ostringstream stream;
  stream << output;
  std::string output_str =  stream.str();
  AT_ASSERT("([pre_hook_overrid_name, outer_mod_name, inner_mod_name], pre_hook_override_fh)" == output_str);
}


void test_submodule_hook_and_pre_hook_multiple_IO(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" + "test_submodule_hook_and_pre_hook_multiple_IO" + ".pt");
  std::vector<torch::jit::IValue> inputs;
  torch::List<std::string> list({"a"});
  inputs.push_back(list);
  inputs.push_back("no_pre_hook");

  auto output = module(inputs);
  std::cout << "----- module output: " << output << std::endl;
  std::ostringstream stream;
  stream << output;
  std::string output_str =  stream.str();
  AT_ASSERT("([pre_hook_overrid_name, inner_mod_name], pre_hook_override_fh)" == output_str);
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: test_jit_hooks <path-to-exported-script-module>\n";
    return -1;
  }
  const std::string path_to_exported_script_module = argv[1];

  // Note: Modules loaded in this file are produced in /test/jit_hooks/model.py

  std::cout << "Tesing JIT Hooks in CPP" << std::endl; 
  std::cout << "testing: test_submodule_forward_and_pre_hooks_single_IO_multiple_hooks" << std::endl;
  test_submodule_forward_and_pre_hooks_single_IO_multiple_hooks(path_to_exported_script_module);
  std::cout << "testing: test_submodule_forward_and_pre_hooks_single_IO_no_change" << std::endl;
  test_submodule_forward_and_pre_hooks_single_IO_no_change(path_to_exported_script_module);
  std::cout << "testing: test_submodule_forward_and_pre_hook_single_IO_same_hook_twice" << std::endl;
  test_submodule_forward_and_pre_hook_single_IO_same_hook_twice(path_to_exported_script_module);
  std::cout << "testing: test_submodule_forward_and_pre_hooks_single_IO" << std::endl;
  test_submodule_forward_and_pre_hooks_single_IO(path_to_exported_script_module);
  std::cout << "testing: test_submodule_hook_and_pre_hook_multiple_IO_multiple_hooks" << std::endl;
  test_submodule_hook_and_pre_hook_multiple_IO_multiple_hooks(path_to_exported_script_module);
  std::cout << "testing: test_submodule_hook_and_pre_hook_multiple_IO" << std::endl;
  test_submodule_hook_and_pre_hook_multiple_IO(path_to_exported_script_module);
  
  std::cout << "testing: test_module_forward_and_pre_hook_single_IO" << std::endl;
  test_module_forward_and_pre_hook_single_IO(path_to_exported_script_module);
  std::cout << "testing: test_module_forward_and_pre_hook_single_IO_multiple_hooks" << std::endl;
  test_module_forward_and_pre_hook_single_IO_multiple_hooks(path_to_exported_script_module);
  std::cout << "testing: test_module_forward_and_pre_hook_single_IO_no_change" << std::endl;
  test_module_forward_and_pre_hook_single_IO_no_change(path_to_exported_script_module);
  std::cout << "testing: test_module_forward_and_pre_hook_single_IO_same_hook_twice" << std::endl;
  test_module_forward_and_pre_hook_single_IO_same_hook_twice(path_to_exported_script_module);
  std::cout << "testing: test_module_hook_and_pre_hook_multiple_IO" << std::endl;
  test_module_hook_and_pre_hook_multiple_IO(path_to_exported_script_module);
  std::cout << "testing: test_module_hook_and_pre_hook_multiple_IO_multiple_hooks" << std::endl;
  test_module_hook_and_pre_hook_multiple_IO_multiple_hooks(path_to_exported_script_module);
  
  std::cout << "testing: test_module_hook_and_pre_hook_no_IO" << std::endl;
  test_module_hook_and_pre_hook_no_IO(path_to_exported_script_module);

  std::cout << "JIT CPP Hooks okay!" << std::endl;
}
