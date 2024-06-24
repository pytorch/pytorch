#include <torch/script.h>

#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include <iostream>

void test_module_forward_invocation_no_hooks_run(
    const std::string &path_to_exported_script_module) {
  std::cout << "testing: "
            << "test_module_forward_invocation_no_hooks_run" << std::endl;
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" +
                       "test_module_forward_multiple_inputs" + ".pt");
  std::vector<torch::jit::IValue> inputs = {torch::List<std::string>({"a"}),
                                            torch::jit::IValue("no_pre_hook")};

  auto output = module(inputs);
  auto output_forward = module.forward(inputs);
  torch::jit::IValue correct_direct_output =
      std::tuple<torch::List<std::string>, std::string>(
          {"a", "outer_mod_name", "inner_mod_name"}, "no_pre_hook_");
  std::cout << "----- module output: " << output << std::endl;
  std::cout << "----- module forward output: " << output_forward << std::endl;
  AT_ASSERT(correct_direct_output == output_forward);
}

void test_submodule_called_directly_with_hooks(
    const std::string &path_to_exported_script_module) {
  std::cout << "testing: "
            << "test_submodule_to_call_directly_with_hooks" << std::endl;
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" +
                       "test_submodule_to_call_directly_with_hooks" + ".pt");
  torch::jit::Module submodule = *module.modules().begin();
  std::vector<torch::jit::IValue> inputs = {"a"};

  auto output = submodule(inputs);
  torch::jit::IValue correct_output = "pre_hook_override_name_inner_mod_fh";
  std::cout << "----- submodule's output: " << output << std::endl;
  std::cout << "----- expected output   : " << correct_output << std::endl;
  AT_ASSERT(correct_output == correct_output);
}

struct HooksTestCase {
  std::string name;
  std::vector<torch::jit::IValue> inputs;
  torch::jit::IValue output;
  HooksTestCase(std::string name, std::vector<torch::jit::IValue> inputs,
                torch::jit::IValue output)
      : name(name), inputs(std::move(inputs)), output(std::move(output)) {}
};

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: test_jit_hooks <path-to-exported-script-module>\n";
    return -1;
  }
  const std::string path_to_exported_script_module = argv[1];
  std::cout << "path to exported module:" << path_to_exported_script_module
            << std::endl;
  std::cout << "Tesing JIT Hooks in CPP" << std::endl;

  // Note: Modules loaded in this file are produced in /test/jit_hooks/model.py

  std::vector<HooksTestCase> test_cases = {
      HooksTestCase("test_submodule_multiple_hooks_single_input",
                    {torch::jit::IValue("a")},
                    "pre_hook_override_name2_inner_mod_fwh1"),
      HooksTestCase("test_submodule_hook_return_nothing",
                    {torch::jit::IValue("a")}, "a_outermod_inner_mod"),
      HooksTestCase("test_submodule_same_hook_repeated",
                    {torch::jit::IValue("a")},
                    "a_outermod_ph_ph_inner_mod_fh_fh"),
      HooksTestCase("test_submodule_forward_single_input",
                    {torch::jit::IValue("a")},
                    "pre_hook_override_name_inner_mod"),
      HooksTestCase(
          "test_submodule_multiple_hooks_multiple_inputs",
          {torch::List<std::string>({"a"}), torch::jit::IValue("no_pre_hook")},
          std::tuple<torch::List<std::string>, std::string>(
              {"pre_hook_override_name", "inner_mod_name"},
              "pre_hook_override2_fh1_fh2")),
      HooksTestCase(
          "test_submodule_forward_multiple_inputs",
          {torch::List<std::string>({"a"}), torch::jit::IValue("no_pre_hook")},
          std::tuple<torch::List<std::string>, std::string>(
              {"pre_hook_override_name", "inner_mod_name"},
              "pre_hook_override_fh")),
      HooksTestCase("test_module_forward_single_input",
                    {torch::jit::IValue("a")},
                    "pre_hook_override_name_outermod_inner_mod_fh"),
      HooksTestCase("test_module_multiple_hooks_single_input",
                    {torch::jit::IValue("a")},
                    "pre_hook_override_name2_outermod_inner_mod_fh1_fh2"),
      HooksTestCase("test_module_hook_return_nothing",
                    {torch::jit::IValue("a")}, "a_outermod_inner_mod"),
      HooksTestCase("test_module_same_hook_repeated", {torch::jit::IValue("a")},
                    "a_ph_ph_outermod_inner_mod_fh_fh"),
      HooksTestCase(
          "test_module_forward_multiple_inputs",
          {torch::List<std::string>({"a"}), torch::jit::IValue("no_pre_hook")},
          std::tuple<torch::List<std::string>, std::string>(
              {"pre_hook_override_name", "outer_mod_name", "inner_mod_name"},
              "pre_hook_override_fh")),
      HooksTestCase(
          "test_module_multiple_hooks_multiple_inputs",
          {torch::List<std::string>({"a"}), torch::jit::IValue("no_pre_hook")},
          std::tuple<torch::List<std::string>, std::string>(
              {"pre_hook_override_name2", "outer_mod_name", "inner_mod_name"},
              "pre_hook_override_fh1_fh2")),
      HooksTestCase("test_module_no_forward_input", {}, torch::jit::IValue()),
      HooksTestCase("test_forward_tuple_input", {std::tuple<int>(11)},
                    {std::tuple<int>(11)}),
  };

  for (HooksTestCase &test_case : test_cases) {
    std::cout << "testing: " << test_case.name << std::endl;
    torch::jit::Module module = torch::jit::load(
        path_to_exported_script_module + "_" + test_case.name + ".pt");
    torch::jit::IValue output = module(test_case.inputs);
    std::cout << "----- module's output: " << output << std::endl;
    std::cout << "----- expected output: " << test_case.output << std::endl;
    AT_ASSERT(output == test_case.output);
  }

  // special test cases that don't call the imported module directly
  test_module_forward_invocation_no_hooks_run(path_to_exported_script_module);
  test_submodule_called_directly_with_hooks(path_to_exported_script_module);

  std::cout << "JIT CPP Hooks okay!" << std::endl;

  return 0;
}
