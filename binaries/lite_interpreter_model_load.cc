#include "ATen/ATen.h"
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include "torch/script.h"

C10_DEFINE_string(model, "", "The given bytecode model to check if it is supported by lite_interpreter.");

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "Check if exported bytecode model is runnable by lite_interpreter.\n"
    "Example usage:\n"
    "./lite_interpreter_model_load"
    " --model=<model_file>");

  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  if (FLAGS_model.empty()) {
    std::cerr << FLAGS_model <<  ":Model file is not provided\n";
    return -1;
  }
  
  // TODO: avoid having to set this guard for custom mobile build with mobile
  // interpreter.
  torch::AutoNonVariableTypeMode non_var_guard{true};
  torch::jit::mobile::Module bc = torch::jit::_load_for_mobile(FLAGS_model);
  return 0;
}
