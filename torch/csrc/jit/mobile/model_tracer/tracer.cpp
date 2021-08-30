#include <iostream>
#include <string>

/**
 * The tracer.cpp generates a binary that accepts a TorchScript model or a
 * Torch Mobile Model (with bytecode.pkl) which has at least 1 bundled
 * input. This binary then feeds the bundled input(s) into the model
 * and executes using the lite interpreter.
 *
 * Both root operators as well as called operators are recorded and saved
 * into a YAML file (whose path is provided on the command line).
 *
 * Note: Root operators may include primary and other operators that
 * are not invoked using the dispatcher, and hence they may not show
 * up in the Traced Operator list.
 *
 */

#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/model_tracer/TensorUtils.h>
#include <torch/csrc/jit/mobile/model_tracer/TracerRunner.h>


C10_DEFINE_string(
    model_input_path,
    "",
    "The path of the input model file (.ptl).");

C10_DEFINE_string(
    build_yaml_path,
    "",
    "The path of the output YAML file containing traced operator information.");

#define REQUIRE_STRING_ARG(name)                            \
  if (FLAGS_##name.empty()) {                               \
    std::cerr << "You must specify the flag --" #name "\n"; \
    return 1;                                               \
  }

#define REQUIRE_INT_ARG(name)                               \
  if (FLAGS_##name == -1) {                                 \
    std::cerr << "You must specify the flag --" #name "\n"; \
    return 1;                                               \
  }

void printYAML(std::ostream& out, const std::set<std::string>& operator_list) {
  std::cout << "test" << std::endl;
  for (auto& it : operator_list) {
    out << "- " << it << std::endl;
  }
}

/**
 * Converts a pytorch model (full/lite) to lite interpreter model for
 * mobile, and additionally writes out a list of root and called
 * operators.
 */
int main(int argc, char* argv[]) {
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  REQUIRE_STRING_ARG(model_input_path);
  REQUIRE_STRING_ARG(build_yaml_path);

  const std::string input_module_path = FLAGS_model_input_path;

  std::ofstream yaml_out(FLAGS_build_yaml_path);

  std::cout << "Processing: " << input_module_path << std::endl;
  std::cout << "Output: " << FLAGS_build_yaml_path << std::endl;

  torch::jit::mobile::TracerResult tracer_result =
      torch::jit::mobile::trace_run(FLAGS_model_input_path);

  for (auto& it : tracer_result.called_kernel_tags) {
    std::cout << "kernal tag, key: " << it.first << " value: " << it.second
              << std::endl;
  }
  for (auto& it : tracer_result.traced_operators) {
    std::cout << "- " << it << std::endl;
  }
  printYAML(yaml_out, tracer_result.traced_operators);
}
