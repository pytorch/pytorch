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
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/script.h>

typedef std::map<std::string, std::set<std::string>> kt_type;
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

void printOpYAML(
    std::ostream& out,
    int indent,
    const std::string& op_name,
    bool is_used_for_training,
    bool is_root_operator,
    bool include_all_overloads) {
  out << std::string(indent, ' ') << op_name << ":" << std::endl;
  out << std::string(indent + 2, ' ')
      << "is_used_for_training: " << (is_used_for_training ? "true" : "false")
      << std::endl;
  out << std::string(indent + 2, ' ')
      << "is_root_operator: " << (is_root_operator ? "true" : "false")
      << std::endl;
  out << std::string(indent + 2, ' ')
      << "include_all_overloads: " << (include_all_overloads ? "true" : "false")
      << std::endl;
}

void printOpsYAML(
    std::ostream& out,
    const std::set<std::string>& operator_list,
    bool is_used_for_training,
    bool is_root_operator,
    bool include_all_overloads) {
  for (auto& it : operator_list) {
    printOpYAML(out, 2, it, false, is_root_operator, false);
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

  yaml_out << "include_all_kernel_dtypes: true" << std::endl;
  yaml_out << "operators:" << std::endl;
  printOpsYAML(
      yaml_out,
      root_ops,
      false /* is_used_for_training */,
      true /* is_root_operator */,
      false /* include_all_overloads */);
  printOpsYAML(
      yaml_out,
      traced_operators,
      false /* is_used_for_training */,
      false /* is_root_operator */,
      false /* include_all_overloads */);
}
