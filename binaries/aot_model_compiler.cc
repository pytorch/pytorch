#include <sstream>
#include <string>

#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
#include <torch/csrc/jit/mobile/nnc/aot_compiler.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>


C10_DEFINE_string(model, "", "The torch script model to optimize.");
C10_DEFINE_string(model_name, "", "The name of the model.");
C10_DEFINE_string(model_version, "", "The version of the model.");
C10_DEFINE_string(
    input_dims,
    "",
    "For input float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple inputs needed, use semicolon "
    "to separate the dimension of different tensors.");
C10_DEFINE_string(
    output_llvm,
    "",
    "Name of the output llvm assembly to be saved.");
C10_DEFINE_string(output_model, "", "Name of the output model to be saved.");

namespace {

std::vector<std::string> split(
    char separator,
    const std::string& string,
    bool ignore_empty = true) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (getline(ss, item, separator)) {
    if (!ignore_empty || !item.empty()) {
      pieces.push_back(std::move(item));
    }
  }
  return pieces;
}

std::vector<std::vector<int64_t>> parseInputShapes() {
  CAFFE_ENFORCE_GE(FLAGS_input_dims.size(), 0, "Input dims must be specified.");
  std::vector<std::string> input_dims_list = split(';', FLAGS_input_dims);
  std::vector<std::vector<int64_t>> inputs;
  for (const auto& input_dims_item : input_dims_list) {
    auto input_dims_str = split(',', input_dims_item);
    std::vector<int64_t> input_dims;
    input_dims.reserve(input_dims_str.size());
    for (const auto& s : input_dims_str) {
      input_dims.push_back(c10::stoi(s));
    }
    inputs.push_back(input_dims);
  }
  return inputs;
}

c10::Dict<c10::IValue, c10::IValue> createCompileSpec() {
  c10::Dict<c10::IValue, c10::IValue> compile_spec(
      c10::StringType::get(), c10::AnyType::get());
  c10::Dict<c10::IValue, c10::IValue> method_spec(
      c10::StringType::get(), c10::AnyType::get());
  auto input_shapes = parseInputShapes();
  TORCH_CHECK(
      input_shapes.size() == 1,
      "Wrong # of input shapes: ",
      input_shapes.size());
  method_spec.insert("sizes", input_shapes[0]); // TODO: support multiple inputs
  compile_spec.insert("forward", method_spec);
  return compile_spec;
}

std::vector<int64_t> getInputSizesForMethod(
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const std::string& method_name) {
  return method_compile_spec.at(method_name)
      .toGenericDict()
      .at("sizes")
      .toIntVector();
}

std::string getNncKernelId(const std::string& method_name) {
  // TODO: calculate the version_token.
  const std::string version_token = "VERTOKEN";
  return FLAGS_model_name + ":" + FLAGS_model_version + ":" + method_name +
      ":" + version_token;
}

void writeOutputLlvmAssembly(const std::string& asm_code) {
  std::string output_llvm_file_name = FLAGS_output_llvm;
  if (output_llvm_file_name.empty()) {
    output_llvm_file_name =
        FLAGS_model.substr(0, FLAGS_model.find('.')) + ".compiled.ll";
  }

  std::ofstream output(output_llvm_file_name);
  output << asm_code;
}

c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {
  const std::string& method_name = "forward";
  auto method = mod.get_method(method_name);
  auto graph = method.function().graph()->copy();
  auto sizes = getInputSizesForMethod(method_compile_spec, method_name);

  std::string llvm_asm_code;
  auto compiled = torch::jit::mobile::nnc::aotCompile(method_name, graph, sizes);
  writeOutputLlvmAssembly(compiled.second);

  auto func = std::move(compiled.first);
  func->set_nnc_kernel_id(getNncKernelId(method_name));

  torch::jit::mobile::nnc::CompilationUnit cu;
  cu.register_function(std::move(func));
  return cu.serialize();
}

static auto reg = torch::jit::backend_preprocess_register("nnc", preprocess);

} // namespace

int main(int argc, char** argv) {
  c10::SetUsageMessage(
      "Run NNC AOT compiler for pytorch model. Example usage:\n"
      "build/bin/aot_model_compiler"
      " --model=<model file>"
      " --model_name=<model name>"
      " --model_version=<model version>"
      " --input_dims='1,3,224,224'"
      " [--output_llvm=<llvm assembly output file path>]"
      " [--output_model=<output model file path>]");

  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    std::cout << c10::UsageMessage() << std::endl;
    return 1;
  }

  CAFFE_ENFORCE(!FLAGS_model.empty(), c10::UsageMessage());

  std::string output_model_name = FLAGS_output_model;
  if (output_model_name.empty()) {
    output_model_name =
        FLAGS_model.substr(0, FLAGS_model.find('.')) + ".compiled.pt";
  }

  auto m = torch::jit::load(FLAGS_model);
  m.eval();
  auto frozen_m = torch::jit::freeze_module(m.clone());
  auto graph = frozen_m.get_method("forward").graph();
  torch::jit::OptimizeFrozenGraph(graph, true);

  auto compile_spec = createCompileSpec();
  auto any_dict_ty =
      c10::DictType::create(c10::StringType::get(), c10::AnyType::get());
  auto compiled_module = torch::jit::detail::codegen_backend_module(
      "nnc", frozen_m, compile_spec, any_dict_ty);
  compiled_module._save_for_mobile(output_model_name);
  std::cout << "The compiled model was saved to " << output_model_name
            << std::endl;
  return 0;
}
