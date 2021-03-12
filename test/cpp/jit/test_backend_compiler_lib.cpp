#include <torch/csrc/jit/backends/backend.h>

namespace torch {
namespace jit {

// Implementation of a PyTorch Backend that can process, compile and execute
// TorchScript Modules composed of 'add' and 'sub' operators. It just supports
// for modules that implement a sum or subtraction of 2 inputs (i.e. in1 + in2
// or in1 - in2). Hence the methods of the models expect exactly 2 inputs of
// type Tensor. This backend is used to demonstrate the flow of compilation and
// execution with minimum amount of work. It's not intended to a practical
// backend that can be used for actual inference.

// Implementation details:
//
// Compilation
// 1. A backend with minimum compilation features, "backend_with_compiler_demo"
// is added.
// 2. The compilation happens AOT in the preprocess function registered to this
// backend.
// 3. Compiled results are stored in a string blob for each method. They are
// serialized to the lowered module with __getstate__ function.
// 4. Error message with model source code is thrown, for features not handled
// by the backend compiler.
//
// Runtime
// 1. The compiled blob is loaded in __setstate__ method.
// 2. The compile function of the backend: parse the preprocessed blob to the
// format (a list of tokens) that the backend can understand.
// 3. The execute function of the backend executes the specified method
// (handle).

namespace {
std::vector<std::string> parseMethodHandle(const std::string& blob) {
  std::vector<std::string> result;
  std::stringstream s_stream(blob);
  while (s_stream.good()) {
    std::string substr;
    getline(s_stream, substr, ',');
    result.push_back(substr);
  }
  return result;
}
} // namespace

class BackendWithCompiler : public PyTorchBackendInterface {
 public:
  // Constructor.
  explicit BackendWithCompiler() {}
  virtual ~BackendWithCompiler() = default;

  bool is_available() override {
    return true;
  }

  // Since the actual compilation is done AOT,
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto dict = processed.toGenericDict();
    auto handles = c10::Dict<std::string, std::vector<std::string>>();
    for (const auto& kv : dict) {
      auto tokens = parseMethodHandle(kv.value().toStringRef());
      handles.insert(kv.key().toStringRef(), tokens);
    }
    return c10::impl::toGenericDict(handles);
  }

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    TORCH_INTERNAL_ASSERT(inputs.size() == 2);
    c10::IValue val0 = inputs[0];
    at::Tensor x = val0.toTensor();
    c10::IValue val1 = inputs[1];
    at::Tensor h = val1.toTensor();

    c10::List<at::Tensor> output_list;
    double scalar_val = 1.0;
    for (const auto& token : handle.toList()) {
      IValue val = token;
      auto instruction = std::string(IValue(token).toStringRef());
      double const_val = 1.0;
      if (instruction.rfind("prim::Constant", 0) == 0) {
        TORCH_CHECK(
            instruction.size() > 15,
            "Constant value is expected in ",
            instruction);
        auto sub = instruction.substr(15);
        const_val = stod(sub);
      } else if (token == "aten::add") {
        output_list.emplace_back(x.add(h, const_val));
      } else if (token == "aten::sub") {
        output_list.emplace_back(x.sub(h, const_val));
      } else {
        TORCH_CHECK(
            false,
            "Instruction, ",
            instruction,
            " is not supported. ",
            "Contact the backend POC for details. ");
      }
    }
    return c10::impl::toList(output_list);
  }
};

namespace {
// For this backend, the actual compilation happens in preprocess function AOT.
// Put here for demonstration of backend
// as a whole piece. It's used when compilation is required. A dummy function
// can be passed when there's no usage of compilation in runtime backend lib.
c10::IValue preprocess(
    const Module& mod,
    const c10::Dict<IValue, IValue>& method_compile_spec) {
  // The output of this process would produce a dictionary
  // Key: method name.
  // Val: compiled blob (represented by a string).
  c10::Dict<IValue, IValue> compiled(StringType::get(), StringType::get());
  for (const auto& method : mod.get_methods()) {
    const auto graph = method.function().graph()->copy();
    auto key = method.name();
    std::stringstream ss;
    for (const auto& node : graph->nodes()) {
      switch (node->kind()) {
        case prim::Constant:
          ss << node->kind().toDisplayString() << "#"
             << toIValue(node->output()).value();
          break;
        case aten::add:
          ss << node->kind().toQualString();
          break;
        case aten::sub:
          ss << node->kind().toQualString();
          break;
        default:
          TORCH_CHECK(
              false,
              "The node of ",
              node->kind().toQualString(),
              " is not supported in this compiler. Source code: ",
              node->sourceRange().str());
          break;
      }
      ss << ",";
    }
    std::string blob = ss.str();
    if (!blob.empty()) {
      blob.pop_back();
    }
    compiled.insert(method.name(), blob);
  }
  return compiled;
}

static auto cls = torch::jit::backend<BackendWithCompiler>(
    "backend_with_compiler_demo",
    preprocess);
} // namespace

} // namespace jit
} // namespace torch
