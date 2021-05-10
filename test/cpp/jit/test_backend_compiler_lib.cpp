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
  // NOLINTNEXTLINE(modernize-use-equals-default)
  explicit BackendWithCompiler() {}
  // NOLINTNEXTLINE(modernize-use-override)
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
        // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
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
constexpr auto backend_name = "backend_with_compiler_demo";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto cls = torch::jit::backend<BackendWithCompiler>(backend_name);
} // namespace

} // namespace jit
} // namespace torch
