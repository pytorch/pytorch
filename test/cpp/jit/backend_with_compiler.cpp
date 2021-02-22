#include <torch/csrc/jit/backends/backend.h>

namespace torch {
namespace jit {

namespace {
std::vector<std::string> parseBlob(const std::string& blob) {
  std::vector <std::string> result;
  std::stringstream s_stream(blob);
  while (s_stream.good()) {
    std::string substr;
    getline(s_stream, substr, ',');
    result.push_back(substr);
  }
  return result;
}
}
// This test JIT backend is intended to do the minimal amount of work
// necessary to test that the JIT backend registration endpoints and
// code generation are working correctly. It is not intended to
// produce numerically correct results.
class BackendWithCompiler : public PyTorchBackendInterface {
 public:
  // Constructor.
  explicit BackendWithCompiler() {}
  virtual ~BackendWithCompiler() = default;

  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {

    auto handles = processed.toGenericDict();
    auto test = c10::impl::toTypedDict<std::string, std::string>(handles);
    for (const auto& handle : test) {
      std::cout << handle.key() << ": " << handle.value() << std::endl;
    }
    return c10::impl::toGenericDict(test);
  }

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    TORCH_INTERNAL_ASSERT(handle.isString());
    TORCH_INTERNAL_ASSERT(inputs.size() == 2);
    c10::IValue val0 = inputs[0];
    at::Tensor x = val0.toTensor();
    c10::IValue val1 = inputs[1];
    at::Tensor h = val1.toTensor();

    c10::List<at::Tensor> output_list;
    auto tokens = parseBlob(handle.toStringRef());
    for (const auto& token: tokens) {
      if (token == "aten::add") {
        output_list.emplace_back(x.add_(h, 1.0));
      }
    }
    return c10::impl::toList(output_list);
  }
};

// Actual AOT compilation function. Put here for demonstration of backend
// as a whole piece. It's used when compilation is required. A dummy function
// can be passed when there's no usage of compilation in runtime backend lib.
c10::IValue preprocess(
    const Module& mod,
    const c10::Dict<IValue, IValue>& method_compile_spec) {
  // The output of this process would produce a dictionary
  // Key: method name.
  // Val: compiled blob (represented by a string).
  c10:Dict<IValue, IValue> compiled(StringType::get(), StringType::get());
  for (const auto& method : mod.get_methods()) {
    auto graph = method.function().graph()->copy();
    auto key = method.name();
    std::stringstream ss;
    for (const auto& node : graph->nodes()) {
      switch (node->kind()) {
        case prim::Constant:
          ss << node->kind().toDisplayString() << "#" << toIValue(node->output()).value();
          break;
        case aten::add:
          ss << node->kind().toQualString();
          break;
        default:
          TORCH_CHECK(true, "The node of ", node->kind().toQualString(),
                      " is not supported in this compiler. Source code: ",
                      node->sourceRange().text());
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

namespace {
static auto cls = torch::jit::backend<BackendWithCompiler>("backend_with_compiler_demo", preprocess);
}

} // namespace jit
} // namespace torch
