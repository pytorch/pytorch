#include <vector>

#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/mobile/nnc/context.h>
#include <torch/script.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

class NNCBackend : public PyTorchBackendInterface {
 public:
  explicit NNCBackend() = default;
  ~NNCBackend() override = default;

  bool is_available() override {
    return true;
  }

  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    cu_ = std::make_shared<CompilationUnit>(processed);

    // Input method_compile_spec:
    //   Key: method name
    //   Value: compile spec for each method
    // Output:
    //   Key: method name
    //   Value: a backend handle for each method
    auto spec =
        c10::impl::toTypedDict<std::string, at::IValue>(method_compile_spec);
    auto handles = c10::Dict<std::string, std::string>();
    for (const auto& it : spec) {
      // The handle for each method is the key (method name) itself.
      handles.insert(it.key(), it.key());
    }
    return c10::impl::toGenericDict(handles);
  }

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    const std::string& method_name = handle.toStringRef();
    auto function_name = c10::QualifiedName(method_name);
    return cu_->run(function_name, inputs);
  }

 private:
  std::shared_ptr<CompilationUnit> cu_;
};

namespace {
static const auto cls = torch::jit::backend<NNCBackend>("nnc");
} // namespace

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
