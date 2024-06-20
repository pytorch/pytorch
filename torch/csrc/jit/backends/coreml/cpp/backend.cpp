#include <torch/csrc/jit/backends/backend.h>
#include <torch/script.h>

namespace {

class CoreMLBackend : public torch::jit::PyTorchBackendInterface {
 public:
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    TORCH_CHECK(false, "The CoreML backend is not supported on server side!");
    auto handles = c10::Dict<std::string, std::string>();
    return c10::impl::toGenericDict(handles);
  }

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    TORCH_CHECK(false, "The CoreML backend is not supported on server side!");
    c10::List<at::Tensor> output_list;
    return c10::impl::toList(output_list);
  }

  bool is_available() override {
    return false;
  }
};

static auto cls = torch::jit::backend<CoreMLBackend>("coreml");

} // namespace
