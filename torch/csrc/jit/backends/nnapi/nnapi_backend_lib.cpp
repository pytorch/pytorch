#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_exception.h>

namespace torch {
namespace jit {

// This file has no implementation yet, but the declarations are necessary to
// register the backend properly and test preprocess
// TODO T91991928: implement compile() and execute()
class NnapiBackend : public PyTorchBackendInterface {
 public:
  // Constructor.
  // NOLINTNEXTLINE(modernize-use-equals-default)
  explicit NnapiBackend() {}
  // NOLINTNEXTLINE(modernize-use-override)
  virtual ~NnapiBackend() = default;

  bool is_available() override {
    return true;
  }

  // Function stub
  // TODO: implement compile
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto handles =
        c10::Dict<std::string, std::vector<std::tuple<std::string, int64_t>>>();
    return c10::impl::toGenericDict(handles);
  }

  // Function stub
  // TODO: implement execute
  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    c10::List<at::Tensor> output_list;
    return c10::impl::toList(output_list);
  }
};

namespace {
constexpr auto backend_name = "nnapi";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto cls = torch::jit::backend<NnapiBackend>(backend_name);
} // namespace

} // namespace jit
} // namespace torch
