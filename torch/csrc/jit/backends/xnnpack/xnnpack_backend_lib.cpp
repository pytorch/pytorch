#include <ATen/Utils.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_exception.h>

#include <xnnpack.h>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

class XNNPackBackend : public PyTorchBackendInterface {
 public:
  // Constructor.
  // NOLINTNEXTLINE(modernize-use-equals-default)
  explicit XNNPackBackend() {}
  // NOLINTNEXTLINE(modernize-use-override)
  virtual ~XNNPackBackend() = default;

  bool is_available() override {
    return xnn_status_success == xnn_initialize(/*allocator=*/nullptr);
  }

  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto dict = processed.toGenericDict();
    c10::Dict<c10::IValue, c10::IValue> handles(
        c10::StringType::get(), c10::AnyType::get());
    handles.insert("forward", dict);

    return handles;
  }

  // Currently this is not implemented, and everything is computed a head of
  // time the current implementation just takes the computed results from ahead
  // of time and grabs them. The inputs are fed in through the compile spec for
  // the sake of testing. In reality, the inputs will be fed in at this stage
  // and ran here.
  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    auto answer = handle.toGenericDict().at("Answer");

    return answer.toList();
  }
};

namespace {
constexpr auto backend_name = "xnnpack";
static auto cls = torch::jit::backend<XNNPackBackend>(backend_name);
} // namespace

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
