#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_exception.h>

#include <caffe2/torch/csrc/jit/backends/xnnpack/compiler/xnn_compiler.h>
#include <torch/csrc/jit/backends/xnnpack/serialization/schema_generated.h>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

class XNNModelWrapper : public CustomClassHolder {
 public:
  XNNExecutor executor_;
  XNNModelWrapper(XNNExecutor executor) : executor_(std::move(executor)){};

  XNNModelWrapper() = delete;

  XNNModelWrapper(const XNNModelWrapper& oldObject) = delete;
};

class XNNPackBackend : public PyTorchBackendInterface {
 public:
  // Constructor.
  // NOLINTNEXTLINE(modernize-use-equals-default)
  explicit XNNPackBackend() {}
  virtual ~XNNPackBackend() override = default;

  bool is_available() override {
    return xnn_status_success == xnn_initialize(/*allocator=*/nullptr);
  }

  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto dict = processed.toGenericDict();

    // Compiling and wrapping exeuction object
    const std::string& ser_model = dict.at("ser_model").toStringRef();
    XNNExecutor executor;
    XNNCompiler::compileModel(ser_model.data(), ser_model.length(), &executor);

    auto model_ptr = c10::make_intrusive<XNNModelWrapper>(std::move(executor));
    auto runtime_handle = IValue::make_capsule(model_ptr);
    auto wrapper = c10::static_intrusive_pointer_cast<XNNModelWrapper>(
        runtime_handle.toCapsule());

    // Packing outputs into generic dict
    c10::Dict<c10::IValue, c10::IValue> handles(
        c10::StringType::get(), c10::AnyType::get());

    c10::Dict<c10::IValue, c10::IValue> ret(
        c10::StringType::get(), c10::AnyType::get());

    ret.insert("runtime", runtime_handle);
    ret.insert("output_shapes", dict.at("outputs"));

    handles.insert("forward", ret);

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
    auto dict = handle.toGenericDict();
    auto output_shapes = dict.at("output_shapes").toList();

    auto capsule = dict.at("runtime").toCapsule();
    auto model_wrapper =
        c10::static_intrusive_pointer_cast<XNNModelWrapper>(capsule);

    XNNExecutor& executor = model_wrapper->executor_;

    std::vector<float*> input_pointers;
    for (int i = 0; i < inputs.size(); ++i) {
      at::IValue val = inputs.get(i);
      TORCH_CHECK(val.isTensor(), "Non-tensor inputs not supported");
      input_pointers.push_back(val.toTensor().data_ptr<float>());
    }

    std::vector<at::Tensor> output_tensors;
    std::vector<float*> output_pointers;
    output_tensors.reserve(output_shapes.size());
    for (int i = 0; i < output_shapes.size(); i++) {
      auto o_shape = output_shapes.get(i).toIntVector();
      auto output = at::empty(o_shape, c10::ScalarType::Float);
      output_tensors.push_back(output);
      output_pointers.push_back(output.data_ptr<float>());
    }

    TORCH_CHECK(
        executor.set_inputs(input_pointers, output_pointers),
        "Number of inputs/outputs does not match expected number of inputs/outputs");
    TORCH_CHECK(executor.forward(), "Failed to invoke XNNPack runtime");

    c10::List<at::Tensor> output_list(output_tensors);
    return c10::impl::toList(output_list);
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
