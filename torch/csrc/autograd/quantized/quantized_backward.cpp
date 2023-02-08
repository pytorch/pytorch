#include <ATen/native/quantized/PackedParams.h>
#include <torch/csrc/autograd/functions/utils.h>// compute_requires_grad
#include <torch/csrc/autograd/FunctionsManual.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <torch/custom_class.h>
#include <torch/library.h>

// This function is very specific to QNNPACK, if application does not use QNNPACK,
// Gradient function dispatching does not need to be registered.

#ifdef USE_PYTORCH_QNNPACK
namespace caffe2 {
CAFFE_KNOWN_TYPE(c10::intrusive_ptr<PackedLinearWeightsQnnp>);
}
namespace{
using namespace torch::autograd;
using namespace generated::details;
using namespace at;
// Gradient into argument 1 (`input`). TODO: Support bias as well.
struct TORCH_API PackedLinearWeightDynamicBackward : public TraceableFunction {
  // Backprop adjoint through layer.
  at::Tensor matmul_backward(const at::Tensor& grad) {
    auto& ctx = at::globalContext();
    at::Tensor weight;
    c10::optional<at::Tensor> bias;
    std::tie(weight, bias) = self_->unpack();
    weight = at::permute(weight, {1,0});
    int gradN = grad.sizes().vec().size();
    std::vector<int64_t> gradArgs;
    for(int i = 0 ; i < gradN; i++){
      gradArgs.push_back(i);
    }
    c10::intrusive_ptr<LinearPackedParamsBase> prepackedWeightTransposed;
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      TORCH_CHECK(
          weight.scalar_type() == at::kQInt8,
          "QNNPACK only supports INT8 bit width currently. Got ",
          c10::toString(weight.scalar_type()));
      prepackedWeightTransposed = PackedLinearWeightsQnnp::prepack(
            std::move(weight), nullopt);
      return prepackedWeightTransposed->apply_dynamic(grad);
    }
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_prepack ",
        toString(ctx.qEngine()));
  }

  // This is invoked by the PyTorch backward() process.
  variable_list apply(variable_list&& grads /*adjoint*/) override {
    std::lock_guard<std::mutex> lock(mutex_);
    IndexRangeGenerator gen;
    auto input_ix = gen.range(1);
    variable_list grad_inputs(gen.size());
    bool any_grad_defined = any_variable_defined(grads);
    const auto& grad = grads[0]; // adjoint
    if (should_compute_output({input_ix})) {
      auto grad_result = any_grad_defined ? matmul_backward(grad) : at::Tensor();
      copy_range(grad_inputs, input_ix, grad_result);
    }
    return grad_inputs;
  }
  std::string name() const override {
    return "PackedLinearWeightDynamicBackward";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    // resetting weight is necessary?
  }
  // SavedVariable self_; // TODO: Any ref counting needed?
  c10::intrusive_ptr<LinearPackedParamsBase> self_;
  std::vector<int64_t> input_sizes_;
};

}// namespace

at::Tensor packed_linear_weight_grad(at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      bool reduce_range){
    at::AutoDispatchBelowADInplaceOrView guard;
    auto _any_requires_grad = compute_requires_grad(input);
    std::shared_ptr<PackedLinearWeightDynamicBackward> grad_fn;

    if (_any_requires_grad) {
        grad_fn = std::shared_ptr<PackedLinearWeightDynamicBackward>(
            new PackedLinearWeightDynamicBackward(), deleteNode);
        grad_fn->self_ = packed_weight;
        grad_fn->input_sizes_ = input.sizes().vec();
        grad_fn->set_next_edges(collect_next_edges(input));
    }
    static auto op = at::Dispatcher::singleton()
    .findSchemaOrThrow("quantized::linear_dynamic", "")
    .typed<decltype(packed_linear_weight_grad)>();
    auto output = op.call(input, packed_weight, reduce_range);
    if (grad_fn) {
        assert(input.requires_grad());
        assert(!output.requires_grad());
        set_history(flatten_tensor_args(output), grad_fn);
        assert(output.requires_grad());
    }
    return output;
}

#endif

namespace at {
namespace native {
namespace {
#ifdef USE_PYTORCH_QNNPACK
TORCH_LIBRARY_IMPL(quantized, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic"),
      TORCH_FN(packed_linear_weight_grad));
}
#endif
} // namespace
} // namespace native
} // namespace at
