#include "recurrent_network_op.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {
CAFFE_KNOWN_TYPE(detail::ScratchWorkspaces);

REGISTER_CPU_OPERATOR(RecurrentNetwork, RecurrentNetworkOp<float, CPUContext>);
OPERATOR_SCHEMA(RecurrentNetwork)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2, INT_MAX)
    .SetDoc(R"DOC(

Run the input network in a recurrent fashion. This can be used to
implement fairly general recurrent neural networks (RNNs).

The operator proceeds as follows.

- First, initialized the states from the input recurrent states
- For each timestep T, apply the links (that map offsets from input/output
  tensors into the inputs/outputs for the `step` network)
- Finally, alias the recurrent states to the specified output blobs.

This is a fairly special-case meta-operator, and so the implementation
is somewhat complex. It trades of generality (and frankly usability)
against performance and control (compared to e.g. TF
dynamic_rnn, Theano scan, etc).

See the usage examples for a flavor of how to use it.

)DOC");

REGISTER_CPU_OPERATOR(
    RecurrentNetworkGradient,
    RecurrentNetworkGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(RecurrentNetworkGradient);

REGISTER_CPU_OPERATOR(
    rnn_internal_accumulate_gradient_input,
    AccumulateInputGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(rnn_internal_accumulate_gradient_input)
    .NumInputs(2)
    .NumOutputs(1)
    .Private()
    .SetDoc("--internal--");

struct GetRecurrentNetworkGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argsHelper(def_);
    auto params = argsHelper.GetRepeatedArgument<int32_t>("param");
    auto recurrentInputs =
        argsHelper.GetRepeatedArgument<int32_t>("initial_recurrent_state_ids");

    std::vector<std::string> gradientInputs;

    // Argument specifies which outputs have external gradient, (0) by default
    auto outputs_with_grads =
        argsHelper.GetRepeatedArgument<int32_t>("outputs_with_grads");
    CAFFE_ENFORCE(outputs_with_grads.size() > 0);
    for (auto id : outputs_with_grads) {
      gradientInputs.push_back(GO(id));
    }

    // All inputs and outputs are passed back
    for (int i = 0; i < def_.input_size(); ++i) {
      gradientInputs.push_back(I(i));
    }
    for (int i = 0; i < def_.output_size(); ++i) {
      gradientInputs.push_back(O(i));
    }

    // We calculate gradients only for parameters and recurrent inputs
    std::vector<std::string> gradientOutputs;
    gradientOutputs.push_back(GI(0));
    for (auto id : params) {
      gradientOutputs.push_back(GI(id));
    }
    for (auto id : recurrentInputs) {
      gradientOutputs.push_back(GI(id));
    }

    VLOG(1) << "Gradient blobs: " << Join(", ", gradientOutputs);

    return SingleGradientDef(
        "RecurrentNetworkGradient", "", gradientInputs, gradientOutputs);
  }
};

REGISTER_GRADIENT(RecurrentNetwork, GetRecurrentNetworkGradient);

namespace detail {
void extractLinks(
    OperatorBase* op,
    const std::string& internalArg,
    const std::string& externalArg,
    const std::string& offsetArg,
    const std::string& windowArg,
    std::vector<detail::Link>* links) {
  const auto& internal = op->GetRepeatedArgument<std::string>(internalArg);
  const auto& external = op->GetRepeatedArgument<std::string>(externalArg);
  const auto& offset = op->GetRepeatedArgument<int32_t>(offsetArg);
  const auto& window = op->GetRepeatedArgument<int32_t>(
      windowArg, vector<int32_t>(offset.size(), 1));
  CAFFE_ENFORCE_EQ(
      internal.size(),
      offset.size(),
      "internal/offset mismatch: ",
      internalArg,
      " ",
      externalArg);
  CAFFE_ENFORCE_EQ(
      external.size(),
      offset.size(),
      "external/offset mismatch: ",
      externalArg,
      " ",
      offsetArg);
  CAFFE_ENFORCE_EQ(
      external.size(),
      window.size(),
      "external/window mismatch: ",
      externalArg,
      " ",
      windowArg);
  for (auto i = 0; i < internal.size(); ++i) {
    detail::Link l;
    l.internal = internal[i];
    l.external = external[i];
    l.offset = offset[i];
    l.window = window[i];
    links->push_back(l);
  }
}
} // namespace detail
}
