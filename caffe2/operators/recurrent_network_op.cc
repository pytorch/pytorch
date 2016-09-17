#include "recurrent_network_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(RecurrentNetwork, RecurrentNetworkOp<float, CPUContext>);
OPERATOR_SCHEMA(RecurrentNetwork)
    .NumInputs(3, INT_MAX)
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

struct GetRecurrentNetworkGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    std::vector<std::string> gradientInputs;

    // Grad output of output (0)
    gradientInputs.push_back(GO(0));

    // All inputs and outputs are passed back
    for (int i = 0; i < def_.input_size(); ++i) {
      gradientInputs.push_back(I(i));
    }

    for (int i = 0; i < def_.output_size(); ++i) {
      gradientInputs.push_back(O(i));
    }

    // Grad WRT all inputs - only a few of these are actually filled
    // (in particular, parameters and input), but this should be OK
    // for now.
    std::vector<std::string> gradientOutputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      gradientOutputs.push_back(GI(i));
    }

    return SingleGradientDef(
        "RecurrentNetworkGradient", "", gradientInputs, gradientOutputs);
  }
};

REGISTER_GRADIENT(RecurrentNetwork, GetRecurrentNetworkGradient);
}
}
