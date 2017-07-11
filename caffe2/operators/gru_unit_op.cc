#include "gru_unit_op.h"

namespace caffe2 {
namespace detail {

template <typename T>
inline T sigmoid(T x) {
  return 1.0f / (1.0f + exp(-x));
}

template <typename T>
inline T host_tanh(T x) {
  return 2.0f * sigmoid(2.0f * x) - 1.0f;
}

template <typename T, typename Context>
void GRUUnit(
    int N,
    int D,
    int t,
    const T* H_prev,
    const T* X,
    const int32_t* seqLengths,
    bool drop_states,
    T* H) {
  for (int n = 0; n < N; ++n) {
    const bool valid = t < seqLengths[n];

    for (int d = 0; d < D; ++d) {
      if (valid == false) {
        if (drop_states) {
          H[d] = 0;
        } else {
          H[d] = H_prev[d];
        }
      } else {
        const T update = X[1 * D + d];
        const T output = X[2 * D + d];
        H[d] = H_prev[d] * sigmoid(update) +
            host_tanh(output) * (1.0f - sigmoid(update));
      }
    }

    H_prev += D;
    X += 3 * D;
    H += D;
  }
}

} // namespace detail

namespace {
REGISTER_CPU_OPERATOR(GRUUnit, GRUUnitOp<float, CPUContext>);
OPERATOR_SCHEMA(GRUUnit)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.

Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].

)DOC")
    .Arg(
        "drop_states",
        "Bool to determine if hidden state is zeroes or passed "
        "along for timesteps past the given sequence_length.")
    .Input(0, "hidden_prev", "The previous GRU hidden state.")
    .Input(
        1,
        "gates",
        "Unactivated gate outputs from forget, update, "
        "and output gates, pre-activation.")
    .Input(
        2,
        "seq_lengths",
        "Array of sequence lengths.  "
        "len(seq_lengths) should equal batch size N.")
    .Input(3, "t", "The timestep for this operation.")
    .Output(0, "hidden", "The new GRU hidden state calculated by this op.");
}
}
