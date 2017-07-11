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

template <typename T, typename Context>
void GRUUnitGradient(
    int N,
    int D,
    int t,
    const T* H_prev,
    const T* X,
    const int32_t* seqLengths,
    const T* H,
    const T* H_diff,
    bool drop_states,
    T* H_prev_diff,
    T* X_diff) {
  for (int n = 0; n < N; ++n) {
    const bool valid = t < seqLengths[n];

    for (int d = 0; d < D; ++d) {
      T* h_prev_diff = H_prev_diff + d;
      T* reset_diff = X_diff + 0 * D + d;
      T* update_diff = X_diff + 1 * D + d;
      T* output_diff = X_diff + 2 * D + d;

      if (!valid) {
        if (drop_states) {
          *h_prev_diff = 0;
        } else {
          *h_prev_diff = H_diff[d];
        }
        *reset_diff = 0;
        *update_diff = 0;
        *output_diff = 0;
      } else {
        // Calculate Gate Outputs
        const T u = sigmoid(X[1 * D + d]);
        const T o = host_tanh(X[2 * D + d]);

        *h_prev_diff = H_diff[d] * u;
        *reset_diff = 0; // 0 contribution to gradient from this operation
        *update_diff = (H_diff[d] * H_prev[d] - H_diff[d] * o) * u * (1.0f - u);
        *output_diff = H_diff[d] * (1.0f - u) * (1.0f - o * o);
      }
    }

    H_prev += D;
    X += 3 * D;
    H += D;
    H_diff += D;
    X_diff += 3 * D;
    H_prev_diff += D;
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
REGISTER_CPU_OPERATOR(GRUUnitGradient, GRUUnitGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(GRUUnitGradient).NumInputs(6).NumOutputs(2);

class GetGRUUnitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "GRUUnitGradient",
        "",
        vector<string>{I(0), I(1), I(2), I(3), O(0), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(GRUUnit, GetGRUUnitGradient);
}
} // namespace caffe2
