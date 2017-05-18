#include "lstm_unit_op.h"

namespace caffe2 {
namespace detail {

// Using macros here instead of linlined functions
// Needed for performance: g++ inliner loses 10-20%
#undef C2_EIGEN_SIGMOID_INLINE
#undef C2_EIGEN_HOST_TANH_INLINE
#define C2_EIGEN_SIGMOID_INLINE(x) (1.0f / ((-(x)).exp() + 1.0))
#define C2_EIGEN_HOST_TANH_INLINE(x) \
  (2.0 * C2_EIGEN_SIGMOID_INLINE(2.0 * (x)) - 1.0)

template <typename T, typename Context>
void LSTMUnit(
    int N,
    int D,
    int t,
    const T* H_prev,
    const T* C_prev,
    const T* X,
    const int32_t* seqLengths,
    bool drop_states,
    T* C,
    T* H,
    const T& forget_bias,
    Context* context) {
  for (int n = 0; n < N; ++n) {
    const bool valid = t < seqLengths[n];

    // create data aliases into Eigen vectors
    EigenVectorArrayMap<T> vH(H, D);
    EigenVectorArrayMap<T> vC(C, D);
    ConstEigenVectorArrayMap<T> vH_prev(H_prev, D);
    ConstEigenVectorArrayMap<T> vC_prev(C_prev, D);
    ConstEigenVectorArrayMap<T> vX0(X + 0 * D, D);
    ConstEigenVectorArrayMap<T> vX1(X + 1 * D, D);
    ConstEigenVectorArrayMap<T> vX2(X + 2 * D, D);
    ConstEigenVectorArrayMap<T> vX3(X + 3 * D, D);

    if (valid == false) {
      if (drop_states) {
        vH.setConstant((T)(0.0));
        vC.setConstant((T)(0.0));
      } else {
        vH = vH_prev;
        vC = vC_prev;
      }
    } else {
      vC = C2_EIGEN_SIGMOID_INLINE(vX1 + forget_bias) * vC_prev +
          C2_EIGEN_SIGMOID_INLINE(vX0) * C2_EIGEN_HOST_TANH_INLINE(vX3);
      vH = C2_EIGEN_SIGMOID_INLINE(vX2) * C2_EIGEN_HOST_TANH_INLINE(vC);
    }

    H_prev += D;
    C_prev += D;
    X += 4 * D;
    C += D;
    H += D;
  }
}

template <typename T, typename Context>
void LSTMUnitGradient(
    int N,
    int D,
    int t,
    const T* C_prev,
    const T* X,
    const int32_t* seqLengths,
    const T* C,
    const T* H,
    const T* C_diff,
    const T* H_diff,
    bool drop_states,
    T* H_prev_diff,
    T* C_prev_diff,
    T* X_diff,
    const T forget_bias,
    Context* context) {
  for (int n = 0; n < N; ++n) {
    const bool valid = t < seqLengths[n];

    // create data aliases into Eigen vectors
    ConstEigenVectorArrayMap<T> vC_prev(C_prev, D);
    ConstEigenVectorArrayMap<T> vX(X, 4 * D);
    ConstEigenVectorArrayMap<T> vX0(X + 0 * D, D);
    ConstEigenVectorArrayMap<T> vX1(X + 1 * D, D);
    ConstEigenVectorArrayMap<T> vX2(X + 2 * D, D);
    ConstEigenVectorArrayMap<T> vX3(X + 3 * D, D);
    ConstEigenVectorArrayMap<T> vC(C, D);
    ConstEigenVectorArrayMap<T> vH(H, D);
    ConstEigenVectorArrayMap<T> vC_diff(C_diff, D);
    ConstEigenVectorArrayMap<T> vH_diff(H_diff, D);
    // Output
    EigenVectorArrayMap<T> vH_prev_diff(H_prev_diff, D);
    EigenVectorArrayMap<T> vC_prev_diff(C_prev_diff, D);
    EigenVectorArrayMap<T> vX_diff(X_diff, 4 * D);
    EigenVectorArrayMap<T> vX0_diff(X_diff + 0 * D, D);
    EigenVectorArrayMap<T> vX1_diff(X_diff + 1 * D, D);
    EigenVectorArrayMap<T> vX2_diff(X_diff + 2 * D, D);
    EigenVectorArrayMap<T> vX3_diff(X_diff + 3 * D, D);

    if (!valid) {
      if (drop_states) {
        vH_prev_diff.setConstant((T)(0.0));
        vC_prev_diff.setConstant((T)(0.0));
      } else {
        vH_prev_diff = vH_diff;
        vC_prev_diff = vC_diff;
      }
      vX_diff.setConstant((T)(0.0));
    } else {
      const Eigen::Array<T, Eigen::Dynamic, 1> i = C2_EIGEN_SIGMOID_INLINE(vX0);
      const Eigen::Array<T, Eigen::Dynamic, 1> f =
          C2_EIGEN_SIGMOID_INLINE(vX1 + forget_bias);
      const Eigen::Array<T, Eigen::Dynamic, 1> o = C2_EIGEN_SIGMOID_INLINE(vX2);
      const Eigen::Array<T, Eigen::Dynamic, 1> g =
          C2_EIGEN_HOST_TANH_INLINE(vX3);
      const Eigen::Array<T, Eigen::Dynamic, 1> host_tanh_c =
          C2_EIGEN_HOST_TANH_INLINE(vC);
      const Eigen::Array<T, Eigen::Dynamic, 1> c_term_diff =
          vC_diff + vH_diff * o * (1 - host_tanh_c * host_tanh_c);
      vC_prev_diff = c_term_diff * f;
      vH_prev_diff = 0; // not used in 'valid' case
      vX0_diff = c_term_diff * g * i * (1 - i);
      vX1_diff = c_term_diff * vC_prev * f * (1 - f);
      vX2_diff = vH_diff * host_tanh_c * o * (1 - o);
      vX3_diff = c_term_diff * i * (1 - g * g);
    }

    C_prev += D;
    X += 4 * D;
    C += D;
    H += D;
    C_diff += D;
    H_diff += D;
    X_diff += 4 * D;
    H_prev_diff += D;
    C_prev_diff += D;
  }
}
#undef C2_EIGEN_SIGMOID_INLINE
#undef C2_EIGEN_HOST_TANH_INLINE
} // namespace detail

namespace {
REGISTER_CPU_OPERATOR(LSTMUnit, LSTMUnitOp<float, CPUContext>);
OPERATOR_SCHEMA(LSTMUnit)
    .NumInputs(5)
    .NumOutputs(2)
    .SetDoc(R"DOC(
LSTMUnit computes the activations of a standard LSTM (without peephole
connections), in a sequence-length aware fashion.

Concretely, given the (fused) inputs X (TxNxD), the previous cell
state (NxD), and the sequence lengths (N), computes the LSTM
activations, avoiding computation if the input is invalid (as in, the
value at X{t][n] >= seqLengths[n].

)DOC")
    .Arg("forget_bias", "Bias term to add in while calculating forget gate");
REGISTER_CPU_OPERATOR(LSTMUnitGradient, LSTMUnitGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(LSTMUnitGradient).NumInputs(9).NumOutputs(3);

class GetLSTMUnitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LSTMUnitGradient",
        "",
        vector<string>{I(0), I(1), I(2), I(3), I(4), O(0), O(1), GO(0), GO(1)},
        vector<string>{GI(0), GI(1), GI(2)});
  }
};
REGISTER_GRADIENT(LSTMUnit, GetLSTMUnitGradient);
}
}
