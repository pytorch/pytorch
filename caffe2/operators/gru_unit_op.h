#ifndef CAFFE2_OPERATORS_GRU_UNIT_OP_H_
#define CAFFE2_OPERATORS_GRU_UNIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

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
    T* H,
    Context* /*context*/) {
  for (int n = 0; n < N; ++n) {
    const bool valid = seqLengths == nullptr || t < seqLengths[n];

    for (int d = 0; d < D; ++d) {
      if (!valid) {
        if (drop_states) {
          H[d] = 0;
        } else {
          H[d] = H_prev[d];
        }
      } else {
        const T update = X[1 * D + d];
        const T output = X[2 * D + d];
        T sigmoid_update = sigmoid(update);
        H[d] = H_prev[d] * sigmoid_update +
            host_tanh(output) * (1.0f - sigmoid_update);
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
    T* X_diff,
    Context* /*context*/) {
  for (int n = 0; n < N; ++n) {
    const bool valid = seqLengths == nullptr || t < seqLengths[n];

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

template <typename T, typename Context>
class GRUUnitOp : public Operator<Context> {
 public:
  GRUUnitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        drop_states_(this->template GetSingleArgument<bool>(
            "drop_states",
            false)),
        sequence_lengths_(this->template GetSingleArgument<bool>(
            "sequence_lengths",
            true)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // handle potentially-missing sequence lengths input
    const size_t TIMESTEP = SEQ_LENGTHS + (sequence_lengths_ ? 1 : 0);

    // Extract N
    const auto N = Input(HIDDEN_T_M_1).size(1);

    // Gates: 1xNxG
    const auto G = Input(GATES).size(2);
    const auto D = Input(HIDDEN_T_M_1).size(2);

    CAFFE_ENFORCE_EQ(3 * D, G);
    const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
    const auto* X = Input(GATES).template data<T>();

    const int32_t* seqLengths = nullptr;
    if (sequence_lengths_) {
      CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).numel(), N);
      seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
    }

    const auto t = static_cast<OperatorBase*>(this)
                       ->Input<Tensor>(TIMESTEP, CPU)
                       .template data<int32_t>()[0];
    Output(HIDDEN_T)->ResizeLike(Input(HIDDEN_T_M_1));
    auto* H = Output(HIDDEN_T)->template mutable_data<T>();

    detail::GRUUnit<T, Context>(
        N, D, t, H_prev, X, seqLengths, drop_states_, H, &context_);
    return true;
  }

 protected:
  INPUT_TAGS(HIDDEN_T_M_1, GATES, SEQ_LENGTHS);
  // additional input tags are determined dynamically based on whether
  // sequence_lengths is present.
  OUTPUT_TAGS(HIDDEN_T);

 private:
  bool drop_states_;
  bool sequence_lengths_;
};

template <typename T, typename Context>
class GRUUnitGradientOp : public Operator<Context> {
 public:
  GRUUnitGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        drop_states_(this->template GetSingleArgument<bool>(
            "drop_states",
            false)),
        sequence_lengths_(this->template GetSingleArgument<bool>(
            "sequence_lengths",
            true)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // handle potentially-missing sequence lengths input
    const size_t inputOffset = SEQ_LENGTHS + (sequence_lengths_ ? 1 : 0);
    const size_t TIMESTEP = inputOffset;
    const size_t HIDDEN_T = inputOffset + 1;
    const size_t HIDDEN_T_GRAD = inputOffset + 2;

    // Extract N
    const auto N = Input(HIDDEN_T_M_1).size(1);

    // Gates: 1xNxG
    const auto G = Input(GATES).size(2);
    const auto D = Input(HIDDEN_T_M_1).size(2);

    CAFFE_ENFORCE_EQ(3 * D, G);
    const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
    const auto* X = Input(GATES).template data<T>();
    const auto t = static_cast<OperatorBase*>(this)
                       ->Input<Tensor>(TIMESTEP, CPU)
                       .template data<int32_t>()[0];
    const auto* H = Input(HIDDEN_T).template data<T>();
    const auto* H_diff = Input(HIDDEN_T_GRAD).template data<T>();

    const int32_t* seqLengths = nullptr;
    if (sequence_lengths_) {
      CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).numel(), N);
      seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
    }

    Output(HIDDEN_T_M_1_GRAD)->ResizeLike(Input(HIDDEN_T_M_1));
    auto* H_prev_diff = Output(HIDDEN_T_M_1_GRAD)->template mutable_data<T>();
    Output(GATES_GRAD)->ResizeLike(Input(GATES));
    auto* X_diff = Output(GATES_GRAD)->template mutable_data<T>();

    detail::GRUUnitGradient<T, Context>(
        N,
        D,
        t,
        H_prev,
        X,
        seqLengths,
        H,
        H_diff,
        drop_states_,
        H_prev_diff,
        X_diff,
        &context_);
    return true;
  }

 protected:
  INPUT_TAGS(HIDDEN_T_M_1, GATES, SEQ_LENGTHS);
  OUTPUT_TAGS(HIDDEN_T_M_1_GRAD, GATES_GRAD);

 private:
  bool drop_states_;
  bool sequence_lengths_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GRU_UNIT_OP_H_
