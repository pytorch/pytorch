#ifndef CAFFE2_OPERATORS_LSTM_UNIT_OP_H_
#define CAFFE2_OPERATORS_LSTM_UNIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace detail {

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
    Context* context);

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
    Context* context);
}; // namespace detail

template <typename T, typename Context>
class LSTMUnitOp : public Operator<Context> {
 public:
  LSTMUnitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        forget_bias_(
            static_cast<T>(OperatorBase::template GetSingleArgument<float>(
                "forget_bias",
                0.0))),
        drop_states_(OperatorBase::template GetSingleArgument<bool>(
            "drop_states",
            false)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;

  bool RunOnDevice() override {
    // Extract N
    const auto N = Input(CELL_T_M_1).dim(1);

    // Gates: 1xNxG
    const auto G = Input(GATES).dim(2);
    const auto D = Input(CELL_T_M_1).dim(2);

    CAFFE_ENFORCE_EQ(4 * D, G);
    const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
    const auto* C_prev = Input(CELL_T_M_1).template data<T>();
    const auto* X = Input(GATES).template data<T>();
    CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).size(), N);
    const auto* seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
    const auto t = OperatorBase::Input<Tensor<CPUContext>>(TIMESTEP)
                       .template data<int32_t>()[0];
    Output(CELL_T)->ResizeLike(Input(CELL_T_M_1));
    auto* C = Output(CELL_T)->template mutable_data<T>();
    Output(HIDDEN_T)->ResizeLike(Input(CELL_T_M_1));
    auto* H = Output(HIDDEN_T)->template mutable_data<T>();
    detail::LSTMUnit<T, Context>(
        N,
        D,
        t,
        H_prev,
        C_prev,
        X,
        seqLengths,
        drop_states_,
        C,
        H,
        forget_bias_,
        &context_);
    return true;
  }

 protected:
  INPUT_TAGS(HIDDEN_T_M_1, CELL_T_M_1, GATES, SEQ_LENGTHS, TIMESTEP);
  OUTPUT_TAGS(HIDDEN_T, CELL_T);

  T forget_bias_;

 private:
  bool drop_states_;
};

template <typename T, typename Context>
class LSTMUnitGradientOp : public Operator<Context> {
 public:
  LSTMUnitGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        forget_bias_(
            static_cast<T>(OperatorBase::template GetSingleArgument<float>(
                "forget_bias",
                0.0))),
        drop_states_(OperatorBase::template GetSingleArgument<bool>(
            "drop_states",
            false)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Extract N
    const auto N = Input(CELL_T_M_1).dim(1);

    // Gates: 1xNxG
    const auto G = Input(GATES).dim(2);
    const auto D = Input(CELL_T_M_1).dim(2);

    CAFFE_ENFORCE_EQ(4 * D, G);
    const auto* C_prev = Input(CELL_T_M_1).template data<T>();
    const auto* X = Input(GATES).template data<T>();
    const auto t = OperatorBase::Input<Tensor<CPUContext>>(TIMESTEP)
                       .template data<int32_t>()[0];
    const auto* C = Input(CELL_T).template data<T>();
    const auto* H = Input(HIDDEN_T).template data<T>();
    const auto* C_diff = Input(CELL_T_GRAD).template data<T>();
    const auto* H_diff = Input(HIDDEN_T_GRAD).template data<T>();
    const auto* seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
    Output(HIDDEN_T_M_1_GRAD)->ResizeLike(Input(HIDDEN_T_M_1));
    auto* H_prev_diff = Output(HIDDEN_T_M_1_GRAD)->template mutable_data<T>();
    Output(CELL_T_M_1_GRAD)->ResizeLike(Input(CELL_T_M_1));
    auto* C_prev_diff = Output(CELL_T_M_1_GRAD)->template mutable_data<T>();
    Output(GATES_GRAD)->ResizeLike(Input(GATES));
    auto* X_diff = Output(GATES_GRAD)->template mutable_data<T>();

    detail::LSTMUnitGradient<T, Context>(
        N,
        D,
        t,
        C_prev,
        X,
        seqLengths,
        C,
        H,
        C_diff,
        H_diff,
        drop_states_,
        H_prev_diff,
        C_prev_diff,
        X_diff,
        forget_bias_,
        &context_);
    return true;
  }

 protected:
  INPUT_TAGS(
      HIDDEN_T_M_1,
      CELL_T_M_1,
      GATES,
      SEQ_LENGTHS,
      TIMESTEP,
      HIDDEN_T,
      CELL_T,
      HIDDEN_T_GRAD,
      CELL_T_GRAD, );
  OUTPUT_TAGS(HIDDEN_T_M_1_GRAD, CELL_T_M_1_GRAD, GATES_GRAD);

  T forget_bias_;

 private:
  bool drop_states_;
};
} // namespace caffe2

#endif // CAFFE2_OPERATORS_LSTM_UNIT_OP_H_
