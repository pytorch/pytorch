#ifndef CAFFE2_OPERATORS_GRU_UNIT_OP_H_
#define CAFFE2_OPERATORS_GRU_UNIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace detail {

template <typename T, typename Context>
void GRUUnit(
    int N,
    int D,
    int t,
    const T* H_prev,
    const T* X,
    const int32_t* seqLengths,
    bool drop_states,
    T* H);

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
    T* X_diff);

}; // namespace detail

template <typename T, typename Context>
class GRUUnitOp : public Operator<Context> {
 public:
  GRUUnitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        drop_states_(OperatorBase::template GetSingleArgument<bool>(
            "drop_states",
            false)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Extract N
    const auto N = Input(HIDDEN_T_M_1).dim(1);

    // Gates: 1xNxG
    const auto G = Input(GATES).dim(2);
    const auto D = Input(HIDDEN_T_M_1).dim(2);

    CAFFE_ENFORCE_EQ(3 * D, G);
    const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
    const auto* X = Input(GATES).template data<T>();
    CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).size(), N);
    const auto* seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
    const auto t = OperatorBase::Input<Tensor<CPUContext>>(TIMESTEP)
                       .template data<int32_t>()[0];
    Output(HIDDEN_T)->ResizeLike(Input(HIDDEN_T_M_1));
    auto* H = Output(HIDDEN_T)->template mutable_data<T>();

    detail::GRUUnit<T, Context>(
        N, D, t, H_prev, X, seqLengths, drop_states_, H);
    return true;
  }

 protected:
  INPUT_TAGS(HIDDEN_T_M_1, GATES, SEQ_LENGTHS, TIMESTEP);
  OUTPUT_TAGS(HIDDEN_T);

 private:
  bool drop_states_;
};

template <typename T, typename Context>
class GRUUnitGradientOp : public Operator<Context> {
 public:
  GRUUnitGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        drop_states_(OperatorBase::template GetSingleArgument<bool>(
            "drop_states",
            false)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Extract N
    const auto N = Input(HIDDEN_T_M_1).dim(1);

    // Gates: 1xNxG
    const auto G = Input(GATES).dim(2);
    const auto D = Input(HIDDEN_T_M_1).dim(2);

    CAFFE_ENFORCE_EQ(3 * D, G);
    const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
    const auto* X = Input(GATES).template data<T>();
    const auto t = OperatorBase::Input<Tensor<CPUContext>>(TIMESTEP)
                       .template data<int32_t>()[0];
    const auto* H = Input(HIDDEN_T).template data<T>();
    const auto* H_diff = Input(HIDDEN_T_GRAD).template data<T>();
    const auto* seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
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
        X_diff);
    return true;
  }

 protected:
  INPUT_TAGS(
      HIDDEN_T_M_1,
      GATES,
      SEQ_LENGTHS,
      TIMESTEP,
      HIDDEN_T,
      HIDDEN_T_GRAD, );
  OUTPUT_TAGS(HIDDEN_T_M_1_GRAD, GATES_GRAD);

 private:
  bool drop_states_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GRU_UNIT_OP_H_
