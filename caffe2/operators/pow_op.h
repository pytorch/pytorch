#ifndef CAFFE2_OPERATORS_POW_OP_H_
#define CAFFE2_OPERATORS_POW_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <
    typename InputTypes,
    class Context,
    class Functor,
    class TypeMap = SameTypeAsInput>
class PowOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit PowOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(bool, "broadcast", enable_broadcast_, 0),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW"),
        functor_() {
    if ((InputSize() == 1) && HasArgument("exponent")) { // UnaryElementwiseOp
      exponent_ = this->template GetSingleArgument<float>(
          "exponent", 0); // based on pow_ops.h
    } else if (InputSize() == 2) { // BinaryElementwiseOp
      // Figure out the correct axis to use.
      if (enable_broadcast_) {
        if (axis_ != -1) {
          // Get axis from an explicit axis argument.
          CAFFE_ENFORCE_EQ(
              axis_str_.size(),
              0U,
              "Args axis and axis_str cannot be used simultaneously.");
        } else if (axis_str_.size()) {
          // Get the axis index semantically.
          CAFFE_ENFORCE_EQ(
              axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
          size_t semantic_axis_ = order_.find(axis_str_);
          CAFFE_ENFORCE_NE(
              semantic_axis_,
              string::npos,
              "Unrecognizable axis string ",
              axis_str_,
              " from order string ",
              order_);
          axis_ = semantic_axis_;
        }
      } else {
        CAFFE_ENFORCE(
            axis_ == -1 && axis_str_.empty(),
            "Do not specify axis or axis_str if broadcast is not enabled.");
      }
    } else {
      CAFFE_THROW(
          "Only a tensor with an argument or two input tensors are supported as input to pow operator.");
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    if ((InputSize() == 1) && HasArgument("exponent")) { // UnaryElementwiseOp
      const auto& A = Input(0);

      auto* C =
          Output(0, A.sizes(), at::dtype<typename TypeMap::template type<T>>());
      const T* Adata = A.template data<T>();
      auto* Cdata =
          C->template mutable_data<typename TypeMap::template type<T>>();
      functor_.template Run<true, T, float, T>(
          A.numel(), Adata, NULL, exponent_, Cdata, &context_);
    } else if (InputSize() == 2) { // BinaryElementwiseOp
      const auto& A = Input(0);
      const auto& B = Input(1);
      CAFFE_ENFORCE(
          !IsInputOutputAlias(1, 0) || !enable_broadcast_,
          "In-place is allowed only with the first tensor when broadcasting");
      auto* C =
          Output(0, A.sizes(), at::dtype<typename TypeMap::template type<T>>());
      const T* Adata = A.template data<T>();
      const T* Bdata = B.template data<T>();
      auto* Cdata =
          C->template mutable_data<typename TypeMap::template type<T>>();
      if (!enable_broadcast_) {
        CAFFE_ENFORCE_EQ(
            A.sizes(),
            B.sizes(),
            "Dimension mismatch - did you forget to set broadcast=1?");
        functor_.template Run<false, T, T, T>(
            A.numel(), Adata, Bdata, 0, Cdata, &context_);
      } else if (B.numel() == 1) {
        functor_.template Run<true, T, T, T>(
            A.numel(), Adata, Bdata, 0, Cdata, &context_);
      } else {
        size_t pre, n, post;
        std::tie(pre, n, post) =
            elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
        if (post == 1) {
          functor_.template RunWithBroadcast<T, T, T>(
              Adata, Bdata, Cdata, pre, n, &context_);
        } else {
          functor_.template RunWithBroadcast2<T, T, T>(
              Adata, Bdata, Cdata, pre, n, post, &context_);
        }
      }
    } else {
      CAFFE_THROW(
          "Only a tensor with an argument or two input tensors are supported as input to pow operator.");
    }
    return true;
  }

 private:
  bool enable_broadcast_;
  int axis_;
  string axis_str_;
  string order_;
  float exponent_;
  Functor functor_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_POW_OP_H_
