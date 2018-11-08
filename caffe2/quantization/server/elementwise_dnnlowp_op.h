#ifndef CAFFE2_OPERATORS_ELEMENTWISE_DNNLOWP_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_DNNLOWP_OP_H_

#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp_op.h"
#include "caffe2/quantization/server/sigmoid.h"

namespace caffe2 {

template <typename T, class Functor>
class UnaryElementwiseWithArgsDNNLowPOp : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  UnaryElementwiseWithArgsDNNLowPOp(
    const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws), functor_() {
  }

  bool RunOnDevice() override {
    if (!arguments_parsed_) {
      dnnlowp::ParseDNNLowPOperatorArguments(this);
      dnnlowp::SetStaticQuantizationParams(
          this, 0, functor_.GetOutputQuantizationParams());
      arguments_parsed_ = true;
    }

    auto& input = OperatorBase::Input<int8::Int8TensorCPU>(0).t;
    auto& output = Outputs()[0]->template GetMutable<int8::Int8TensorCPU>()->t;
    output.ResizeLike(input);
    functor_(
        input.size(),
        input.template data<T>(),
        output.template mutable_data<T>());

    PropagateOutputTensorQuantizationParams(
        this, 0, functor_.GetOutputQuantizationParams());
    return true;
  }

 private:
  Functor functor_;
  bool arguments_parsed_{false};
};

template <typename T, typename FP32_OP>
class BinaryElementwiseDNNLowPOp : public DNNLowPOp<T, FP32_OP> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  BinaryElementwiseDNNLowPOp(const OperatorDef& operator_def, Workspace* ws)
    : DNNLowPOp<T, FP32_OP>(operator_def, ws),
      OP_SINGLE_ARG(bool, "broadcast", enable_broadcast_, 0),
      OP_SINGLE_ARG(int, "axis", axis_, -1),
      OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
      OP_SINGLE_ARG(string, "order", order_, "NCHW") {

    // Figure out the correct axis to use.
    if (enable_broadcast_) {
      if (axis_ != -1) {
        // Get axis from an explicit axis argument.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(),
            0,
            "Args axis and axis_str cannot be used simultaneously.");
      } else if (axis_str_.size()) {
        // Get the axis index semantically.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(), 1, "Unsupported axis string", axis_str_);
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
          axis_ == -1 && axis_str_.size() == 0,
          "Do not specify axis or axis_str if broadcast is not enabled.");
    }
  }

 protected:
  bool enable_broadcast_;
  int axis_;
  string axis_str_;
  string order_;

  dnnlowp::RequantizationParams requantization_params_;
}; // BinaryElementwiseDNNLowPOp

// For arithmetic operators, Eigen provides a good way to vectorize even
// when broadcasting.
#define DECLARE_EIGEN_FUNCTOR(name, eigen_op, input_type, output_type)       \
  struct Eigen##name##Functor {                                              \
    template <int b_is_scalar, typename T, typename R>                       \
    inline void Run(size_t n, const T* a, const T* b, R* out, CPUContext*) { \
      if (b_is_scalar) {                                                     \
        EigenVectorArrayMap<R>(out, n) =                                     \
            eigen_op((ConstEigenVectorArrayMap<T>(a, n)), (b[0]));           \
      } else {                                                               \
        EigenVectorArrayMap<R>(out, n) = eigen_op(                           \
            (ConstEigenVectorArrayMap<T>(a, n)),                             \
            (ConstEigenVectorArrayMap<T>(b, n)));                            \
      }                                                                      \
    }                                                                        \
    template <typename T, typename R>                                        \
    void RunWithBroadcast(                                                   \
        const T* a,                                                          \
        const T* b,                                                          \
        R* out,                                                              \
        size_t pre,                                                          \
        size_t n,                                                            \
        CPUContext*) {                                                       \
      EigenArrayMap<R>(out, n, pre) = eigen_op(                              \
          (ConstEigenArrayMap<T>(a, n, pre).colwise()),                      \
          (ConstEigenVectorArrayMap<T>(b, n)));                              \
    }                                                                        \
    template <typename T, typename R>                                        \
    void RunWithBroadcast2(                                                  \
        const T* a,                                                          \
        const T* b,                                                          \
        R* out,                                                              \
        size_t pre,                                                          \
        size_t n,                                                            \
        size_t post,                                                         \
        CPUContext*) {                                                       \
      for (int i = 0; i < pre; ++i) {                                        \
        EigenArrayMap<R>(out + i * n * post, post, n) = eigen_op(            \
            (ConstEigenArrayMap<T>(a + i * n * post, post, n).rowwise()),    \
            (Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>>(b, n)));   \
      }                                                                      \
    }                                                                        \
  };
} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_DNNLOWP_OP_H_
