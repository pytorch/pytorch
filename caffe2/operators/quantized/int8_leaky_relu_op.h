#ifndef CAFFE2_OPERATORS_INT8_LEAKY_RELU_OP_H_
#define CAFFE2_OPERATORS_INT8_LEAKY_RELU_OP_H_

#include <qnnpack.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8LeakyReluOp final : public Operator<CPUContext> {
 public:
  explicit Int8LeakyReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    const float alpha = this->template GetSingleArgument<float>("alpha", 0.01);
    CAFFE_ENFORCE_GT(alpha, 0.0);
    CAFFE_ENFORCE_LT(alpha, 1.0);
    this->alpha_ = alpha;
#if !defined(FBCODE_CAFFE2) && defined(USE_INTERNAL_PTHREADPOOL_IMPL)
    this->ws_ = ws;
#endif
  }

  ~Int8LeakyReluOp() override {
    if (this->qnnpackOperator_ != nullptr) {
      qnnp_delete_operator(this->qnnpackOperator_);
      this->qnnpackOperator_ = nullptr;
    }
  }

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    const int32_t Y_zero_point =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    TORCH_CHECK_GE(Y_zero_point, std::numeric_limits<uint8_t>::min());
    TORCH_CHECK_LE(Y_zero_point, std::numeric_limits<uint8_t>::max());

    /*
     * Record quantization parameters for the input, because if the op is
     * in-place, we may overwrite these parameters later, when we set
     * quantization parameters for output tensor.
     */
    const uint8_t X_zero_point = X.zero_point;
    const float X_scale = X.scale;

    Y->scale = Y_scale;
    Y->zero_point = Y_zero_point;
    Y->t.ResizeLike(X.t);

    initQNNPACK();

    if (this->qnnpackOperator_ == nullptr) {
      const qnnp_status createStatus = qnnp_create_leaky_relu_nc_q8(
          1 /* channels */,
          this->alpha_,
          static_cast<uint8_t>(X_zero_point),
          X_scale,
          static_cast<uint8_t>(Y_zero_point),
          Y_scale,
          0 /* output min */,
          255 /* output max */,
          0 /* flags */,
          &qnnpackOperator_);
      CAFFE_ENFORCE(
          createStatus == qnnp_status_success,
          "failed to create QNNPACK Leaky ReLU operator");
      CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
    }

    const qnnp_status setupStatus = qnnp_setup_leaky_relu_nc_q8(
        this->qnnpackOperator_,
        X.t.numel() /* batch size */,
        X.t.template data<uint8_t>(),
        1 /* X stride */,
        Y->t.template mutable_data<uint8_t>(),
        1 /* Y stride */);
    CAFFE_ENFORCE(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK Leaky ReLU operator");

#if defined(FBCODE_CAFFE2) || !defined(USE_INTERNAL_PTHREADPOOL_IMPL)
    const qnnp_status runStatus =
        qnnp_run_operator(this->qnnpackOperator_, nullptr /* thread pool */);
#else
    pthreadpool_t threadpool =
        reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());
    const qnnp_status runStatus =
        qnnp_run_operator(this->qnnpackOperator_, threadpool);
#endif
    CAFFE_ENFORCE(
        runStatus == qnnp_status_success,
        "failed to run QNNPACK Leaky ReLU operator");

    return true;
  }

 private:
  float alpha_;
#if !defined(FBCODE_CAFFE2) && defined(USE_INTERNAL_PTHREADPOOL_IMPL)
  Workspace* ws_;
#endif
  // QNNPACK Leaky ReLU operator
  qnnp_operator_t qnnpackOperator_{nullptr};
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_LEAKY_RELU_OP_H_
