#ifndef CAFFE2_OPERATORS_INT8_RELU_OP_H_
#define CAFFE2_OPERATORS_INT8_RELU_OP_H_

#include <qnnpack.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8ReluOp final : public Operator<CPUContext> {
 public:
  explicit Int8ReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws), ws_(ws) {}

  ~Int8ReluOp() {
    if (this->qnnpackOperator_ != nullptr) {
      qnnp_delete_operator(this->qnnpackOperator_);
      this->qnnpackOperator_ = nullptr;
    }
  }

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->t.ResizeLike(X.t);
    Y->scale = X.scale;
    Y->zero_point = X.zero_point;
    CHECK_GE(X.zero_point, std::numeric_limits<uint8_t>::min());
    CHECK_LE(X.zero_point, std::numeric_limits<uint8_t>::max());
    const int32_t Y_offset =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale =
        this->template GetSingleArgument<float>("Y_scale", 1.0f);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);

    initQNNPACK();

    if (this->qnnpackOperator_ == nullptr) {
      const qnnp_status createStatus = qnnp_create_clamp_nc_u8(
          1 /* channels */,
          X.zero_point /* output min */,
          255 /* output max */,
          0 /* flags */,
          &qnnpackOperator_);
      CAFFE_ENFORCE(
          createStatus == qnnp_status_success,
          "failed to create QNNPACK Clamp operator");
      CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
    }

    const qnnp_status setupStatus = qnnp_setup_clamp_nc_u8(
        this->qnnpackOperator_,
        X.t.numel() /* batch size */,
        X.t.template data<uint8_t>(),
        1 /* X stride */,
        Y->t.template mutable_data<uint8_t>(),
        1 /* Y stride */);
    CAFFE_ENFORCE(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK Clamp operator");

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
        "failed to run QNNPACK Clamp operator");

    return true;
  }

 private:
  Workspace* ws_;
  // QNNPACK Clamp operator
  qnnp_operator_t qnnpackOperator_{nullptr};
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_RELU_OP_H_
