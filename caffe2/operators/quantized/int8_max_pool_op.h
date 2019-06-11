#ifndef CAFFE2_OPERATORS_INT8_MAX_POOL_OP_H_
#define CAFFE2_OPERATORS_INT8_MAX_POOL_OP_H_

#include <qnnpack.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

template <Activation Ac>
class Int8MaxPoolOp final : public ConvPoolOpBase<CPUContext> {
 public:
  template <class... Args>
  explicit Int8MaxPoolOp(Args&&... args)
      : ConvPoolOpBase<CPUContext>(std::forward<Args>(args)...) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NHWC, "Int8 only supports NHWC order.");
  }

  ~Int8MaxPoolOp() {
    if (this->qnnpackOperator_ != nullptr) {
      qnnp_delete_operator(this->qnnpackOperator_);
      this->qnnpackOperator_ = nullptr;
    }
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->scale = X.scale;
    Y->zero_point = X.zero_point;
    const int32_t Y_zero_point =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_zero_point, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);

    CHECK_EQ(X.t.dim(), 4);
    const int channels = X.t.dim32(3);
    ConvPoolOpBase<CPUContext>::SetOutputSize(X.t, &(Y->t), channels);

    initQNNPACK();

    if (this->qnnpackOperator_ == nullptr) {
      const qnnp_status createStatus = qnnp_create_max_pooling2d_nhwc_u8(
        pad_t(), pad_r(), pad_b(), pad_l(),
        kernel_h(), kernel_w(),
        stride_h(), stride_w(),
        1 /* dilation height */, 1 /* dilation width */,
        channels,
        activationLimits(Y->scale, Y->zero_point, Ac).first,
        activationLimits(Y->scale, Y->zero_point, Ac).second,
        0 /* flags */,
        &this->qnnpackOperator_);
      CAFFE_ENFORCE(
          createStatus == qnnp_status_success,
          "failed to create QNNPACK Max Pooling operator");
      CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
    }

    const qnnp_status setupStatus = qnnp_setup_max_pooling2d_nhwc_u8(
        this->qnnpackOperator_,
        X.t.dim32(0), X.t.dim32(1), X.t.dim32(2),
        X.t.template data<uint8_t>(), channels,
        Y->t.template mutable_data<uint8_t>(), channels,
        nullptr /* thread pool */);
    CAFFE_ENFORCE(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK Max Pooling operator");

#ifdef FBCODE_CAFFE2
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
        "failed to run QNNPACK Max Pooling operator");
    return true;
  }

 private:
  // QNNPACK Max Pooling operator
  qnnp_operator_t qnnpackOperator_{nullptr};
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_MAX_POOL_OP_H_
