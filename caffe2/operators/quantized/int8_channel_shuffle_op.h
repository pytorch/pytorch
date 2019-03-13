#ifndef CAFFE2_OPERATORS_INT8_CHANNEL_SHUFFLE_OP_H_
#define CAFFE2_OPERATORS_INT8_CHANNEL_SHUFFLE_OP_H_

#include <qnnpack.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8ChannelShuffleOp final : public ConvPoolOpBase<CPUContext> {
 public:
  explicit Int8ChannelShuffleOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws), ws_(ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NHWC,
        "Int8ChannelShuffleOp only supports NHWC order");
  }

  ~Int8ChannelShuffleOp() {
    if (this->qnnpackOperator_ != nullptr) {
      qnnp_delete_operator(this->qnnpackOperator_);
      this->qnnpackOperator_ = nullptr;
    }
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->t.ResizeLike(X.t);
    Y->scale = X.scale;
    Y->zero_point = X.zero_point;
    const int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale = this->template GetSingleArgument<float>("Y_scale", 1.0f);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);
    CHECK_GE(X.zero_point, std::numeric_limits<uint8_t>::min());
    CHECK_LE(X.zero_point, std::numeric_limits<uint8_t>::max());

    const auto C = X.t.dim32(3);
    const auto G = this->group_;
    CAFFE_ENFORCE(C % G == 0, "");
    const auto B = X.t.numel() / C;

    initQNNPACK();

    if (this->qnnpackOperator_ == nullptr) {
      const qnnp_status createStatus = qnnp_create_channel_shuffle_nc_x8(
        G /* groups */,
        C / G /* group channels */,
        0 /* flags */,
        &this->qnnpackOperator_);
      CAFFE_ENFORCE(
          createStatus == qnnp_status_success,
          "failed to create QNNPACK channel shuffle operator");
      CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
    }

    const qnnp_status setupStatus = qnnp_setup_channel_shuffle_nc_x8(
        this->qnnpackOperator_,
        X.t.numel() / C /* batch size */,
        X.t.template data<uint8_t>(),
        C /* X stride */,
        Y->t.template mutable_data<uint8_t>(),
        C /* Y stride */);
    CAFFE_ENFORCE(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK channel shuffle operator");

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
        "failed to run QNNPACK channel shuffle operator");

    return true;
  }

 private:
  Workspace* ws_;
  // QNNPACK channel shuffle operator
  qnnp_operator_t qnnpackOperator_{nullptr};
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_CHANNEL_SHUFFLE_OP_H_
