#ifndef CAFFE2_OPERATORS_INT8_AVERAGE_POOL_OP_H_
#define CAFFE2_OPERATORS_INT8_AVERAGE_POOL_OP_H_

#include <qnnpack.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

template <Activation Ac>
class Int8AveragePoolOp final : public ConvPoolOpBase<CPUContext> {
 public:
  template <class... Args>
  explicit Int8AveragePoolOp(Args&&... args)
      : ConvPoolOpBase<CPUContext>(std::forward<Args>(args)...) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NHWC, "Int8 only supports NHWC order.");
  }

  ~Int8AveragePoolOp() override {
    if (this->qnnpackOperator_ != nullptr) {
      qnnp_delete_operator(this->qnnpackOperator_);
      this->qnnpackOperator_ = nullptr;
    }
    if (this->qnnpackGlobalOperator_ != nullptr) {
      qnnp_delete_operator(this->qnnpackGlobalOperator_);
      this->qnnpackGlobalOperator_ = nullptr;
    }
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    int32_t Y_zero_point =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    Y->scale = Y_scale;
    Y->zero_point = Y_zero_point;

    TORCH_CHECK_EQ(X.t.dim(), 4);
    const int channels = X.t.dim32(3);
    ConvPoolOpBase<CPUContext>::SetOutputSize(X.t, &(Y->t), channels);

    initQNNPACK();

    const bool anyPadding =
        pad_t() != 0 || pad_r() != 0 || pad_b() != 0 || pad_l() != 0;
    const bool anyStride = stride_h() > 1 || stride_w() > 1;
    const bool globalPooling = !anyPadding && !anyStride &&
        (X.t.dim32(1) == kernel_h() && X.t.dim32(2) == kernel_w());
    if (globalPooling) {
      if (this->qnnpackGlobalOperator_ == nullptr) {
        const qnnp_status createStatus =
            qnnp_create_global_average_pooling_nwc_q8(
                channels,
                X.zero_point,
                X.scale,
                Y->zero_point,
                Y->scale,
                activationLimits(Y->scale, Y->zero_point, Ac).first,
                activationLimits(Y->scale, Y->zero_point, Ac).second,
                0 /* flags */,
                &this->qnnpackGlobalOperator_);
        CAFFE_ENFORCE(
            createStatus == qnnp_status_success,
            "failed to create QNNPACK Global Average Pooling operator");
        CAFFE_ENFORCE(this->qnnpackGlobalOperator_ != nullptr);
      }

      const qnnp_status setupStatus = qnnp_setup_global_average_pooling_nwc_q8(
          this->qnnpackGlobalOperator_,
          X.t.dim32(0),
          X.t.dim32(1) * X.t.dim32(2),
          X.t.template data<uint8_t>(),
          channels,
          Y->t.template mutable_data<uint8_t>(),
          channels);
      CAFFE_ENFORCE(
          setupStatus == qnnp_status_success,
          "failed to setup QNNPACK Global Average Pooling operator");

#if defined(FBCODE_CAFFE2) || !defined(USE_INTERNAL_PTHREADPOOL_IMPL)
      const qnnp_status runStatus = qnnp_run_operator(
          this->qnnpackGlobalOperator_, nullptr /* thread pool */);
#else
      pthreadpool_t threadpool =
          reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());
      const qnnp_status runStatus =
          qnnp_run_operator(this->qnnpackGlobalOperator_, threadpool);
#endif
      CAFFE_ENFORCE(
          runStatus == qnnp_status_success,
          "failed to run QNNPACK Global Average Pooling operator");
    } else {
      if (this->qnnpackOperator_ == nullptr) {
        const qnnp_status createStatus = qnnp_create_average_pooling2d_nhwc_q8(
            pad_t(),
            pad_r(),
            pad_b(),
            pad_l(),
            kernel_h(),
            kernel_w(),
            stride_h(),
            stride_w(),
            channels,
            X.zero_point,
            X.scale,
            Y->zero_point,
            Y->scale,
            activationLimits(Y->scale, Y->zero_point, Ac).first,
            activationLimits(Y->scale, Y->zero_point, Ac).second,
            0 /* flags */,
            &this->qnnpackOperator_);
        CAFFE_ENFORCE(
            createStatus == qnnp_status_success,
            "failed to create QNNPACK Average Pooling operator");
        CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
      }

      const qnnp_status setupStatus = qnnp_setup_average_pooling2d_nhwc_q8(
          this->qnnpackOperator_,
          X.t.dim32(0),
          X.t.dim32(1),
          X.t.dim32(2),
          X.t.template data<uint8_t>(),
          channels,
          Y->t.template mutable_data<uint8_t>(),
          channels,
          nullptr /* thread pool */);
      CAFFE_ENFORCE(
          setupStatus == qnnp_status_success,
          "failed to setup QNNPACK Average Pooling operator");

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
          "failed to run QNNPACK Average Pooling operator");
    }

    return true;
  }

 private:
  // QNNPACK Average Pooling operator
  qnnp_operator_t qnnpackOperator_{nullptr};
  // QNNPACK Global Average Pooling operator
  qnnp_operator_t qnnpackGlobalOperator_{nullptr};
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_AVERAGE_POOL_OP_H_
