#ifndef CAFFE2_OPERATORS_INT8_CONV_TRANSPOSE_OP_H_
#define CAFFE2_OPERATORS_INT8_CONV_TRANSPOSE_OP_H_

#include <qnnpack.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8ConvTransposeOp final : public ConvTransposeUnpoolBase<CPUContext> {
 public:
  USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(CPUContext);
  template <class... Args>
  explicit Int8ConvTransposeOp(Args&&... args)
      : ConvTransposeUnpoolBase(std::forward<Args>(args)...) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NHWC,
        "Int8ConvTransposeOp only supports NHWC order");
    createSharedBuffer<CPUContext>(ws_);
  }

  ~Int8ConvTransposeOp() {
    if (this->qnnpackObject_ != nullptr) {
      qnnp_delete_operator(this->qnnpackObject_);
      this->qnnpackObject_ = nullptr;
    }
  }

  bool RunOnDeviceWithOrderNHWC() override {
    CAFFE_ENFORCE_EQ(Inputs().size(), 3);
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    const auto& W = Inputs()[1]->template Get<Int8TensorCPU>();
    const auto& B = Inputs()[2]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    const auto X_offset = -X.zero_point;
    const auto W_offset = -W.zero_point;
    const int32_t Y_offset =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    double Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    Y->scale = Y_scale;
    Y->zero_point = Y_offset;

    const auto N = X.t.size(0);
    const auto IH = X.t.size(1);
    const auto IW = X.t.size(2);
    const auto IC = X.t.size(3);

    CHECK_EQ(IC, W.t.size(0));
    const auto KH = W.t.size(1);
    const auto KW = W.t.size(2);
    const auto OC = W.t.size(3);

    auto sizes = ConvTransposeUnpoolBase<CPUContext>::GetOutputSize(X.t, OC);
    ReinitializeTensor(&(Y->t), sizes, at::dtype<uint8_t>().device(CPU));
    CHECK_EQ(OC, Y->t.size(3));

    runWithSharedBuffer<CPUContext>(ws_, [&](Tensor* buffer) {
      initQNNPACK();

      pthreadpool_t threadpool =
          reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());

      if (this->qnnpackObject_ == nullptr) {
        const qnnp_status createStatus = qnnp_create_deconvolution2d_nhwc_q8(
            pad_t(),
            pad_r(),
            pad_b(),
            pad_l(),
            adj_h(),
            adj_w(),
            KH,
            KW,
            stride_h(),
            stride_w(),
            1 /* dilation height */,
            1 /* dilation width */,
            1 /* groups */,
            IC,
            OC,
            X.zero_point,
            X.scale,
            W.zero_point,
            W.scale,
#ifndef _MSC_VER
            W.t.template data<uint8_t>(),
            B.t.template data<int32_t>(),
#else
            W.t.data<uint8_t>(),
            B.t.data<int32_t>(),
#endif
            Y->zero_point,
            Y->scale,
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max(),
            0 /* flags */,
            &this->qnnpackObject_);
        CAFFE_ENFORCE(
            createStatus == qnnp_status_success,
            "failed to create QNNPACK convolution object");
        CAFFE_ENFORCE(this->qnnpackObject_ != nullptr);
      }

      uint8_t* inputPtr = X.t.template mutable_data<uint8_t>();
      if (IC < 8) {
        buffer->Resize(std::vector<int64_t>{X.t.numel() + 8});
        inputPtr = buffer->template mutable_data<uint8_t>() + 8;
        memcpy(inputPtr, X.t.template data<uint8_t>(), X.t.numel());
      }

      if (lastBatchSize_ != static_cast<size_t>(X.t.size(0)) ||
          lastInputHeight_ != static_cast<size_t>(X.t.size(1)) ||
          lastInputWidth_ != static_cast<size_t>(X.t.size(2)) ||
          lastInputPointer_ != inputPtr ||
          lastOutputPointer_ != Y->t.template mutable_data<uint8_t>()) {
        const qnnp_status setupStatus = qnnp_setup_deconvolution2d_nhwc_q8(
            this->qnnpackObject_,
            X.t.size(0),
            X.t.size(1),
            X.t.size(2),
            inputPtr,
            X.t.size(3) /* input pixel stride */,
            Y->t.template mutable_data<uint8_t>(),
            Y->t.size(3) /* output pixel stride */,
            nullptr /* threadpool */);
        CAFFE_ENFORCE(
            setupStatus == qnnp_status_success,
            "failed to setup QNNPACK convolution object");

        lastBatchSize_ = static_cast<size_t>(X.t.size(0));
        lastInputHeight_ = static_cast<size_t>(X.t.size(1));
        lastInputWidth_ = static_cast<size_t>(X.t.size(2));
        lastInputPointer_ = inputPtr;
        lastOutputPointer_ = Y->t.template mutable_data<uint8_t>();
      }

#ifdef FBCODE_CAFFE2
      const qnnp_status runStatus =
          qnnp_run_operator(this->qnnpackObject_, nullptr /* thread pool */);
#else
      const qnnp_status runStatus =
          qnnp_run_operator(this->qnnpackObject_, threadpool);
#endif
      CAFFE_ENFORCE(
          runStatus == qnnp_status_success,
          "failed to run QNNPACK convolution");
    });
    return true;
  }

 private:
  // QNNPACK convolution object
  qnnp_operator_t qnnpackObject_{nullptr};
  // batch size in the previous call to RunOnDeviceWithOrderNHWC
  size_t lastBatchSize_{0};
  // input height in the previous call to RunOnDeviceWithOrderNHWC
  size_t lastInputHeight_{0};
  // input width in the previous call to RunOnDeviceWithOrderNHWC
  size_t lastInputWidth_{0};
  // input pointer in the previous call to RunOnDeviceWithOrderNHWC
  const void* lastInputPointer_{nullptr};
  // output pointer in the previous call to RunOnDeviceWithOrderNHWC
  void* lastOutputPointer_{nullptr};
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_CONV_TRANSPOSE_OP_H_
