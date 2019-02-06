#ifndef CAFFE2_OPERATORS_INT8_FC_OP_H_
#define CAFFE2_OPERATORS_INT8_FC_OP_H_

#include <qnnpack.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8FCOp final : public Operator<CPUContext> {
 public:
  Int8FCOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        ws_(ws) {
    createSharedBuffer<CPUContext>(ws_);
  }

  ~Int8FCOp() {
    if (this->qnnpackObject_ != nullptr) {
      qnnp_delete_operator(this->qnnpackObject_);
      this->qnnpackObject_ = nullptr;
    }
  }

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->Get<Int8TensorCPU>();
    const auto& W = Inputs()[1]->Get<Int8TensorCPU>();
    const auto& B = Inputs()[2]->Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    Y->scale = Y_scale;
    Y->zero_point = Y_offset;
    // (NxHxW)xC == MxK x (NxK) -> MxN
    const auto K = X.t.size_from_dim(1);
    const auto N = W.t.size(0);
    CHECK_EQ(K, W.t.size(1));
    CHECK_EQ(N, B.t.numel());
    const auto M = X.t.numel() / K;
    ReinitializeTensor(&Y->t, {M, N}, at::dtype<uint8_t>().device(CPU));

    runWithSharedBuffer<CPUContext>(ws_, [&](Tensor* buffer) {
      initQNNPACK();

      pthreadpool_t threadpool =
          reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());

      if (this->qnnpackObject_ == nullptr) {
        const qnnp_status createStatus = qnnp_create_fully_connected_nc_q8(
            K,
            N,
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
            "failed to create QNNPACK fully connected operator");
        CAFFE_ENFORCE(this->qnnpackObject_ != nullptr);
      }

      uint8_t* inputPtr = X.t.template mutable_data<uint8_t>();
      if (K < 8) {
        buffer->Resize(std::vector<int64_t>{X.t.numel() + 8});
        inputPtr = buffer->template mutable_data<uint8_t>() + 8;
        memcpy(inputPtr, X.t.template data<uint8_t>(), X.t.numel());
      }

      if (lastBatchSize_ != static_cast<size_t>(M) ||
          lastInputPointer_ != inputPtr ||
          lastOutputPointer_ != Y->t.template mutable_data<uint8_t>()) {
        const qnnp_status setupStatus = qnnp_setup_fully_connected_nc_q8(
            this->qnnpackObject_,
            M,
            inputPtr,
            K /* input stride */,
            Y->t.template mutable_data<uint8_t>(),
            N /* output stride */);
        CAFFE_ENFORCE(
            setupStatus == qnnp_status_success,
            "failed to setup QNNPACK fully connected operator");

        lastBatchSize_ = static_cast<size_t>(M);
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
          runStatus == qnnp_status_success, "failed to run QNNPACK operator");
    });
    return true;
  }

 private:
  Workspace* ws_;
  // QNNPACK convolution object
  qnnp_operator_t qnnpackObject_{nullptr};
  // batch size in the previous call to RunOnDeviceWithOrderNHWC
  size_t lastBatchSize_{0};
  // input pointer in the previous call to RunOnDeviceWithOrderNHWC
  const void* lastInputPointer_{nullptr};
  // output pointer in the previous call to RunOnDeviceWithOrderNHWC
  void* lastOutputPointer_{nullptr};
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_FC_OP_H_
