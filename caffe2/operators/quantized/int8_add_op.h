#ifndef CAFFE2_OPERATORS_INT8_ADD_OP_H_
#define CAFFE2_OPERATORS_INT8_ADD_OP_H_

#include <qnnpack.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

template <Activation Ac>
class Int8AddOp final : public Operator<CPUContext> {
 public:
  explicit Int8AddOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws), ws_(ws) {}

  ~Int8AddOp() {
    if (this->qnnpackOperator_ != nullptr) {
      qnnp_delete_operator(this->qnnpackOperator_);
      this->qnnpackOperator_ = nullptr;
    }
  }

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(Inputs().size(), 2);
    const auto& A = Inputs()[0]->template Get<Int8TensorCPU>();
    const auto& B = Inputs()[1]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();

    CAFFE_ENFORCE_EQ(
        A.t.sizes(),
        B.t.sizes(),
        "inputs must have the same shape (broadcast semantics is not supported)");

    /*
     * Record quantization parameters for A and B inputs, because if the op is
     * in-place, we may overwrite these parameters later, when we set
     * quantization parameters for Y tensor.
     */
    const uint8_t A_zero_point = A.zero_point;
    const uint8_t B_zero_point = B.zero_point;
    const float A_scale = A.scale;
    const float B_scale = B.scale;

    const int32_t Y_zero_point =
      this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale =
      this->template GetSingleArgument<float>("Y_scale", 1);
    Y->t.ResizeLike(A.t);
    Y->zero_point = Y_zero_point;
    Y->scale = Y_scale;

    initQNNPACK();

    pthreadpool_t threadpool =
        reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());

    if (this->qnnpackOperator_ == nullptr) {
      const qnnp_status createStatus = qnnp_create_add_nc_q8(
        1 /* channels */,
        A_zero_point, A_scale,
        B_zero_point, B_scale,
        static_cast<uint8_t>(Y_zero_point), Y_scale,
        activationLimits(Y_scale, Y_zero_point, Ac).first,
        activationLimits(Y_scale, Y_zero_point, Ac).second,
        0 /* flags */,
        &qnnpackOperator_);
      CAFFE_ENFORCE(
          createStatus == qnnp_status_success,
          "failed to create QNNPACK add operator");
      CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
    }

    const qnnp_status setupStatus = qnnp_setup_add_nc_q8(
        this->qnnpackOperator_,
        A.t.numel() /* batch size */,
        A.t.template data<uint8_t>(),
        1 /* A stride */,
        B.t.template data<uint8_t>(),
        1 /* B stride */,
        Y->t.template mutable_data<uint8_t>(),
        1 /* Y stride */);
    CAFFE_ENFORCE(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK add operator");

#ifdef FBCODE_CAFFE2
    const qnnp_status runStatus =
        qnnp_run_operator(this->qnnpackOperator_, nullptr /* thread pool */);
#else
    const qnnp_status runStatus =
        qnnp_run_operator(this->qnnpackOperator_, threadpool);
#endif
    CAFFE_ENFORCE(
        runStatus == qnnp_status_success,
        "failed to run QNNPACK add operator");

    return true;
  }

 private:
  Workspace* ws_;
  // QNNPACK add operator
  qnnp_operator_t qnnpackOperator_{nullptr};
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_ADD_OP_H_
