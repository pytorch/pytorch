#ifndef CAFFE2_OPERATORS_ACCUMULATE_OP_H_
#define CAFFE2_OPERATORS_ACCUMULATE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Accumulate operator accumulates the input tensor to the output tensor. If the
// output tensor already has the right size, we add to it; otherwise, we first
// initialize the output tensor to all zeros, and then do accumulation. Any
// further calls to the operator, given that no one else fiddles with the output
// in the interim, will do simple accumulations.
template <typename dtype, class DeviceContext>
class AccumulateOp final : public Operator<dtype, DeviceContext> {
 public:
  AccumulateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        kOne(static_cast<dtype>(1), &device_context_),
        gamma_(static_cast<dtype>(
            OperatorBase::template GetSingleArgument<float>("gamma", 1.0)),
            &device_context_) {}
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    if (output->dims() != input.dims()) {
      LOG(INFO) << "Reshaping and initializing output.";
      output->ReshapeLike(input);
      math::Set<dtype, DeviceContext>(
          output->size(), 0, output->mutable_data(), &device_context_);
    }
    math::Axpby<dtype, DeviceContext>(
        input.size(), kOne.data(), input.data(), gamma_.data(),
        output->mutable_data(), &device_context_);
    return true;
  }

 protected:
  Tensor<dtype, DeviceContext> kOne;
  Tensor<dtype, DeviceContext> gamma_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(AccumulateOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ACCUMULATE_OP_H_
