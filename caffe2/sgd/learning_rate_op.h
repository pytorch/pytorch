#ifndef CAFFE2_SGD_LEARNING_RATE_OP_H_
#define CAFFE2_SGD_LEARNING_RATE_OP_H_

#include <cfloat>
#include <cmath>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/sgd/learning_rate_functors.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class LearningRateOp final : public Operator<dtype, DeviceContext> {
 public:
  LearningRateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws), functor_(nullptr),
        base_lr_(
            OperatorBase::template GetSingleArgument<float>("base_lr", FLT_MAX)) {
    CHECK_NE(base_lr_, FLT_MAX) << "Base learning rate must be set.";
    const string policy = OperatorBase::GetSingleArgument<string>("policy", "");
    CHECK(policy.size()) << "Must specify a learning rate policy.";
    if (policy == "fixed") {
      functor_.reset(new FixedLearningRate<dtype>());
    } else if (policy == "step") {
      int stepsize =
          OperatorBase::template GetSingleArgument<int>("stepsize", 0);
      dtype gamma = OperatorBase::template GetSingleArgument<float>("gamma", 0);
      DCHECK_GT(stepsize, 0);
      DCHECK_GT(gamma, 0);
      functor_.reset(new StepLearningRate<dtype>(stepsize, gamma));
    } else if (policy == "exp") {
      dtype gamma = OperatorBase::template GetSingleArgument<float>("gamma", 0);
      DCHECK_GT(gamma, 0);
      functor_.reset(new ExpLearningRate<dtype>(gamma));
    } else if (policy == "inv") {
      dtype gamma = OperatorBase::template GetSingleArgument<float>("gamma", 0);
      dtype power = OperatorBase::template GetSingleArgument<float>("power", 0);
      DCHECK_GT(gamma, 0);
      DCHECK_GT(power, 0);
      functor_.reset(new InvLearningRate<dtype>(gamma, power));
    } else {
      LOG(FATAL) << "Unknown learning rate policy: " << policy;
    }
  }
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override {
    int iter = OperatorBase::Input<int>(0);
    dtype learning_rate = base_lr_ * (*functor_)(iter);
    // Write to output.
    auto* output = Output(0);
    output->Reshape(std::vector<int>{1});
    device_context_.template Copy<dtype, DeviceContext, CPUContext>(
        Output(0)->mutable_data(), &learning_rate, 1);
    return true;
  }

 private:
  unique_ptr<LearningRateFunctor<dtype> > functor_;
  dtype base_lr_;

  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(LearningRateOp);
};

}  // namespace caffe2

#endif  // CAFFE2_SGD_LEARNING_RATE_OP_H_
