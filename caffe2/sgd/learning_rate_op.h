#ifndef CAFFE2_SGD_LEARNING_RATE_OP_H_
#define CAFFE2_SGD_LEARNING_RATE_OP_H_

#include <cfloat>
#include <cmath>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/sgd/learning_rate_functors.h"

namespace caffe2 {

template <typename T, class Context>
class LearningRateOp final : public Operator<Context> {
 public:
  LearningRateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        functor_(nullptr),
        base_lr_(OperatorBase::template GetSingleArgument<float>(
            "base_lr",
            FLT_MAX)) {
    CAFFE_ENFORCE_NE(base_lr_, FLT_MAX, "Base learning rate must be set.");
    const string policy = OperatorBase::GetSingleArgument<string>("policy", "");
    CAFFE_ENFORCE(policy.size(), "Must specify a learning rate policy.");
    if (policy == "fixed") {
      functor_.reset(new FixedLearningRate<T>());
    } else if (policy == "alter") {
      bool active_first =
          OperatorBase::template GetSingleArgument<bool>("active_first", true);
      int64_t active_period = OperatorBase::template GetSingleArgument<int64_t>(
          "active_period", -1);
      int64_t inactive_period =
          OperatorBase::template GetSingleArgument<int64_t>(
              "inactive_period", -1);
      DCHECK_GE(active_period, 0);
      DCHECK_GE(inactive_period, 0);
      functor_.reset(new AlternateLearningRate<T>(
          active_period, inactive_period, active_first));
    } else if (policy == "hill") {
      int64_t num_iter =
          OperatorBase::template GetSingleArgument<int>("num_iter", 0);
      DCHECK_GT(num_iter, 0);
      T start_multiplier = OperatorBase::template GetSingleArgument<float>(
          "start_multiplier", 0.);
      DCHECK_GE(start_multiplier, 0); // start_multiplier in range [0, 1]
      DCHECK_LE(start_multiplier, 1);
      T gamma = OperatorBase::template GetSingleArgument<float>("gamma", 0);
      DCHECK_GT(gamma, 0);
      T power = OperatorBase::template GetSingleArgument<float>("power", 0);
      DCHECK_GT(power, 0);
      T end_multiplier =
          OperatorBase::template GetSingleArgument<float>("end_multiplier", 0);
      DCHECK_GE(end_multiplier, 0); // end_multiplier in range [0, 1]
      DCHECK_LE(end_multiplier, 1);
      functor_.reset(new HillLearningRate<T>(
          num_iter, start_multiplier, gamma, power, end_multiplier));
    } else if (policy == "step") {
      int stepsize =
          OperatorBase::template GetSingleArgument<int>("stepsize", 0);
      T gamma = OperatorBase::template GetSingleArgument<float>("gamma", 0);
      DCHECK_GT(stepsize, 0);
      DCHECK_GT(gamma, 0);
      functor_.reset(new StepLearningRate<T>(stepsize, gamma));
    } else if (policy == "exp") {
      T gamma = OperatorBase::template GetSingleArgument<float>("gamma", 0);
      DCHECK_GT(gamma, 0);
      functor_.reset(new ExpLearningRate<T>(gamma));
    } else if (policy == "inv") {
      T gamma = OperatorBase::template GetSingleArgument<float>("gamma", 0);
      T power = OperatorBase::template GetSingleArgument<float>("power", 0);
      DCHECK_GT(gamma, 0);
      DCHECK_GT(power, 0);
      functor_.reset(new InvLearningRate<T>(gamma, power));
    } else if (policy == "poly") {
      int max_iter = OperatorBase::template GetSingleArgument<int>("max_iter", -1);
      T power = OperatorBase::template GetSingleArgument<float>("power", 0);
      DCHECK_GT(power, 0);
      functor_.reset(new PolyLearningRate<T>(power, max_iter));
    } else if (policy == "linearWarmup") {
      T start_multiplier = OperatorBase::template GetSingleArgument<float>(
          "start_multiplier", 0.);
      int num_iter =
          OperatorBase::template GetSingleArgument<int>("num_iter", 0);
      DCHECK_GT(start_multiplier, 0);
      functor_.reset(
          new LinearWarmupLearningRate<T>(start_multiplier, num_iter));
    } else if (policy == "constantWarmup") {
      T multiplier =
          OperatorBase::template GetSingleArgument<float>("multiplier", 0.5);
      int num_iter =
          OperatorBase::template GetSingleArgument<int>("num_iter", 0);
      DCHECK_GT(multiplier, 0);
      functor_.reset(new ConstantWarmupLearningRate<T>(multiplier, num_iter));
    } else {
      LOG(FATAL) << "Unknown learning rate policy: " << policy;
    }
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    int64_t iter =
        OperatorBase::Input<TensorCPU>(0).template data<int64_t>()[0];
    T learning_rate = base_lr_ * (*functor_)(iter);
    // Write to output.
    auto* output = Output(0);
    output->Resize(vector<TIndex>());
    context_.template Copy<T, CPUContext, Context>(
        1, &learning_rate, Output(0)->template mutable_data<T>());
    return true;
  }

 private:
  unique_ptr<LearningRateFunctor<T> > functor_;
  T base_lr_;

};

}  // namespace caffe2

#endif  // CAFFE2_SGD_LEARNING_RATE_OP_H_
