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
        base_lr_(this->template GetSingleArgument<float>(
            "base_lr",
            FLT_MAX)) {
    CAFFE_ENFORCE_NE(base_lr_, FLT_MAX, "Base learning rate must be set.");
    const string policy = this->template GetSingleArgument<string>("policy", "");
    CAFFE_ENFORCE(policy.size(), "Must specify a learning rate policy.");
    functor_.reset(createLearningRateFunctor(policy));
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    int64_t iter =
        OperatorBase::Input<Tensor>(0, CPU).template data<int64_t>()[0];
    T learning_rate = cur_base_lr_ * (*functor_)(iter);
    // Write to output.
    auto* output = Output(0);
    output->Resize(vector<int64_t>());
    context_.template CopyFromCPU<T>(
        1, &learning_rate, Output(0)->template mutable_data<T>());
    return true;
  }

 private:
  unique_ptr<LearningRateFunctor<T>> functor_;
  T base_lr_;
  T base_lr_scale_;
  T cur_base_lr_;

  LearningRateFunctor<T>* createLearningRateFunctor(
      const string& policy,
      const string& arg_prefix = "") {
    if (policy != "composite") {
      base_lr_scale_ =
          this->template GetSingleArgument<float>(arg_prefix + "lr_scale", 1.0);
      cur_base_lr_ = base_lr_scale_ * base_lr_;
    }
    if (policy == "fixed") {
      return new FixedLearningRate<T>();
    } else if (policy == "alter") {
      bool active_first = this->template GetSingleArgument<bool>(
          arg_prefix + "active_first", true);
      int64_t active_period = this->template GetSingleArgument<int64_t>(
          arg_prefix + "active_period", -1);
      int64_t inactive_period =
          this->template GetSingleArgument<int64_t>(
              arg_prefix + "inactive_period", -1);
      DCHECK_GE(active_period, 0);
      DCHECK_GE(inactive_period, 0);
      return new AlternateLearningRate<T>(
          active_period, inactive_period, active_first);
    } else if (policy == "hill") {
      int64_t num_iter = this->template GetSingleArgument<int>(
          arg_prefix + "num_iter", 0);
      DCHECK_GT(num_iter, 0);
      T start_multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "start_multiplier", 0.);
      DCHECK_GE(start_multiplier, 0); // start_multiplier in range [0, 1]
      DCHECK_LE(start_multiplier, 1);
      T gamma = this->template GetSingleArgument<float>(
          arg_prefix + "gamma", 0);
      DCHECK_GT(gamma, 0);
      T power = this->template GetSingleArgument<float>(
          arg_prefix + "power", 0);
      DCHECK_GT(power, 0);
      T end_multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "end_multiplier", 0);
      DCHECK_GE(end_multiplier, 0); // end_multiplier in range [0, 1]
      DCHECK_LE(end_multiplier, 1);
      return new HillLearningRate<T>(
          num_iter, start_multiplier, gamma, power, end_multiplier);
    } else if (policy == "step") {
      int stepsize = this->template GetSingleArgument<int>(
          arg_prefix + "stepsize", 0);
      T gamma = this->template GetSingleArgument<float>(
          arg_prefix + "gamma", 0);
      DCHECK_GT(stepsize, 0);
      DCHECK_GT(gamma, 0);
      return new StepLearningRate<T>(stepsize, gamma);
    } else if (policy == "exp") {
      T gamma = this->template GetSingleArgument<float>(
          arg_prefix + "gamma", 0);
      DCHECK_GT(gamma, 0);
      return new ExpLearningRate<T>(gamma);
    } else if (policy == "inv") {
      T gamma = this->template GetSingleArgument<float>(
          arg_prefix + "gamma", 0);
      T power = this->template GetSingleArgument<float>(
          arg_prefix + "power", 0);
      DCHECK_GT(gamma, 0);
      DCHECK_GT(power, 0);
      return new InvLearningRate<T>(gamma, power);
    } else if (policy == "poly") {
      int max_iter = this->template GetSingleArgument<int>(
          arg_prefix + "max_iter", -1);
      T power = this->template GetSingleArgument<float>(
          arg_prefix + "power", 0);
      DCHECK_GT(power, 0);
      return new PolyLearningRate<T>(power, max_iter);
    } else if (policy == "linearWarmup") {
      T start_multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "start_multiplier", 0.);
      int num_iter = this->template GetSingleArgument<int>(
          arg_prefix + "num_iter", 0);
      DCHECK_GE(start_multiplier, 0);
      return new LinearWarmupLearningRate<T>(start_multiplier, num_iter);
    } else if (policy == "constantWarmup") {
      T multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "multiplier", 0.5);
      int num_iter = this->template GetSingleArgument<int>(
          arg_prefix + "num_iter", 0);
      DCHECK_GT(multiplier, 0);
      return new ConstantWarmupLearningRate<T>(multiplier, num_iter);
    } else if (policy == "composite") {
      std::vector<int> sub_policy_num_iters =
          this->template GetRepeatedArgument<int>(
              "sub_policy_num_iters");
      std::list<CompositeLearningRateItem<T>> sub_policies;
      CAFFE_ENFORCE_GT(
          sub_policy_num_iters.size(),
          0,
          "Must specify at least one sub learning rate policy.");
      for (int i = 0; i < sub_policy_num_iters.size(); ++i) {
        CAFFE_ENFORCE_GT(
            sub_policy_num_iters[i],
            0,
            "The number of iterations for sub learning rate policy should be positive.");
        std::stringstream sub_policy_arg_prefix;
        sub_policy_arg_prefix << "sub_policy_" << i << "_";
        const string sub_policy_arg_prefix_str = sub_policy_arg_prefix.str();
        const string sub_policy = this->template GetSingleArgument<string>(
            sub_policy_arg_prefix_str + "policy", "");
        if (sub_policy == "composite") {
          CAFFE_THROW(
              "Defining composite LR policy as a subpolicy of composite LR "
              "policy is not allowed.");
        }
        sub_policies.push_back(CompositeLearningRateItem<T>(
            sub_policy_num_iters[i],
            createLearningRateFunctor(sub_policy, sub_policy_arg_prefix_str)));
      }
      return new CompositeLearningRate<T>(sub_policies);
    } else {
      CAFFE_THROW("Unknown learning rate policy: ", policy);
      return NULL;
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_SGD_LEARNING_RATE_OP_H_
