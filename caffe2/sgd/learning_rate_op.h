#ifndef CAFFE2_SGD_LEARNING_RATE_OP_H_
#define CAFFE2_SGD_LEARNING_RATE_OP_H_

#include <cfloat>
#include <cmath>
#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include <c10/util/irange.h>
#include "caffe2/core/operator.h"
#include "caffe2/sgd/learning_rate_functors.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(LearningRate);

namespace caffe2 {

template <typename T, class Context>
class LearningRateOp final : public Operator<Context> {
 public:
  template <class... Args>
  LearningRateOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        functor_(nullptr),
        base_lr_(this->template GetSingleArgument<float>("base_lr", FLT_MAX)) {
    CAFFE_ENFORCE_NE(base_lr_, FLT_MAX, "Base learning rate must be set.");
    const string policy =
        this->template GetSingleArgument<string>("policy", "");
    CAFFE_ENFORCE(policy.size(), "Must specify a learning rate policy.");
    functor_.reset(createLearningRateFunctor(policy));
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    int64_t iter =
        OperatorBase::Input<Tensor>(0, CPU).template data<int64_t>()[0];
    T learning_rate = base_lr_ * (*functor_)(iter);
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

  LearningRateFunctor<T>* createLearningRateFunctor(
      const string& policy,
      const string& arg_prefix = "") {
    if (policy == "fixed") {
      return new FixedLearningRate<T>();
    } else if (policy == "alter") {
      bool active_first = this->template GetSingleArgument<bool>(
          arg_prefix + "active_first", true);
      int64_t active_period = this->template GetSingleArgument<int64_t>(
          arg_prefix + "active_period", -1);
      int64_t inactive_period = this->template GetSingleArgument<int64_t>(
          arg_prefix + "inactive_period", -1);
      TORCH_DCHECK_GE(active_period, 0);
      TORCH_DCHECK_GE(inactive_period, 0);
      return new AlternateLearningRate<T>(
          active_period, inactive_period, active_first);
    } else if (policy == "hill") {
      int64_t num_iter =
          this->template GetSingleArgument<int64_t>(arg_prefix + "num_iter", 0);
      TORCH_DCHECK_GT(num_iter, 0);
      T start_multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "start_multiplier", 0.);
      TORCH_DCHECK_GE(start_multiplier, 0); // start_multiplier in range [0, 1]
      TORCH_DCHECK_LE(start_multiplier, 1);
      T gamma =
          this->template GetSingleArgument<float>(arg_prefix + "gamma", 0);
      TORCH_DCHECK_GT(gamma, 0);
      T power =
          this->template GetSingleArgument<float>(arg_prefix + "power", 0);
      TORCH_DCHECK_GT(power, 0);
      T end_multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "end_multiplier", 0);
      TORCH_DCHECK_GE(end_multiplier, 0); // end_multiplier in range [0, 1]
      TORCH_DCHECK_LE(end_multiplier, 1);
      return new HillLearningRate<T>(
          num_iter, start_multiplier, gamma, power, end_multiplier);
    } else if (policy == "slope") {
      int64_t num_iter_1 = this->template GetSingleArgument<int64_t>(
          arg_prefix + "num_iter_1", 0);
      TORCH_DCHECK_GT(num_iter_1, 0);
      T multiplier_1 = this->template GetSingleArgument<float>(
          arg_prefix + "multiplier_1", 0.);
      int64_t num_iter_2 = this->template GetSingleArgument<int64_t>(
          arg_prefix + "num_iter_2", 0);
      TORCH_DCHECK_GT(num_iter_1, 0);
      T multiplier_2 = this->template GetSingleArgument<float>(
          arg_prefix + "multiplier_2", 0.);
      TORCH_DCHECK_GT(num_iter_2, num_iter_1);
      return new SlopeLearningRate<T>(
          num_iter_1, multiplier_1, num_iter_2, multiplier_2);
    } else if (policy == "step") {
      int stepsize =
          this->template GetSingleArgument<int>(arg_prefix + "stepsize", 0);
      T gamma =
          this->template GetSingleArgument<float>(arg_prefix + "gamma", 0);
      TORCH_DCHECK_GT(stepsize, 0);
      TORCH_DCHECK_GT(gamma, 0);
      return new StepLearningRate<T>(stepsize, gamma);
    } else if (policy == "exp") {
      T gamma =
          this->template GetSingleArgument<float>(arg_prefix + "gamma", 0);
      TORCH_DCHECK_GT(gamma, 0);
      return new ExpLearningRate<T>(gamma);
    } else if (policy == "gate") {
      T multiplier_1 = this->template GetSingleArgument<float>(
          arg_prefix + "multiplier_1", 1);
      T multiplier_2 = this->template GetSingleArgument<float>(
          arg_prefix + "multiplier_2", 1);
      int num_iter =
          this->template GetSingleArgument<int>(arg_prefix + "num_iter", 0);
      // no constraint on the range of multiplier_1 and multiplier_2
      return new GateLearningRate<T>(multiplier_1, multiplier_2, num_iter);
    } else if (policy == "inv") {
      T gamma =
          this->template GetSingleArgument<float>(arg_prefix + "gamma", 0);
      T power =
          this->template GetSingleArgument<float>(arg_prefix + "power", 0);
      TORCH_DCHECK_GT(gamma, 0);
      TORCH_DCHECK_GT(power, 0);
      return new InvLearningRate<T>(gamma, power);
    } else if (policy == "poly") {
      int max_iter =
          this->template GetSingleArgument<int>(arg_prefix + "max_iter", -1);
      T power =
          this->template GetSingleArgument<float>(arg_prefix + "power", 0);
      TORCH_DCHECK_GT(power, 0);
      return new PolyLearningRate<T>(power, max_iter);
    } else if (policy == "linearWarmup") {
      T start_multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "start_multiplier", 0.);
      int num_iter =
          this->template GetSingleArgument<int>(arg_prefix + "num_iter", 0);
      TORCH_DCHECK_GE(start_multiplier, 0);
      return new LinearWarmupLearningRate<T>(start_multiplier, num_iter);
    } else if (policy == "constantWarmup") {
      T multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "multiplier", 0.5);
      int num_iter =
          this->template GetSingleArgument<int>(arg_prefix + "num_iter", 0);
      TORCH_DCHECK_GT(multiplier, 0);
      return new ConstantWarmupLearningRate<T>(multiplier, num_iter);
    } else if (policy == "pieceWarmup") {
      T m1 = this->template GetSingleArgument<float>(arg_prefix + "m1", 0.5);
      int64_t n1 =
          this->template GetSingleArgument<int64_t>(arg_prefix + "n1", 0);
      T m2 = this->template GetSingleArgument<float>(arg_prefix + "m2", 0.5);
      int64_t n2 =
          this->template GetSingleArgument<int64_t>(arg_prefix + "n2", 0);
      T m3 = this->template GetSingleArgument<float>(arg_prefix + "m3", 0.5);
      return new PieceWarmupLearningRate<T>(m1, n1, m2, n2, m3);
    } else if (policy == "composite") {
      std::vector<int> sub_policy_num_iters =
          this->template GetRepeatedArgument<int>("sub_policy_num_iters");
      std::list<CompositeLearningRateItem<T>> sub_policies;
      CAFFE_ENFORCE_GT(
          sub_policy_num_iters.size(),
          0,
          "Must specify at least one sub learning rate policy.");
      for (const auto i : c10::irange(sub_policy_num_iters.size())) {
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
        const float scale_lr = this->template GetSingleArgument<float>(
            sub_policy_arg_prefix_str + "lr_scale", 1.0);
        sub_policies.push_back(CompositeLearningRateItem<T>(
            sub_policy_num_iters[i],
            scale_lr,
            createLearningRateFunctor(sub_policy, sub_policy_arg_prefix_str)));
      }
      return new CompositeLearningRate<T>(sub_policies);
    } else if (policy == "cyclical") {
      T max_lr =
          this->template GetSingleArgument<float>(arg_prefix + "max_lr", 0.005);
      int stepsize =
          this->template GetSingleArgument<int>(arg_prefix + "stepsize", 0);
      T decay =
          this->template GetSingleArgument<float>(arg_prefix + "decay", 1.0);
      TORCH_DCHECK_GT(stepsize, 0);
      TORCH_DCHECK_GE(max_lr, base_lr_);
      return new CyclicalLearningRate<T>(base_lr_, max_lr, stepsize, decay);
    } else if (policy == "constantThenLinearWarmup") {
      T start_warmup_multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "start_warmup_multiplier", 0.1);
      int64_t constant_warmup_num_iter = this->template GetSingleArgument<int64_t>(
          arg_prefix + "constant_warmup_num_iter", 10000000);
      int64_t linear_warmup_num_iter = this->template GetSingleArgument<int64_t>(
          arg_prefix + "linear_warmup_num_iter", 10000000);
      return new ConstantThenLinearWarmupLearningRate<T>(
          start_warmup_multiplier,
          constant_warmup_num_iter,
          linear_warmup_num_iter);
    } else if (policy == "compositeCyclical") {
      T start_warmup_multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "start_warmup_multiplier", 0.1);
      int64_t constant_warmup_num_iter = this->template GetSingleArgument<int64_t>(
          arg_prefix + "constant_warmup_num_iter", 10000000);
      int64_t linear_warmup_num_iter = this->template GetSingleArgument<int64_t>(
          arg_prefix + "linear_warmup_num_iter", 10000000);
      T cyclical_max_lr = this->template GetSingleArgument<float>(
          arg_prefix + "cyclical_max_lr", 0.05);
      int cyclical_step_size = this->template GetSingleArgument<int>(
          arg_prefix + "cyclical_step_size", 1000000);
      T cyclical_decay = this->template GetSingleArgument<float>(
          arg_prefix + "cyclical_decay", 1.0);
      TORCH_DCHECK_GE(cyclical_max_lr, base_lr_);
      return new CompositeCyclicalLearningRate<T>(
          base_lr_,
          start_warmup_multiplier,
          constant_warmup_num_iter,
          linear_warmup_num_iter,
          cyclical_max_lr,
          cyclical_step_size,
          cyclical_decay);
    } else if (policy == "cosine") {
      T max_lr =
          this->template GetSingleArgument<float>(arg_prefix + "max_lr", 0.5);
      T min_lr =
          this->template GetSingleArgument<float>(arg_prefix + "min_lr", 0.1);
      int64_t period =
          this->template GetSingleArgument<int>(arg_prefix + "period", 50);
      T t_mult =
          this->template GetSingleArgument<float>(arg_prefix + "t_mult", 1.0);
      T lr_shrink = this->template GetSingleArgument<float>(
          arg_prefix + "lr_shrink", 0.99);
      TORCH_DCHECK_GE(max_lr, min_lr);
      return new CosineLearningRate<T>(
          min_lr, max_lr, period, t_mult, lr_shrink);
    } else if (policy == "compositeCosine") {
      T start_warmup_multiplier = this->template GetSingleArgument<float>(
          arg_prefix + "start_warmup_multiplier", 0.1);
      int64_t constant_warmup_num_iter = this->template GetSingleArgument<int64_t>(
          arg_prefix + "constant_warmup_num_iter", 10000000);
      int64_t linear_warmup_num_iter = this->template GetSingleArgument<int64_t>(
          arg_prefix + "linear_warmup_num_iter", 10000000);
      T cosine_max_lr = this->template GetSingleArgument<float>(
          arg_prefix + "cosine_max_lr", 0.5);
      T cosine_min_lr = this->template GetSingleArgument<float>(
          arg_prefix + "cosine_min_lr", 0.1);
      int64_t cosine_period = this->template GetSingleArgument<int>(
          arg_prefix + "cosine_period", 50);
      T cosine_t_mult = this->template GetSingleArgument<float>(
          arg_prefix + "cosine_t_mult", 1.0);
      T cosine_lr_shrink = this->template GetSingleArgument<float>(
          arg_prefix + "cosine_lr_shrink", 0.99);

      TORCH_DCHECK_GE(cosine_max_lr, cosine_min_lr);
      return new CompositeCosineLearningRate<T>(
          start_warmup_multiplier,
          constant_warmup_num_iter,
          linear_warmup_num_iter,
          cosine_min_lr,
          cosine_max_lr,
          cosine_period,
          cosine_t_mult,
          cosine_lr_shrink);
    } else {
      CAFFE_THROW("Unknown learning rate policy: ", policy);
      return NULL;
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_SGD_LEARNING_RATE_OP_H_
