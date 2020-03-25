#ifndef CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_
#define CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_

#include <cmath>
#include <list>
#include <map>

#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif // _MSC_VER

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// LearningRateFunctor is a functor that when fed with an iter number, produces
// the learning rate for the corresponding iteration.
template <typename T>
class LearningRateFunctor {
 public:
  virtual ~LearningRateFunctor() {}
  virtual T operator()(const int64_t iter) const = 0;
};

// Fixed: not changing the learning rate at all.
template <typename T>
class FixedLearningRate : public LearningRateFunctor<T> {
 public:
  T operator()(const int64_t /*iter*/) const override {
    return 1.;
  }
};

// Alter: alternatate learning rate with active_period and inactive_period.
// update for for a duration of active_period and then stop for a duration of
// inactive_period if active_first, and vice versa
template <typename T>
class AlternateLearningRate : public LearningRateFunctor<T> {
 public:
  AlternateLearningRate(
      const int64_t active_period,
      const int64_t inactive_period,
      const bool active_first)
      : active_period_(active_period),
        inactive_period_(inactive_period),
        active_first_(active_first) {}
  T operator()(const int64_t iter) const override {
    if (iter % (active_period_ + inactive_period_) <
        (active_first_ ? active_period_ : inactive_period_)) {
      return active_first_ ? 1. : 0.;
    } else {
      return active_first_ ? 0. : 1.;
    };
  };

  int64_t active_period_;
  int64_t inactive_period_;
  bool active_first_;
};

// Step: return gamma ^ (floor(iter / step))
template <typename T>
class StepLearningRate : public LearningRateFunctor<T> {
 public:
  StepLearningRate(const int stepsize, const T gamma)
      : stepsize_(stepsize), gamma_(gamma) {}
  T operator()(const int64_t iter) const override {
    return std::pow(gamma_, static_cast<T>(iter / stepsize_));
  }

  int stepsize_;
  T gamma_;
};

// Exp: return gamma ^ iter
template <typename T>
class ExpLearningRate : public LearningRateFunctor<T> {
 public:
  explicit ExpLearningRate(const T gamma) : gamma_(gamma) {}
  T operator()(const int64_t iter) const override {
    return std::pow(gamma_, static_cast<T>(iter));
  }

  T gamma_;
};

// Gate: return multiplier_1 if before num_iter, else multiplier_2
template <typename T>
class GateLearningRate : public LearningRateFunctor<T> {
 public:
  GateLearningRate(
      const T multiplier_1,
      const T multiplier_2,
      const int64_t num_iter)
      : multiplier_1_(multiplier_1),
        multiplier_2_(multiplier_2),
        num_iter_(num_iter) {}
  T operator()(const int64_t iter) const override {
    if (iter >= int64_t(num_iter_)) {
      return T(multiplier_2_);
    }
    return T(multiplier_1_);
  }
  T multiplier_1_;
  T multiplier_2_;
  uint64_t num_iter_;
};

// Inv: return (1 + gamma * iter) ^ (-power)
template <typename T>
class InvLearningRate : public LearningRateFunctor<T> {
 public:
  InvLearningRate(const T gamma, const T power)
      : gamma_(gamma), power_(power) {}
  T operator()(const int64_t iter) const override {
    return std::pow(T(1) + gamma_ * iter, -power_);
  }
  T gamma_;
  T power_;
};

// Poly: return (1 - iter/max_iter) ^ (power)
template <typename T>
class PolyLearningRate : public LearningRateFunctor<T> {
 public:
  PolyLearningRate(const T power, const int64_t max_iter)
      : power_(power), max_iter_(max_iter) {}
  T operator()(const int64_t iter) const override {
    return std::pow(1 - T(iter) / T(max_iter_), power_);
  }
  T power_;
  uint64_t max_iter_;
};

// LinearWarmup: return max(iter/num_iter, 1)
template <typename T>
class LinearWarmupLearningRate : public LearningRateFunctor<T> {
 public:
  LinearWarmupLearningRate(const T start_multiplier, const int64_t num_iter)
      : start_multiplier_(start_multiplier), num_iter_(num_iter) {}
  T operator()(const int64_t iter) const override {
    if (iter >= int64_t(num_iter_)) {
      return 1.;
    }
    return start_multiplier_ +
        (1. - start_multiplier_) * T(iter) / T(num_iter_);
  }
  T start_multiplier_;
  uint64_t num_iter_;
};

// ConstantWarmup: return scale when iter < num_iter, and 1 otherwise
template <typename T>
class ConstantWarmupLearningRate : public LearningRateFunctor<T> {
 public:
  ConstantWarmupLearningRate(const T multiplier, const int64_t num_iter)
      : multiplier_(multiplier), num_iter_(num_iter) {}
  T operator()(const int64_t iter) const override {
    if (iter >= int64_t(num_iter_)) {
      return 1.;
    }
    return T(multiplier_);
  }
  T multiplier_;
  uint64_t num_iter_;
};

// ConstantWarmup: return scale when iter < num_iter, and 1 otherwise
template <typename T>
class PieceWarmupLearningRate : public LearningRateFunctor<T> {
 public:
  PieceWarmupLearningRate(
      const T m1,
      const int64_t n1,
      const T m2,
      const int64_t n2,
      const T m3)
      : m1_(m1), m2_(m2), m3_(m3), n1_(n1), n2_(n2){};

  T operator()(const int64_t iter) const override {
    if (iter < int64_t(n1_)) {
      return m1_;
    } else if (iter < int64_t(n2_)) {
      return m2_;
    }
    return m3_;
  }

  T m1_, m2_, m3_;
  uint64_t n1_, n2_;
};

// hill: the learning rate changes according to following 3 stages
// 1) linear warmup (increasing) at first num_iter steps from start_multiplier
// 2) inverse shrink (decreasing) afterwards (gamma, power)
// 3) lower bounded by end_multiplier
template <typename T>
class HillLearningRate : public LearningRateFunctor<T> {
 public:
  HillLearningRate(
      const int64_t num_iter,
      const T start_multiplier,
      const T gamma,
      const T power,
      const T end_multiplier)
      : linear_warmup_lr_(start_multiplier, num_iter),
        inv_lr_(gamma, power),
        num_iter_(num_iter),
        end_multiplier_(end_multiplier) {}
  T operator()(const int64_t iter) const override {
    if (iter < num_iter_) {
      return linear_warmup_lr_(iter);
    } else {
      return std::max(end_multiplier_, inv_lr_(iter - num_iter_));
    }
  }
  LinearWarmupLearningRate<T> linear_warmup_lr_;
  InvLearningRate<T> inv_lr_;
  int64_t num_iter_;
  T end_multiplier_;
};

template <typename T>
class CompositeLearningRateItem {
 public:
  CompositeLearningRateItem(
      int64_t num_iter,
      float lr_scale,
      LearningRateFunctor<T>* policy)
      : num_iter_(num_iter), lr_scale_(lr_scale), policy_(policy) {}
  int64_t num_iter_;
  float lr_scale_;
  LearningRateFunctor<T>* policy_;
};

// composite: the learning policy changes according to current iteration #
template <typename T>
class CompositeLearningRate : public LearningRateFunctor<T> {
 public:
  CompositeLearningRate(
      const std::list<CompositeLearningRateItem<T>>& sub_policies) {
    DCHECK_GT(sub_policies.size(), 0);
    int64_t num_iter_start = 1;
    for (auto it = sub_policies.begin(); it != sub_policies.end(); ++it) {
      DCHECK_GT(it->num_iter_, 0);
      sub_policies_[num_iter_start].reset(it->policy_);
      sub_policy_lr_scales_[num_iter_start] = it->lr_scale_;
      num_iter_start += it->num_iter_;
    }
  }
  T operator()(const int64_t iter) const override {
    auto sub_policy = sub_policies_.upper_bound(iter);
    DCHECK(sub_policy != sub_policies_.begin());
    --sub_policy;
    auto sub_policy_lr_scale = sub_policy_lr_scales_.upper_bound(iter);
    DCHECK(sub_policy_lr_scale != sub_policy_lr_scales_.begin());
    --sub_policy_lr_scale;
    return ((*sub_policy->second)(iter)) * (sub_policy_lr_scale->second);
  }

 private:
  std::map<int64_t, std::unique_ptr<LearningRateFunctor<T>>> sub_policies_;
  std::map<int64_t, float> sub_policy_lr_scales_;
};

// Cyclical: return a learning rate with period 2 * stepsize and
// lower bound base_lr, upper bound max_lr.
// See https://arxiv.org/pdf/1506.01186.pdf
template <typename T>
class CyclicalLearningRate : public LearningRateFunctor<T> {
 public:
  CyclicalLearningRate(
      const T base_lr,
      const T max_lr,
      const int stepsize,
      const T decay)
      : base_lr_(base_lr),
        max_lr_(max_lr),
        stepsize_(stepsize),
        decay_(decay) {}
  T operator()(const int64_t iter) const override {
    int64_t cycle = static_cast<int>((iter / (2 * stepsize_)) + 1);
    T x = abs(static_cast<T>(iter) / stepsize_ - 2 * cycle + 1);
    return 1 +
        (T(abs(max_lr_)) / T(abs(base_lr_)) - 1) * std::max(T(0.0), (1 - x)) *
        std::pow(decay_, static_cast<int>(iter / (2 * stepsize_)));
  }
  T base_lr_;
  T max_lr_;
  int stepsize_;
  T decay_;
};

// Cosine: return a learning rate with a cosine schedule
// lower bound min_lr, upper bound max_lr.
// See https://arxiv.org/pdf/1608.03983.pdf
template <typename T>
class CosineLearningRate : public LearningRateFunctor<T> {
 public:
  CosineLearningRate(
      const T min_lr,
      const T max_lr,
      const int64_t period,
      const T t_mult,
      const T lr_shrink)
      : min_lr_(min_lr),
        max_lr_(max_lr),
        period_(period),
        t_mult_(t_mult),
        lr_shrink_(lr_shrink) {}
  T operator()(const int64_t iter) const override {
    T i, t_i, t_curr;
    if (t_mult_ != 1.0) {
      // the period is changed every time
      i = floor(
          log(1 - double(iter) / double(period_) * (1.0 - t_mult_)) /
          log(t_mult_));
      t_i = pow(t_mult_, i) * period_;
      t_curr = iter - (1.0 - pow(t_mult_, i)) / (1.0 - t_mult_) * period_;
    } else {
      // fixed period
      i = floor(double(iter) / double(period_));
      t_i = period_;
      t_curr = iter - t_i * i;
    }
    T lr_shrink = pow(lr_shrink_, i);
    T min_lr = min_lr_ * lr_shrink;
    T max_lr = max_lr_ * lr_shrink;
    T final_lr =
        min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(M_PI * t_curr / t_i));
    return final_lr;
  }
  T min_lr_;
  T max_lr_;
  int64_t period_;
  T t_mult_;
  T lr_shrink_;
};

// constantThenLinearWarmup: first use a constant multiplier
// and then ramp up to the global lr
template <typename T>
class ConstantThenLinearWarmupLearningRate : public LearningRateFunctor<T> {
 public:
  ConstantThenLinearWarmupLearningRate(
      const T start_warmup_multiplier,
      const int64_t constant_warmup_num_iter,
      const int64_t linear_warmup_num_iter)
      : constant_warmup_num_iter_(constant_warmup_num_iter),
        linear_warmup_num_iter_(linear_warmup_num_iter),
        constant_warmup_lr_(start_warmup_multiplier, constant_warmup_num_iter),
        linear_warmup_lr_(start_warmup_multiplier, linear_warmup_num_iter) {}

  T operator()(const int64_t iter) const override {
    if (iter < constant_warmup_num_iter_) {
      return constant_warmup_lr_(iter);
    } else if (iter < constant_warmup_num_iter_ + linear_warmup_num_iter_) {
      return linear_warmup_lr_(iter - constant_warmup_num_iter_);
    } else {
      return 1.0;
    }
  }
  int64_t constant_warmup_num_iter_;
  int64_t linear_warmup_num_iter_;
  ConstantWarmupLearningRate<T> constant_warmup_lr_;
  LinearWarmupLearningRate<T> linear_warmup_lr_;
};

// CompositeCosineLearningRate: first use a constant multiplier
// and then ramp up to the global lr, and then use a cosine learning rate
template <typename T>
class CompositeCosineLearningRate : public LearningRateFunctor<T> {
 public:
  CompositeCosineLearningRate(
      const T start_warmup_multiplier,
      const int64_t constant_warmup_num_iter,
      const int64_t linear_warmup_num_iter,
      const T cosine_min_lr,
      const T cosine_max_lr,
      const int64_t cosine_period,
      const T consine_t_mult,
      const T cosine_lr_shrink)
      : constant_warmup_num_iter_(constant_warmup_num_iter),
        linear_warmup_num_iter_(linear_warmup_num_iter),
        constant_then_linear_warmup_lr_(
            start_warmup_multiplier,
            constant_warmup_num_iter,
            linear_warmup_num_iter),
        cosine_lr_(
            cosine_min_lr,
            cosine_max_lr,
            cosine_period,
            consine_t_mult,
            cosine_lr_shrink) {}

  T operator()(const int64_t iter) const override {
    if (iter < constant_warmup_num_iter_ + linear_warmup_num_iter_) {
      return constant_then_linear_warmup_lr_(iter);
    }
    return cosine_lr_(
        iter - constant_warmup_num_iter_ - linear_warmup_num_iter_);
  }

  int64_t constant_warmup_num_iter_;
  int64_t linear_warmup_num_iter_;
  ConstantThenLinearWarmupLearningRate<T> constant_then_linear_warmup_lr_;
  CosineLearningRate<T> cosine_lr_;
};

// CompositeCyclicalLearningRate: first use a constant multiplier
// and then ramp up to the global lr, and then use a cyclical learning rate
template <typename T>
class CompositeCyclicalLearningRate : public LearningRateFunctor<T> {
 public:
  CompositeCyclicalLearningRate(
      const T base_lr,
      const T start_warmup_multiplier,
      const int64_t constant_warmup_num_iter,
      const int64_t linear_warmup_num_iter,
      const T cyclical_max_lr,
      const int cyclical_step_size,
      const T cyclical_decay)
      : constant_warmup_num_iter_(constant_warmup_num_iter),
        linear_warmup_num_iter_(linear_warmup_num_iter),
        constant_then_linear_warmup_lr_(
            start_warmup_multiplier,
            constant_warmup_num_iter,
            linear_warmup_num_iter),
        cyclical_lr_(
            base_lr,
            cyclical_max_lr,
            cyclical_step_size,
            cyclical_decay) {}

  T operator()(const int64_t iter) const override {
    if (iter < constant_warmup_num_iter_ + linear_warmup_num_iter_) {
      return constant_then_linear_warmup_lr_(iter);
    }
    return cyclical_lr_(
        iter - constant_warmup_num_iter_ - linear_warmup_num_iter_);
  }

  int64_t constant_warmup_num_iter_;
  int64_t linear_warmup_num_iter_;
  ConstantThenLinearWarmupLearningRate<T> constant_then_linear_warmup_lr_;
  CyclicalLearningRate<T> cyclical_lr_;
};

} // namespace caffe2

#endif // CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_
