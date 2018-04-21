#ifndef CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_
#define CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_

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
    if (iter >= num_iter_) {
      return 1.;
    }
    return start_multiplier_ + (1. - start_multiplier_) * T(iter) / T(num_iter_);
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
    if (iter >= num_iter_) {
      return 1.;
    }
    return T(multiplier_);
  }
  T multiplier_;
  uint64_t num_iter_;
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

} // namespace caffe2

#endif // CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_
