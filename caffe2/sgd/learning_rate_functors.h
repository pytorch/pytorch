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
    return std::pow(1 - T(iter)/T(max_iter_), power_);
  }
  T power_;
  uint64_t max_iter_;
};

}  // namespace caffe2

#endif  // CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_
