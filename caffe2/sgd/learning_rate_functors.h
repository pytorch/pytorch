#ifndef CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_
#define CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename dtype>
class LearningRateFunctor {
 public:
  virtual dtype operator()(const int iter) const = 0;
};

// Fixed: not changing the learning rate at all.
template <typename dtype>
class FixedLearningRate : public LearningRateFunctor<dtype> {
 public:
  dtype operator()(const int iter) const override { return 1.; }
};

// Step: return gamma ^ (floor(iter / step))
template <typename dtype>
class StepLearningRate : public LearningRateFunctor<dtype> {
 public:
  StepLearningRate(const int stepsize, const dtype gamma)
      : stepsize_(stepsize), gamma_(gamma) {}
  dtype operator()(const int iter) const override {
    return std::pow(gamma_, static_cast<dtype>(iter / stepsize_));
  }

  int stepsize_;
  dtype gamma_;
};

// Exp: return gamma ^ iter
template <typename dtype>
class ExpLearningRate : public LearningRateFunctor<dtype> {
 public:
  explicit ExpLearningRate(const dtype gamma) : gamma_(gamma) {}
  dtype operator()(const int iter) const override {
    return std::pow(gamma_, static_cast<dtype>(iter));
  }

  dtype gamma_;
};

// Inv: return (1 + gamma * iter) ^ (-power)
template <typename dtype>
class InvLearningRate : public LearningRateFunctor<dtype> {
 public:
  InvLearningRate(const dtype gamma, const dtype power)
      : gamma_(gamma), power_(power) {}
  dtype operator()(const int iter) const override {
      return std::pow(dtype(1) + gamma_ * iter, -power_);
  }
  dtype gamma_;
  dtype power_;
};

}  // namespace caffe2

#endif  // CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_
