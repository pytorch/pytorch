/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

} // namespace caffe2

#endif // CAFFE2_SGD_LEARNING_RATE_FUNCTORS_H_
