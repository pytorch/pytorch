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

#ifndef CAFFE2_CONTRIB_OBSERVERS_TIME_OBSERVER_H_
#define CAFFE2_CONTRIB_OBSERVERS_TIME_OBSERVER_H_

#include <unordered_map>

#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

namespace caffe2 {

template <class T>
class TimeObserverBase : public ObserverBase<T> {
 public:
  explicit TimeObserverBase<T>(T* subject) : ObserverBase<T>(subject) {}
  inline float average_time() const {
    return total_time_ / iterations_;
  }
  ~TimeObserverBase() {}

  bool Start() override;
  bool Stop() override;

 protected:
  Timer timer_;
  float start_time_ = 0.0f;
  float total_time_ = 0.0f;
  int iterations_ = 0;
};

template <class T>
class TimeObserver final : public TimeObserverBase<T> {
 public:
  explicit TimeObserver<T>(T* subject) : TimeObserverBase<T>(subject) {}
};

template <>
class TimeObserver<OperatorBase> final : public TimeObserverBase<OperatorBase> {
 public:
  explicit TimeObserver<OperatorBase>(OperatorBase* subject)
      : TimeObserverBase<OperatorBase>(subject) {}

  std::unique_ptr<ObserverBase<OperatorBase>> clone() override {
    return std::unique_ptr<ObserverBase<OperatorBase>>(
        new TimeObserver<OperatorBase>(this->subject_));
  }
};

template <>
class TimeObserver<NetBase> final : public TimeObserverBase<NetBase> {
 public:
  explicit TimeObserver<NetBase>(NetBase* subject)
      : TimeObserverBase<NetBase>(subject) {}
  float average_time_children() const {
    float sum = 0.0f;
    for (const auto* observer : operator_observers_) {
      sum += observer->average_time();
    }
    return sum / subject_->GetOperators().size();
  }

  bool Start() override {
    for (auto* op : subject_->GetOperators()) {
      const auto* observer = op->AttachObserver(
          caffe2::make_unique<TimeObserver<OperatorBase>>(op));
      CAFFE_ENFORCE(observer != nullptr);
      operator_observers_.push_back(
          dynamic_cast_if_rtti<const TimeObserver<OperatorBase>*>(observer));
    }
    start_time_ = timer_.MilliSeconds();
    ++iterations_;
    return true;
  }

 private:
  vector<const TimeObserver<OperatorBase>*> operator_observers_;
};

} // namespace caffe2

#endif // CAFFE2_CONTRIB_OBSERVERS_TIME_OBSERVER_H_
