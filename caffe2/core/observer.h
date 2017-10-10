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

#pragma once

#include "caffe2/core/logging.h"

namespace caffe2 {

/**
 *  Use this to implement a Observer using the Observer Pattern template.
 */

template <class T>
class ObserverBase {
 public:
  explicit ObserverBase(T* subject) : subject_(subject) {}

  virtual bool Start() {
    return false;
  }
  virtual bool Stop() {
    return false;
  }

  virtual ~ObserverBase() noexcept {};

  T* subject() const {
    return subject_;
  }

 protected:
  T* subject_;
};

/**
 *  Inherit to make your class observable.
 */
template <class T>
class Observable {
 public:
  using Observer = ObserverBase<T>;

  /* Returns a reference to the observer after addition. */
  Observer* AddObserver(std::unique_ptr<Observer> observer) {
    observers_.emplace_back(std::move(observer));
    return observers_.back().get();
  }

  size_t NumObservers() {
    return observers_.size();
  }

  void SetObserver(std::unique_ptr<Observer> observer, size_t index) {
    CAFFE_ENFORCE(index < observers_.size(), "Index out of bounds.");
    observers_.at(index) = std::move(observer);
  }

  size_t GetObserverIndex(Observer* observer) {
    for (size_t index = 0; index < observers_.size(); ++index) {
      if (observers_.at(index).get() == observer) {
        return index;
      }
    }
    CAFFE_THROW("Observer not added to this net.");
  }

  Observer* GetObserver(size_t index) {
    CAFFE_ENFORCE(index < observers_.size(), "Index out of bounds.");
    return observers_.at(index).get();
  }

  void RemoveObserver(size_t index) {
    CAFFE_ENFORCE(index < observers_.size(), "Index out of bounds.");
    observers_.erase(observers_.begin() + index);
  }

  void RemoveObserver(Observer* observer) {
    observers_.erase(observers_.begin() + GetObserverIndex(observer));
  }

  void StartAllObservers() {
    for (const auto& observer : observers_) {
      observer->Start();
    }
  }

  void StopAllObservers() {
    for (const auto& observer : observers_) {
      observer->Stop();
    }
  }

 protected:
  vector<std::unique_ptr<ObserverBase<T>>> observers_;
};

} // namespace caffe2
