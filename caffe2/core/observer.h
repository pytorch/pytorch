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

#include <map>
#include <memory>

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
  const Observer* AttachObserver(std::unique_ptr<Observer> observer) {
    const Observer* weak_observer = observer.get();
    observers_[weak_observer] = std::move(observer);
    return weak_observer;
  }

  /* Returns a unique_ptr to the observer. */
  std::unique_ptr<Observer> DetachObserver(const Observer* observer) {
    std::unique_ptr<Observer> strong_observer = std::move(observers_[observer]);
    observers_.erase(observer);
    return strong_observer;
  }

  size_t NumObservers() {
    return observers_.size();
  }

  void StartAllObservers() {
    for (const auto& observer : observers_) {
      observer.second->Start();
    }
  }

  void StopAllObservers() {
    for (const auto& observer : observers_) {
      observer.second->Stop();
    }
  }

 protected:
  std::map<const Observer*, std::unique_ptr<ObserverBase<T>>> observers_;
};

} // namespace caffe2
