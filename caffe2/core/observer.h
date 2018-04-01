#pragma once

#include <memory>
#include <unordered_set>
#include "caffe2/core/logging.h"

namespace caffe2 {

/**
 *  Use this to implement a Observer using the Observer Pattern template.
 */

template <class T>
class ObserverBase {
 public:
  explicit ObserverBase(T* subject) : subject_(subject) {}

  virtual void Start() {}
  virtual void Stop() {}

  virtual std::string debugInfo() {
    return "Not implemented.";
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
  virtual ~Observable(){};
  using Observer = ObserverBase<T>;

  /* Returns a reference to the observer after addition. */
  const Observer* AttachObserver(std::unique_ptr<Observer> observer) {
    CAFFE_ENFORCE(observer, "Couldn't attach a null observer.");
    std::unordered_set<const Observer*> observers;
    for (auto& ob : observers_list_) {
      observers.insert(ob.get());
    }

    const auto* observer_ptr = observer.get();
    if (observers.count(observer_ptr)) {
      return observer_ptr;
    }
    observers_list_.push_back(std::move(observer));

    return observer_ptr;
  }

  /**
   * Returns a unique_ptr to the removed observer. If not found, return a
   * nullptr
   */
  std::unique_ptr<Observer> DetachObserver(const Observer* observer_ptr) {
    for (auto it = observers_list_.begin(); it != observers_list_.end(); ++it) {
      if (it->get() == observer_ptr) {
        auto res = std::move(*it);
        observers_list_.erase(it);
        return res;
      }
    }
    return nullptr;
  }

  virtual size_t NumObservers() {
    return observers_list_.size();
  }

  void StartAllObservers() {
    for (auto& observer : observers_list_) {
      observer->Start();
    }
  }

  void StopAllObservers() {
    for (auto& observer : observers_list_) {
      observer->Stop();
    }
  }

 protected:
  std::vector<std::unique_ptr<Observer>> observers_list_;
};

} // namespace caffe2
