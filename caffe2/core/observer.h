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

  virtual std::unique_ptr<ObserverBase<T>> rnnCopy(T* subject, int rnn_order)
      const {
    return nullptr;
  };

 protected:
  T* subject_;
};

/**
 *  Inherit to make your class observable.
 */
template <class T>
class Observable {
 public:
  Observable() = default;

  Observable(Observable&&) = default;
  Observable& operator =(Observable&&) = default;

  virtual ~Observable() = default;

  C10_DISABLE_COPY_AND_ASSIGN(Observable);

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
    UpdateCache();

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
        UpdateCache();
        return res;
      }
    }
    return nullptr;
  }

  virtual size_t NumObservers() {
    return num_observers_;
  }

 private:
  inline static void StartObserver(Observer* observer) {
    try {
      observer->Start();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Exception from observer: " << e.what();
    } catch (...) {
      LOG(ERROR) << "Exception from observer: unknown";
    }
  }

  inline static void StopObserver(Observer* observer) {
    try {
      observer->Stop();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Exception from observer: " << e.what();
    } catch (...) {
      LOG(ERROR) << "Exception from observer: unknown";
    }
  }

  void UpdateCache() {
    num_observers_ = observers_list_.size();
    if (num_observers_ != 1) {
      // we cannot take advantage of the cache
      return;
    }
    observer_cache_ = observers_list_[0].get();
  }

 public:
  void StartAllObservers() {
    // do not access observers_list_ unless necessary
    if (num_observers_ == 0) {
      return;
    } else if (num_observers_ == 1) {
      StartObserver(observer_cache_);
    } else {
      for (auto& observer : observers_list_) {
        StartObserver(observer.get());
      }
    }
  }

  void StopAllObservers() {
    // do not access observers_list_ unless necessary
    if (num_observers_ == 0) {
      return;
    } else if (num_observers_ == 1) {
      StopObserver(observer_cache_);
    } else {
      for (auto& observer : observers_list_) {
        StopObserver(observer.get());
      }
    }
  }

 private:
  // an on-stack cache for fast iteration;
  // ideally, inside StartAllObservers and StopAllObservers,
  // we should never access observers_list_
  Observer* observer_cache_;
  size_t num_observers_ = 0;

 protected:
  std::vector<std::unique_ptr<Observer>> observers_list_;
};

} // namespace caffe2
