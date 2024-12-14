#pragma once

#include <atomic>
#include <utility>

namespace c10 {

/**
 * Thread-safe lazy value with opportunistic concurrency: on concurrent first
 * access, the factory may be called by multiple threads, but only one result is
 * stored and its reference returned to all the callers.
 *
 * Value is heap-allocated; this optimizes for the case in which the value is
 * never actually computed.
 */
template <class T>
class OptimisticLazy {
 public:
  OptimisticLazy() = default;
  OptimisticLazy(const OptimisticLazy& other) {
    if (T* value = other.value_.load(std::memory_order_acquire)) {
      value_ = new T(*value);
    }
  }
  OptimisticLazy(OptimisticLazy&& other) noexcept
      : value_(other.value_.exchange(nullptr, std::memory_order_acq_rel)) {}
  ~OptimisticLazy() {
    reset();
  }

  template <class Factory>
  T& ensure(const Factory& factory) {
    if (T* value = value_.load(std::memory_order_acquire)) {
      return *value;
    }
    T* value = new T(factory());
    T* old = nullptr;
    if (!value_.compare_exchange_strong(
            old, value, std::memory_order_release, std::memory_order_acquire)) {
      delete value;
      value = old;
    }
    return *value;
  }

  // The following methods are not thread-safe: they should not be called
  // concurrently with any other method.

  OptimisticLazy& operator=(const OptimisticLazy& other) {
    *this = OptimisticLazy{other};
    return *this;
  }

  OptimisticLazy& operator=(OptimisticLazy&& other) noexcept {
    if (this != &other) {
      reset();
      value_.store(
          other.value_.exchange(nullptr, std::memory_order_acquire),
          std::memory_order_release);
    }
    return *this;
  }

  void reset() {
    if (T* old = value_.load(std::memory_order_relaxed)) {
      value_.store(nullptr, std::memory_order_relaxed);
      delete old;
    }
  }

 private:
  std::atomic<T*> value_{nullptr};
};

/**
 * Interface for a value that is computed on first access.
 */
template <class T>
class LazyValue {
 public:
  virtual ~LazyValue() = default;

  virtual const T& get() const = 0;
};

/**
 * Convenience thread-safe LazyValue implementation with opportunistic
 * concurrency.
 */
template <class T>
class OptimisticLazyValue : public LazyValue<T> {
 public:
  const T& get() const override {
    return value_.ensure([this] { return compute(); });
  }

 private:
  virtual T compute() const = 0;

  mutable OptimisticLazy<T> value_;
};

/**
 * Convenience immutable (thus thread-safe) LazyValue implementation for cases
 * in which the value is not actually lazy.
 */
template <class T>
class PrecomputedLazyValue : public LazyValue<T> {
 public:
  PrecomputedLazyValue(T value) : value_(std::move(value)) {}

  const T& get() const override {
    return value_;
  }

 private:
  T value_;
};

} // namespace c10
