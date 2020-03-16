#pragma once

#include <stdint.h>
#include <memory>
#include <mutex>
#include <deque>
#include <atomic>
#include <typeinfo>
#include <utility>

#include <c10/util/Exception.h>
#include <c10/util/C++17.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/GeneratorImpl.h>

/**
 * Note [Generator]
 * ~~~~~~~~~~~~~~~~
 * A Pseudo Random Number Generator (PRNG) is an engine that uses an algorithm to
 * generate a seemingly random sequence of numbers, that may be later be used in creating
 * a random distribution. Such an engine almost always maintains a state and requires a
 * seed to start off the creation of random numbers. Often times, users have
 * found it beneficial to be able to explicitly create, retain, and destroy
 * PRNG states and also be able to have control over the seed value.
 *
 * A Generator in ATen gives users the ability to read, write and modify a PRNG engine.
 * For instance, it does so by letting users seed a PRNG engine, fork the state of the
 * engine, etc.
 *
 * By default, there is one generator per device, and a device's generator is
 * lazily created. A user can use the torch.Generator() api to create their own generator.
 * Currently torch.Generator() can only create a CPUGenerator.
 */

/**
 * Note [Acquire lock when using random generators]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Generator and its derived classes are NOT thread-safe. Please note that most of the
 * places where we have inserted locking for generators are historically based, and we
 * haven't actually checked that everything is truly thread safe (and it probably isn't).
 * Please use the public mutex_ when using any methods from these classes, except for the
 * read-only methods. You can learn about the usage by looking into the unittests
 * (aten/src/ATen/cpu_generator_test.cpp) and other places where we have used lock_guard.
 * 
 * TODO: Look into changing the threading semantics of Generators in ATen (e.g., making
 * them non-thread safe and instead making the generator state splittable, to accommodate
 * forks into other threads).
 */

namespace at {

struct CAFFE2_API Generator {
  Generator() {}

  Generator(c10::GeneratorImpl* g) { this->impl_ = std::shared_ptr<c10::GeneratorImpl>(g); }
  Generator& operator=(c10::GeneratorImpl* g) { this->impl_ = std::shared_ptr<c10::GeneratorImpl>(g); return *this; }

  Generator(std::shared_ptr<c10::GeneratorImpl> g) { this->impl_ = g; }
  Generator& operator=(std::shared_ptr<c10::GeneratorImpl> g) { this->impl_ = g; return *this; }

  bool operator==(const Generator& that) const {
    return (!(this->impl_) && !(that.impl_)) || (this->impl_ == that.impl_);
  }

  bool operator!=(const Generator& that) const {
    return !((*this) == that);
  }

  bool operator==(c10::GeneratorImpl* g) const {
    return this->impl_ && this->impl_.get() == g;
  }

  bool operator!=(c10::GeneratorImpl* g) const {
    return !((*this) == g);
  }

  bool defined() const {
    return (bool)impl_;
  }

  c10::GeneratorImpl* operator->() const { return impl_.get(); }

  c10::GeneratorImpl* get() const { return impl_.get(); }

  template<typename T>
  T* get() { return dynamic_cast<T*>(impl_.get()); }

 private:
  std::shared_ptr<c10::GeneratorImpl> impl_;
};

template<class Impl, class... Args>
Generator make_generator(Args&&... args) {
  return Generator(std::make_shared<Impl>(args...));
}

} // namespace at
