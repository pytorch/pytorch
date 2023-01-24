#pragma once

#include <stdint.h>
#include <mutex>
#include <deque>
#include <atomic>
#include <typeinfo>
#include <utility>
#include <cstddef>

#include <c10/util/Exception.h>
#include <c10/util/C++17.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>

// For the record I don't think this is a correct pimpl idiom.
// Including Impl header in interface header defeats the purpose
// because you can't change Impl private members without forcing
// everything that included the interface to rebuild.
// Impl should be forward-declared in the interface header instead.
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

class Tensor;

struct TORCH_API Generator {
  Generator() = default;

  explicit Generator(c10::intrusive_ptr<c10::GeneratorImpl> gen_impl)
   : impl_(std::move(gen_impl)) {
    if (impl_.get() == nullptr) {
      throw std::runtime_error("GeneratorImpl with nullptr is not supported");
    }
  }

  bool operator==(const Generator& rhs) const {
    return this->impl_ == rhs.impl_;
  }

  bool operator!=(const Generator& rhs) const {
    return !((*this) == rhs);
  }

  bool defined() const {
    return static_cast<bool>(impl_);
  }

  c10::GeneratorImpl* unsafeGetGeneratorImpl() const {
    return impl_.get();
  }

  c10::GeneratorImpl* unsafeReleaseGeneratorImpl() {
    return impl_.release();
  }

  const c10::intrusive_ptr<c10::GeneratorImpl>& getIntrusivePtr() const {
    return impl_;
  }

  void set_current_seed(uint64_t seed) { impl_->set_current_seed(seed); }

  uint64_t current_seed() const { return impl_->current_seed(); }

  uint64_t seed() { return impl_->seed(); }

  // Implementation not inlined to prevent cycle reference between
  // `ATen/core/Generator.h` and `ATen/core/Tensor.h`
  void set_state(const at::Tensor& new_state);

  at::Tensor get_state() const;

  std::mutex& mutex() {
    return impl_->mutex_;
  }

  DispatchKeySet key_set() const {
    return impl_->key_set();
  }

  Device device() const { return impl_->device(); }

  inline void set_pyobj(PyObject* pyobj) const noexcept {
    impl_->set_pyobj(pyobj);
  }

  inline PyObject* pyobj() const noexcept {
    return impl_->pyobj();
  }

  template<typename T>
  T* get() const { return static_cast<T*>(impl_.get()); }

  Generator clone() const {
    return Generator(impl_->clone());
  }

 private:
  c10::intrusive_ptr<c10::GeneratorImpl> impl_;
};

template<class Impl, class... Args>
Generator make_generator(Args&&... args) {
  return Generator(c10::make_intrusive<Impl>(std::forward<Args>(args)...));
}

/**
 * Utility function to static cast input Generator* to
 * the backend generator type (CPU/CUDAGeneratorImpl etc.)
 */
template <typename T>
static inline T * check_generator(c10::optional<Generator> gen) {
  TORCH_CHECK(gen.has_value(), "Expected Generator but received nullopt");
  TORCH_CHECK(gen->defined(), "Generator with undefined implementation is not allowed");
  TORCH_CHECK(T::device_type() == gen->device().type(), "Expected a '", T::device_type(), "' device type for generator but found '", gen->device().type(), "'");
  return gen->get<T>();
}

/**
 * Utility function used in tensor implementations, which
 * supplies the default generator to tensors, if an input generator
 * is not supplied. The input Generator* is also static casted to
 * the backend generator type (CPU/CUDAGeneratorImpl etc.)
 */
template <typename T>
static inline T* get_generator_or_default(const c10::optional<Generator>& gen, const Generator& default_gen) {
  return gen.has_value() && gen->defined() ? check_generator<T>(gen) : check_generator<T>(default_gen);
}

namespace detail {

/**
 * Helper function for checking the validity of new random generator
 * state. Right now following conditions are checked:
 *
 * - The new state tensor must be a torch.ByteTensor
 * - Data of the new state tensor must be contiguous
 */
static inline void check_rng_state(const c10::TensorImpl& new_state) {
  TORCH_CHECK_TYPE(
    new_state.layout() == kStrided && new_state.device().type() == kCPU && new_state.dtype() == kByte,
    "RNG state must be a torch.ByteTensor"
  );

  TORCH_CHECK(new_state.is_contiguous(), "RNG state must be contiguous");
}

} // namespace detail

} // namespace at
