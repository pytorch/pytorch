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

// The default seed is selected to be a large number
// with good distribution of 0s and 1s in bit representation
constexpr uint64_t default_rng_seed_val = 67280421310721;

struct CAFFE2_API Generator {
  // Constructors
  Generator(Device device_in, DispatchKeySet key_set);

  // Delete all copy and move assignment in favor of clone()
  // method
  Generator(const Generator& other) = delete;
  Generator(Generator&& other) = delete;
  Generator& operator=(const Generator& other) = delete;

  virtual ~Generator() = default;
  std::shared_ptr<Generator> clone() const;

  // Common methods for all generators
  virtual void set_current_seed(uint64_t seed) = 0;
  virtual uint64_t current_seed() const = 0;
  virtual uint64_t seed() = 0;
  Device device() const;

  // See Note [Acquire lock when using random generators]
  std::mutex mutex_;

  DispatchKeySet key_set() const { return key_set_; }

  private:
    Device device_;
    DispatchKeySet key_set_;
    virtual Generator* clone_impl() const = 0;
};

typedef Generator* GeneratorHolder;

namespace detail {

CAFFE2_API uint64_t getNonDeterministicRandom(bool is_cuda = false);

} // namespace detail

} // namespace at

