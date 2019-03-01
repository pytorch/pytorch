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
#include <ATen/Device.h>

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

namespace at {

/**
 * CloneableGenerator class based on CRTP pattern. It is a
 * helper class used for cloning Generator subclasses while 
 * preserving covariance in clone() method.
 */
template <typename Derived, typename Base>
struct CloneableGenerator : public Base {

  CloneableGenerator(Device device_in) : Base(device_in) {}
  virtual ~CloneableGenerator() = default;

  std::unique_ptr<Derived> clone() const {
    return std::unique_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
  }

private:
  virtual CloneableGenerator* clone_impl() const = 0;
};

// The default seed is selected to be a large number
// with good distribution of 0s and 1s in bit representation
constexpr uint64_t default_rng_seed_val = 67280421310721;

struct CAFFE2_API Generator {
  // Constructors
  Generator(Device device_in);

  // Delete all copy and move assignment in favor of clone()
  // method
  Generator(const Generator& other) = delete;
  Generator(Generator&& other) = delete;
  Generator& operator=(Generator& other) = delete;
  Generator& operator=(Generator&& other) = delete;

  virtual ~Generator() = default;
  std::unique_ptr<Generator> clone() const;

  // Common methods for all generators
  virtual void setCurrentSeed(uint64_t seed) = 0;
  virtual uint64_t getCurrentSeed() const = 0;
  Device getDevice() const;

  // stubbed. will be removed
  virtual Generator& manualSeedAll(uint64_t seed);

  protected:
    mutable std::mutex mutex_;

  private:
    Device device_;
    virtual Generator* clone_impl() const = 0;
};

} // namespace at
