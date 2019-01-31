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

/*
* Generator note.
* A Pseudo Random Number Generator (PRNG) is an engine that uses an algorithm to 
* generate a seemingly random sequence of numbers, that may be later be used in creating 
* a random distribution. Such an engine almost always maintains a state and requires a
* seed to start off the creation of random numbers. Often times, users have
* encountered that it could be beneficial to be able to create, retain, and destroy 
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

constexpr uint64_t default_rng_seed_val = 67280421310721;

struct CAFFE2_API Generator {
  // Constructors
  Generator(Device device_in, uint64_t seed_in);
  Generator(const Generator& other);
  Generator(Generator&& other);
  virtual ~Generator() = default;

  // Common methods for all generators
  virtual void setCurrentSeed(uint64_t seed);
  uint64_t getCurrentSeed();
  Device getDevice();

  // stubbed. will be removed
  virtual Generator& manualSeedAll(uint64_t seed);

  protected:
    mutable std::mutex mutex;
    Device device_;
    uint64_t current_seed_;

    // constructor forwarding to grab mutex before any copying or moving
    Generator(const Generator &other, const std::lock_guard<std::mutex> &);
    Generator(const Generator &&other, const std::lock_guard<std::mutex> &);
};

} // namespace at
