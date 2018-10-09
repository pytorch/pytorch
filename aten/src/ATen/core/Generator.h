#pragma once

#include <stdint.h>
#include <memory>
#include <mutex>
#include <random>
#include <deque>
#include <atomic>
#include <typeinfo>
#include <utility>

#include "c10/util/Exception.h"
#include <c10/util/C++17.h>
#include "ATen/core/Backend.h"

#if !C10_MOBILE
#include "ATen/detail/CUDAHooksInterface.h"
#endif

/*
* Generator note.
* A Pseudo Random Number Generator (PRNG) is an engine that uses an algorithm to 
* generate a seemingly random sequence of numbers, that may be later be used in creating 
* a random distribution. Such an engine almost always maintains a state and requires a
* seed to start off the creation of random numbers. Often times, users have
* encountered that it could be beneficial to be able to create, retain, and destroy 
* generator states and also be able to have control over the seed value.
*
* A Generator in ATen gives users the ability to read, write and modify a PRNG engine.
* For instance, it does so by letting users seed a PRNG engine, get/set the state of the
* engine, etc.
*
* By default, there is one generator state per device, and a device's generator state is 
* lazily created. A user can use the torch.Generator() api to create their own generator.
*/

namespace at {

struct Generator;

/*
* A GeneratorState object contains a generator engine and other state variables.
* It also has copy and assign constructors so that generator states can be deep copied.
*/
struct CAFFE2_API GeneratorState {
  int64_t device = -1;
  DeviceType device_type;
  uint64_t current_seed = 67280421310721;
  uint64_t philox_offset_per_thread;
  std::mt19937_64 cpu_engine;
};

namespace detail {

// API (for internal use)
CAFFE2_API GeneratorState createGenerator(DeviceType device_type=kCPU, int64_t device=-1);
CAFFE2_API Generator& getDefaultGenerator(DeviceType device_type=kCPU, int64_t device=-1);
CAFFE2_API Generator* checkGeneratorWithDefault(Generator* expr, Generator* defaultValue);

} // namespace detail

struct CAFFE2_API Generator {
  // Constructors
  Generator() = default;
  Generator(GeneratorState state_in)
  : state_{c10::guts::make_unique<GeneratorState>(state_in)} { }
  Generator(const Generator& other) 
  : Generator(other, std::lock_guard<std::mutex>(other.mutex)) { }

  // Getter/Setter
  GeneratorState* getState();
  void setState(GeneratorState* state_in);
  uint64_t getCurrentSeed();
  void setCurrentSeed(uint64_t seed);
  std::mt19937_64& getCPUEngine();
  void setCPUEngine(std::mt19937_64 engine);

  // Methods
  uint64_t random64();
  std::pair<uint64_t, uint64_t> incrementPhiloxOffset(uint64_t total_elements,
                                                uint64_t grid_size,
                                                uint64_t block_size,
                                                uint64_t num_engine_calls);

private:
  std::unique_ptr<GeneratorState> state_;
  mutable std::mutex mutex;

  // constructor forwarding to grab mutex before any copying or moving
  Generator(const Generator &other, const std::lock_guard<std::mutex> &)
   : state_(c10::guts::make_unique<GeneratorState>(*(other.state_))) {}
};

} // namespace at
