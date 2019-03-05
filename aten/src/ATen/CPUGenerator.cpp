#include <ATen/CPUGenerator.h>
#include <c10/util/C++17.h>

namespace at {

namespace detail {

// Ensures default_gen_cpu is initialized once.
static std::once_flag cpu_device_flag;

// Default, global CPU generator.
static std::unique_ptr<CPUGenerator> default_gen_cpu;

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultCPUGenerator gets the default generator for a particular
 * device.
 */
std::unique_ptr<CPUGenerator>& getDefaultCPUGenerator() {
  std::call_once(cpu_device_flag, [&] {
    default_gen_cpu = c10::guts::make_unique<CPUGenerator>(default_rng_seed_val);
  });
  return default_gen_cpu;
}

/**
 * Utility to create a CPUGenerator. Returns a unique_ptr
 */
std::unique_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val) {
  return c10::guts::make_unique<CPUGenerator>(seed_val, Philox4_32_10(seed_val));
}

} // namespace detail

/**
 * CPUGenerator class implementation
 */
CPUGenerator::CPUGenerator(uint64_t seed_in, Philox4_32_10 engine_in)
  : CloneableGenerator(Device(DeviceType::CPU)), current_seed_(seed_in), engine_(engine_in) {}

/**
 * Manually seeds the engine with the seed input
 */
void CPUGenerator::set_current_seed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(mutex_);
  current_seed_ = seed;
  engine_ = Philox4_32_10(seed);
}

/*
 * Gets the current seed of CPUGenerator.
 */
uint64_t CPUGenerator::current_seed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return current_seed_;
}

/*
 * Gets the DeviceType of CPUGenerator.
 * Used for type checking during run time.
 */
DeviceType CPUGenerator::device_type() {
  return DeviceType::CPU;
}

/* 
* Gets a random 32 bit unsigned integer from the engine
*/
uint32_t CPUGenerator::random() {
  std::lock_guard<std::mutex> lock(mutex_);
  return engine_();
}

/* 
* Gets a random 64 bit unsigned integer from the engine
*/
uint64_t CPUGenerator::random64() {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t hi = static_cast<uint64_t>(engine_()) << 32;
  uint64_t lo = static_cast<uint64_t>(engine_());
  return hi | lo;
}

/*
Private clone method implementation
*/
CloneableGenerator<CPUGenerator, Generator>* CPUGenerator::clone_impl() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return new CPUGenerator(current_seed_, engine_);
}

} // namespace at
