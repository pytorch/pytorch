#include <ATen/CPUGenerator.h>
#include <c10/util/C++17.h>
#include <algorithm>
#include <random>
#include <chrono>

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
    default_gen_cpu = c10::guts::make_unique<CPUGenerator>(getNonDeterministicRandom());
  });
  return default_gen_cpu;
}

/**
 * Utility to create a CPUGenerator. Returns a unique_ptr
 */
std::unique_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val) {
  return c10::guts::make_unique<CPUGenerator>(seed_val);
}

/**
 * Helper function to concatenate two 32 bit unsigned int
 * and return them as a 64 bit unsigned int
 */
inline uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

/**
 * Gets a non deterministic random number number from either the
 * std::random_device or the current time
 */
uint64_t getNonDeterministicRandom() {
  std::random_device rd;
  uint32_t random1;
  uint32_t random2;
  if (rd.entropy() != 0) {
    random1 = rd();
    random2 = rd();
    return make64BitsFrom32Bits(random1, random2);
  }
  else {
    random1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    random2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return make64BitsFrom32Bits(random1, random2);
  }
}

} // namespace detail

/**
 * CPUGenerator class implementation
 */
CPUGenerator::CPUGenerator(uint64_t seed_in)
  : CloneableGenerator(Device(DeviceType::CPU)),
    engine_(seed_in),
    normal_cache_index_(0) { }

CPUGenerator::CPUGenerator(mt19937 engine_in)
  : CloneableGenerator(Device(DeviceType::CPU)),
    engine_(engine_in),
    normal_cache_index_(0) { }

/**
 * Manually seeds the engine with the seed input
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_current_seed(uint64_t seed) {
  normal_cache_index_ = 0;
  engine_ = mt19937(seed);
}

/**
 * Gets the current seed of CPUGenerator.
 * 
 * See Note [Thread-safety and Generators]
 */
uint64_t CPUGenerator::current_seed() const {
  return engine_.seed();
}

/**
 * Gets the DeviceType of CPUGenerator.
 * Used for type checking during run time.
 * 
 * See Note [Thread-safety and Generators]
 */
DeviceType CPUGenerator::device_type() {
  return DeviceType::CPU;
}

/**
 * Gets a random 32 bit unsigned integer from the engine
 * 
 * See Note [Thread-safety and Generators]
 */
uint32_t CPUGenerator::random() {
  return engine_();
}

/**
 * Gets a random 64 bit unsigned integer from the engine
 * 
 * See Note [Thread-safety and Generators]
 */
uint64_t CPUGenerator::random64() {
  uint32_t random1 = engine_();
  uint32_t random2 = engine_();
  return detail::make64BitsFrom32Bits(random1, random2);
}

/**
 * Gets the current cache index of normal randoms in normal
 * distribution.
 * 
 * See Note [Thread-safety and Generators]
 */
uint32_t CPUGenerator::normal_cache_index() {
  return normal_cache_index_;
}

/**
 * Set the current cache index of normal randoms in normal
 * distribution.
 * 
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_normal_cache_index(uint32_t index) {
  normal_cache_index_ = index;
}

/**
 * Get the cached normal randoms in floats
 * 
 * See Note [Thread-safety and Generators]
 */
at::detail::Array<float, 2> CPUGenerator::normal_cache_floats() {
  return normal_cache_floats_;
}

/**
 * Get the cached normal randoms in doubles
 * 
 * See Note [Thread-safety and Generators]
 */
at::detail::Array<double, 2> CPUGenerator::normal_cache_doubles() {
  return normal_cache_doubles_;
}

/**
 * Cache normal randoms in floats
 * 
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_normal_cache_floats(at::detail::Array<float, 2> randoms) {
  normal_cache_floats_ = randoms;
}

/**
 * Cache normal randoms in doubles
 * 
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_normal_cache_doubles(at::detail::Array<double, 2> randoms) {
  normal_cache_doubles_ = randoms;
}

/**
 * Private clone method implementation
 * 
 * See Note [Thread-safety and Generators]
 */
CloneableGenerator<CPUGenerator, Generator>* CPUGenerator::clone_impl() const {
  return new CPUGenerator(engine_);
}

} // namespace at
