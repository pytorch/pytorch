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
    is_normal_cache_available_(false) { }

CPUGenerator::CPUGenerator(mt19937 engine_in)
  : CloneableGenerator(Device(DeviceType::CPU)),
    engine_(engine_in),
    is_normal_cache_available_(false) { }

/**
 * Manually seeds the engine with the seed input
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_current_seed(uint64_t seed) {
  is_normal_cache_available_ = false;
  engine_ = mt19937(seed);
}

/**
 * Gets the current seed of CPUGenerator.
 */
uint64_t CPUGenerator::current_seed() const {
  return engine_.seed();
}

/**
 * Gets the DeviceType of CPUGenerator.
 * Used for type checking during run time.
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
bool CPUGenerator::is_normal_cache_available() {
  return is_normal_cache_available_;
}

/**
 * Get the cached normal random in float
 * 
 * See Note [Thread-safety and Generators]
 */
float CPUGenerator::normal_cache_float() {
  is_normal_cache_available_ = false;
  return normal_cache_float_;
}

/**
 * Get the cached normal random in double
 * 
 * See Note [Thread-safety and Generators]
 */
double CPUGenerator::normal_cache_double() {
  is_normal_cache_available_ = false;
  return normal_cache_double_;
}

/**
 * Cache normal random in float
 * 
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_normal_cache_float(float randn) {
  normal_cache_float_ = randn;
  is_normal_cache_available_ = true;
}

/**
 * Cache normal random in double
 * 
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_normal_cache_double(double randn) {
  normal_cache_double_ = randn;
  is_normal_cache_available_ = true;
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
