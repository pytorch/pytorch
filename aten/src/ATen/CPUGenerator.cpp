#include <ATen/CPUGenerator.h>
#include <c10/util/C++17.h>
#include <algorithm>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

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
uint64_t make64BitsFrom32Bits(uint32_t a, uint32_t b) {
  uint64_t hi = static_cast<uint64_t>(a) << 32;
  uint64_t lo = static_cast<uint64_t>(b);
  return hi | lo;
}

#ifndef _WIN32
static uint64_t readURandomLong() {
  int randDev = open("/dev/urandom", O_RDONLY);
  uint64_t randValue;
  if (randDev < 0) {
    AT_ASSERT("Unable to open /dev/urandom");
  }
  ssize_t readBytes = read(randDev, &randValue, sizeof(randValue));
  if (readBytes < (ssize_t) sizeof(randValue)) {
    AT_ASSERT("Unable to read from /dev/urandom");
  }
  close(randDev);
  return randValue;
}
#endif // _WIN32

/**
 * Gets a non deterministic random number number from either the
 * std::random_device or the current time
 */
uint32_t getNonDeterministicRandom() {
  #ifdef _WIN32
  uint64_t s = (uint64_t)time(0);
  #else
  uint64_t s = readURandomLong();
  #endif
  return s;
}

} // namespace detail

/**
 * CPUGenerator class implementation
 */
CPUGenerator::CPUGenerator(uint64_t seed_in)
  : CloneableGenerator(Device(DeviceType::CPU)),
    engine_(seed_in) { }

CPUGenerator::CPUGenerator(mt19937 engine_in)
  : CloneableGenerator(Device(DeviceType::CPU)),
    engine_(engine_in) { }

/**
 * Manually seeds the engine with the seed input
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_current_seed(uint64_t seed) {
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
  return detail::make64BitsFrom32Bits(engine_(), engine_());
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
