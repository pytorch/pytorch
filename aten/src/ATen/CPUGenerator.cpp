#include <ATen/CPUGenerator.h>
#include <c10/util/C++17.h>
#include <algorithm>

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
  return c10::guts::make_unique<CPUGenerator>(seed_val);
}

} // namespace detail

/**
 * Defines mersenne twister seeding mechanism.
 * 
 * Our algorithm for seeding mt19937 is:
 *  - Take uint64_t seed value from the user and get three uint64_t-s
 *    from a splitmix64 generator
 *  - Give the three splitmix64 randoms to a Philox PRNG and get 
 *    MERSENNE_STATE_N*INIT_KEY_MULTIPLIER randoms from it
 *  - Use the philox randoms array to initialize mt19937.
 * 
 * Details of initialization problems are described in this research
 * paper: https://dl.acm.org/citation.cfm?id=1276928.
 * 
 * Moreover, Vigna also suggests that "initialization must be performed with a 
 * generator radically different in nature from the one initialized to avoid 
 * correlation on similar seeds": http://xoshiro.di.unimi.it/
 */

at::mt19937 seedMersenneTwisterEngine(uint64_t seed_in) {
  auto splitmix64 = [&seed_in]() {
    uint64_t z = (seed_in += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  };
  
  uint64_t philox_seed = splitmix64();
  uint64_t philox_subsequence = splitmix64();
  uint64_t philox_offset = splitmix64();
  Philox4_32_10 philox_engine(philox_seed, philox_subsequence, philox_offset);
  // get 3x the amount of seed data needed for better state initialization
  std::array<uint32_t, MERSENNE_STATE_N*INIT_KEY_MULTIPLIER> seed_data;
  std::generate_n(seed_data.data(), seed_data.size(), std::ref(philox_engine));
  return at::mt19937(seed_in, seed_data);
}

/**
 * CPUGenerator class implementation
 */
CPUGenerator::CPUGenerator(uint64_t seed_in)
  : CloneableGenerator(Device(DeviceType::CPU)),
    engine_(seedMersenneTwisterEngine(seed_in)) { }

CPUGenerator::CPUGenerator(mt19937 engine_in)
  : CloneableGenerator(Device(DeviceType::CPU)),
    engine_(engine_in) { }

/**
 * Manually seeds the engine with the seed input
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_current_seed(uint64_t seed) {
  engine_ = mt19937(seedMersenneTwisterEngine(seed));
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
  uint64_t hi = static_cast<uint64_t>(engine_()) << 32;
  uint64_t lo = static_cast<uint64_t>(engine_());
  return hi | lo;
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
