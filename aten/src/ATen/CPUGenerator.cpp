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
  return c10::guts::make_unique<CPUGenerator>(seed_val);
}

} // namespace detail

/**
 * Utility to seed a mersenne twister engine
 * See: https://kristerw.blogspot.com/2017/05/seeding-stdmt19937-random-number-engine.html
 * and http://www.pcg-random.org/posts/cpp-seeding-surprises.html
 */
// std::mt19937 seedMersenneTwisterEngine(uint64_t seed_in) {
//   Philox4_32_10 philox_engine(seed_in);
//   std::array<int, std::mt19937::state_size> seed_data;
//   std::generate_n(seed_data.data(), seed_data.size(), std::ref(philox_engine));
//   std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
//   return std::mt19937(seq);
// }

/**
 * CPUGenerator class implementation
 */
CPUGenerator::CPUGenerator(uint64_t seed_in)
  : CloneableGenerator(Device(DeviceType::CPU)),
    current_seed_(seed_in),
    engine_(seed_in) { }

/**
 * Manually seeds the engine with the seed input
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_current_seed(uint64_t seed) {
  current_seed_ = seed;
  engine_ = mt19937(seed);
}

/**
 * Gets the current seed of CPUGenerator.
 * 
 * See Note [Thread-safety and Generators]
 */
uint64_t CPUGenerator::current_seed() const {
  return current_seed_;
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
  return new CPUGenerator(current_seed_);
}

} // namespace at
