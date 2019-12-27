#include <ATen/CPUGenerator.h>
#include <ATen/MT19937CPUGenerator.h>
#include <c10/util/C++17.h>
#include <algorithm>

namespace at {

namespace detail {

// Ensures default_gen_cpu is initialized once.
static std::once_flag cpu_gen_init_flag;

// Default, global CPU generator.
static std::shared_ptr<CPUGenerator> default_gen_cpu;

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultCPUGenerator gets the default generator for a particular
 * device.
 */
CPUGenerator* getDefaultCPUGenerator() {
  std::call_once(cpu_gen_init_flag, [&] {
    default_gen_cpu = createCPUGenerator(getNonDeterministicRandom());
  });
  return default_gen_cpu.get();
}

/**
 * Utility to create a CPUGenerator. Returns a shared_ptr
 */
std::shared_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val) {
  return std::make_shared<MT19937CPUGenerator>(seed_val);
}

/**
 * Helper function to concatenate two 32 bit unsigned int
 * and return them as a 64 bit unsigned int
 */
inline uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

} // namespace detail

/**
 * CPUGenerator class implementation
 */
CPUGenerator::CPUGenerator(uint64_t seed_in)
  : Generator{Device(DeviceType::CPU)},
    next_float_normal_sample_{c10::optional<float>()},
    next_double_normal_sample_{c10::optional<double>()} { }

/**
 * Gets the DeviceType of CPUGenerator.
 * Used for type checking during run time.
 */
DeviceType CPUGenerator::device_type() {
  return DeviceType::CPU;
}

/**
 * Gets a random 64 bit unsigned integer from the engine
 * 
 * See Note [Acquire lock when using random generators]
 */
uint64_t CPUGenerator::random64() {
  uint32_t random1 = random();
  uint32_t random2 = random();
  return detail::make64BitsFrom32Bits(random1, random2);
}

/**
 * Get the cached normal random in float
 */
c10::optional<float> CPUGenerator::next_float_normal_sample() {
  return next_float_normal_sample_;
}

/**
 * Get the cached normal random in double
 */
c10::optional<double> CPUGenerator::next_double_normal_sample() {
  return next_double_normal_sample_;
}

/**
 * Cache normal random in float
 * 
 * See Note [Acquire lock when using random generators]
 */
void CPUGenerator::set_next_float_normal_sample(c10::optional<float> randn) {
  next_float_normal_sample_ = randn;
}

/**
 * Cache normal random in double
 * 
 * See Note [Acquire lock when using random generators]
 */
void CPUGenerator::set_next_double_normal_sample(c10::optional<double> randn) {
  next_double_normal_sample_ = randn;
}

/**
 * Public clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<CPUGenerator> CPUGenerator::clone() const {
  return std::shared_ptr<CPUGenerator>(dynamic_cast<CPUGenerator*>(this->clone_impl()));
}

} // namespace at
