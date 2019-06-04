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
CPUGenerator* getDefaultCPUGenerator() {
  std::call_once(cpu_device_flag, [&] {
    default_gen_cpu = c10::guts::make_unique<CPUGenerator>(getNonDeterministicRandom());
  });
  return default_gen_cpu.get();
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
 * Gets a random number for /dev/urandom
 * Note this is a legacy method (from THRandom.cpp)
 */
#ifndef _WIN32
static uint64_t readURandomLong()
{
  int randDev = open("/dev/urandom", O_RDONLY);
  uint64_t randValue;
  TORCH_CHECK(randDev >= 0, "Unable to open /dev/urandom");
  ssize_t readBytes = read(randDev, &randValue, sizeof(randValue));
  TORCH_CHECK(readBytes >= (ssize_t) sizeof(randValue), "Unable to read from /dev/urandom");
  close(randDev);
  return randValue;
}
#endif // _WIN32

/**
 * Gets a non deterministic random number number from either the
 * /dev/urandom or the current time
 * FIXME: The behavior in this function is from legacy code (THRandom_seed)
 * and is probably not the right thing to do, even though our tests pass.
 * Figure out if tests get perturbed when using C++11 std objects, such as
 * std::random_device and chrono.
 */
uint64_t getNonDeterministicRandom() {
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
    engine_(seed_in),
    next_float_normal_sample_(c10::optional<float>()),
    next_double_normal_sample_(c10::optional<double>()) { }

/**
 * Manually seeds the engine with the seed input
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_current_seed(uint64_t seed) {
  next_float_normal_sample_.reset();
  next_double_normal_sample_.reset();
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
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_next_float_normal_sample(c10::optional<float> randn) {
  next_float_normal_sample_ = randn;
}

/**
 * Cache normal random in double
 * 
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_next_double_normal_sample(c10::optional<double> randn) {
  next_double_normal_sample_ = randn;
}

/**
 * Get the engine of the CPUGenerator
 */
at::mt19937 CPUGenerator::engine() {
  return engine_;
}

/**
 * Set the engine of the CPUGenerator
 * 
 * See Note [Thread-safety and Generators]
 */
void CPUGenerator::set_engine(at::mt19937 engine) {
  engine_ = engine;
}

/**
 * Private clone method implementation
 * 
 * See Note [Thread-safety and Generators]
 */
CloneableGenerator<CPUGenerator, Generator>* CPUGenerator::clone_impl() const {
  auto gen = new CPUGenerator();
  gen->set_engine(engine_);
  gen->set_next_float_normal_sample(next_float_normal_sample_);
  gen->set_next_double_normal_sample(next_double_normal_sample_);
  return gen;
}

} // namespace at
