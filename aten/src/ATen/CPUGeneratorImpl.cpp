#include <ATen/CPUGeneratorImpl.h>
#include <c10/util/C++17.h>
#include <algorithm>

namespace at {

namespace detail {

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultCPUGenerator gets the default generator for a particular
 * device.
 */
const Generator& getDefaultCPUGenerator() {
  static auto default_gen_cpu = createCPUGenerator(c10::detail::getNonDeterministicRandom());
  return default_gen_cpu;
}

/**
 * Utility to create a CPUGeneratorImpl. Returns a shared_ptr
 */
Generator createCPUGenerator(uint64_t seed_val) {
  return make_generator<CPUGeneratorImpl>(seed_val);
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
 * CPUGeneratorImpl class implementation
 */
CPUGeneratorImpl::CPUGeneratorImpl(uint64_t seed_in)
  : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(c10::DispatchKey::CPU)},
    engine_{seed_in},
    next_float_normal_sample_{c10::optional<float>()},
    next_double_normal_sample_{c10::optional<double>()} { }

/**
 * Manually seeds the engine with the seed input
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_current_seed(uint64_t seed) {
  next_float_normal_sample_.reset();
  next_double_normal_sample_.reset();
  engine_ = mt19937(seed);
}

/**
 * Gets the current seed of CPUGeneratorImpl.
 */
uint64_t CPUGeneratorImpl::current_seed() const {
  return engine_.seed();
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and CUDA
 */
uint64_t CPUGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom();
  this->set_current_seed(random);
  return random;
}

/**
 * Gets the DeviceType of CPUGeneratorImpl.
 * Used for type checking during run time.
 */
DeviceType CPUGeneratorImpl::device_type() {
  return DeviceType::CPU;
}

/**
 * Gets a random 32 bit unsigned integer from the engine
 *
 * See Note [Acquire lock when using random generators]
 */
uint32_t CPUGeneratorImpl::random() {
  return engine_();
}

/**
 * Gets a random 64 bit unsigned integer from the engine
 *
 * See Note [Acquire lock when using random generators]
 */
uint64_t CPUGeneratorImpl::random64() {
  uint32_t random1 = engine_();
  uint32_t random2 = engine_();
  return detail::make64BitsFrom32Bits(random1, random2);
}

/**
 * Get the cached normal random in float
 */
c10::optional<float> CPUGeneratorImpl::next_float_normal_sample() {
  return next_float_normal_sample_;
}

/**
 * Get the cached normal random in double
 */
c10::optional<double> CPUGeneratorImpl::next_double_normal_sample() {
  return next_double_normal_sample_;
}

/**
 * Cache normal random in float
 *
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_next_float_normal_sample(c10::optional<float> randn) {
  next_float_normal_sample_ = randn;
}

/**
 * Cache normal random in double
 *
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_next_double_normal_sample(c10::optional<double> randn) {
  next_double_normal_sample_ = randn;
}

/**
 * Get the engine of the CPUGeneratorImpl
 */
at::mt19937 CPUGeneratorImpl::engine() {
  return engine_;
}

/**
 * Set the engine of the CPUGeneratorImpl
 *
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_engine(at::mt19937 engine) {
  engine_ = engine;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<CPUGeneratorImpl> CPUGeneratorImpl::clone() const {
  return std::shared_ptr<CPUGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
CPUGeneratorImpl* CPUGeneratorImpl::clone_impl() const {
  auto gen = new CPUGeneratorImpl();
  gen->set_engine(engine_);
  gen->set_next_float_normal_sample(next_float_normal_sample_);
  gen->set_next_double_normal_sample(next_double_normal_sample_);
  return gen;
}

} // namespace at
