#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Utils.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <c10/util/MathConstants.h>
#include <algorithm>

namespace at {

namespace detail {

/**
 * CPUGeneratorImplStateLegacy is a POD class needed for memcpys
 * in torch.get_rng_state() and torch.set_rng_state().
 * It is a legacy class and even though it is replaced with
 * at::CPUGeneratorImpl, we need this class and some of its fields
 * to support backward compatibility on loading checkpoints.
 */
struct CPUGeneratorImplStateLegacy {
  /* The initial seed. */
  uint64_t the_initial_seed;
  int left;  /* = 1; */
  int seeded; /* = 0; */
  uint64_t next;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  uint64_t state[at::MERSENNE_STATE_N]; /* the array for the state vector  */

  /********************************/

  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
};

/**
 * CPUGeneratorImplState is a POD class containing
 * new data introduced in at::CPUGeneratorImpl and the legacy state. It is used
 * as a helper for torch.get_rng_state() and torch.set_rng_state()
 * functions.
 */
struct CPUGeneratorImplState {
  CPUGeneratorImplStateLegacy legacy_pod;
  float next_float_normal_sample;
  bool is_next_float_normal_sample_valid;
};

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
inline static uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

} // namespace detail

/**
 * CPUGeneratorImpl class implementation
 */
CPUGeneratorImpl::CPUGeneratorImpl(uint64_t seed_in)
  : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(c10::DispatchKey::CPU)},
    engine_{seed_in},
    next_float_normal_sample_{std::optional<float>()},
    next_double_normal_sample_{std::optional<double>()} { }

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
 * Sets the offset of RNG state.
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_offset(uint64_t offset [[maybe_unused]]) {
  TORCH_CHECK(false, "CPU Generator does not use offset");
}

/**
 * Gets the current offset of CPUGeneratorImpl.
 */
uint64_t CPUGeneratorImpl::get_offset() const {
  TORCH_CHECK(false, "CPU Generator does not use offset");
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
 * Sets the internal state of CPUGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and of the same size as CPUGeneratorImplState.
 */
void CPUGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  using detail::CPUGeneratorImplState;
  using detail::CPUGeneratorImplStateLegacy;

  static_assert(std::is_standard_layout_v<CPUGeneratorImplState>, "CPUGeneratorImplState is not a PODType");
  constexpr size_t size = sizeof(CPUGeneratorImplState);

  detail::check_rng_state(new_state);

  at::mt19937 engine;
  auto new_state_size = new_state.numel();

  TORCH_CHECK(new_state_size == size, "Expected a CPUGeneratorImplState of size ", size,
            " but found the input RNG state size to be ", new_state_size);

  auto rng_state = new_state.data_ptr_impl<CPUGeneratorImplState>();
  auto legacy_pod = &(rng_state->legacy_pod);
  // construct engine_
  // Note that CPUGeneratorImplStateLegacy stored a state array of 64 bit uints, whereas in our
  // redefined mt19937, we have changed to a state array of 32 bit uints. Hence, we are
  // doing a std::copy.
  at::mt19937_data_pod rng_data{};
  std::copy(std::begin(legacy_pod->state), std::end(legacy_pod->state), rng_data.state_.begin());
  rng_data.seed_ = legacy_pod->the_initial_seed;
  rng_data.left_ = legacy_pod->left;
  rng_data.seeded_ = legacy_pod->seeded;
  rng_data.next_ = static_cast<uint32_t>(legacy_pod->next);
  engine.set_data(rng_data);
  TORCH_CHECK(engine.is_valid(), "Invalid mt19937 state");
  this->engine_ = engine;
  this->next_float_normal_sample_ = rng_state->is_next_float_normal_sample_valid
      ? std::optional<float>(rng_state->next_float_normal_sample)
      : std::optional<float>();
  this->next_double_normal_sample_ = legacy_pod->normal_is_valid
      ? std::optional<double>(legacy_pod->normal_y)
      : std::optional<double>();
}

/**
 * Gets the current internal state of CPUGeneratorImpl. The internal
 * state is returned as a CPU byte tensor.
 */
c10::intrusive_ptr<c10::TensorImpl> CPUGeneratorImpl::get_state() const {
  using detail::CPUGeneratorImplState;

  static const size_t size = sizeof(CPUGeneratorImplState);
  static_assert(std::is_standard_layout_v<CPUGeneratorImplState>, "CPUGeneratorImplState is not a PODType");

  auto state_tensor = at::detail::empty_cpu({static_cast<int64_t>(size)}, ScalarType::Byte, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  auto rng_state = state_tensor.data_ptr();

  // accumulate generator data to be copied into byte tensor
  auto accum_state = std::make_unique<CPUGeneratorImplState>();
  auto rng_data = this->engine_.data();
  accum_state->legacy_pod.the_initial_seed = rng_data.seed_;
  accum_state->legacy_pod.left = rng_data.left_;
  accum_state->legacy_pod.seeded = rng_data.seeded_;
  accum_state->legacy_pod.next = rng_data.next_;
  std::copy(rng_data.state_.begin(), rng_data.state_.end(), std::begin(accum_state->legacy_pod.state));
  accum_state->legacy_pod.normal_x = 0.0; // we don't use it anymore and this is just a dummy
  accum_state->legacy_pod.normal_rho = 0.0; // we don't use it anymore and this is just a dummy
  accum_state->legacy_pod.normal_is_valid = false;
  accum_state->legacy_pod.normal_y = 0.0;
  accum_state->next_float_normal_sample = 0.0f;
  accum_state->is_next_float_normal_sample_valid = false;
  if (this->next_double_normal_sample_) {
    accum_state->legacy_pod.normal_is_valid = true;
    accum_state->legacy_pod.normal_y = *(this->next_double_normal_sample_);
  }
  if (this->next_float_normal_sample_) {
    accum_state->is_next_float_normal_sample_valid = true;
    accum_state->next_float_normal_sample = *(this->next_float_normal_sample_);
  }

  memcpy(rng_state, accum_state.get(), size);
  return state_tensor.getIntrusivePtr();
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
std::optional<float> CPUGeneratorImpl::next_float_normal_sample() {
  return next_float_normal_sample_;
}

/**
 * Get the cached normal random in double
 */
std::optional<double> CPUGeneratorImpl::next_double_normal_sample() {
  return next_double_normal_sample_;
}

/**
 * Cache normal random in float
 *
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_next_float_normal_sample(std::optional<float> randn) {
  next_float_normal_sample_ = randn;
}

/**
 * Cache normal random in double
 *
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_next_double_normal_sample(std::optional<double> randn) {
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
