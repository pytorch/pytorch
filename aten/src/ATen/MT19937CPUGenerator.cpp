#include <ATen/MT19937CPUGenerator.h>
#include <TH/THGenerator.hpp>
#include <TH/THTensor.h>
#include <TH/TH.h>
#include <c10/util/C++17.h>
#include <algorithm>
#include <iostream>

namespace at {

/**
 * MT19937CPUGenerator class implementation
 */
MT19937CPUGenerator::MT19937CPUGenerator(uint64_t seed_in)
  : CPUGenerator(seed_in), engine_{seed_in} {
}

/**
 * Manually seeds the engine with the seed input
 * See Note [Acquire lock when using random generators]
 */
void MT19937CPUGenerator::set_current_seed(uint64_t seed) {
  next_float_normal_sample_.reset();
  next_double_normal_sample_.reset();
  engine_ = mt19937(seed);
}

/**
 * Gets the current seed of CPUGenerator.
 */
uint64_t MT19937CPUGenerator::current_seed() const {
  return engine_.seed();
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGenerator with it and then returns that number.
 * 
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and CUDA
 */
uint64_t MT19937CPUGenerator::seed() {
  auto random = detail::getNonDeterministicRandom();
  this->set_current_seed(random);
  return random;
}

/**
 * Gets a random 32 bit unsigned integer from the engine
 *
 * See Note [Acquire lock when using random generators]
 */
uint32_t MT19937CPUGenerator::random() {
  const auto res = engine_();
  return res;
}

void MT19937CPUGenerator::getRNGState(void* target) {
  // cast byte tensor to POD type
  THGeneratorStateNew* rng_state = (THGeneratorStateNew*)target;
  static const size_t size = sizeof(THGeneratorStateNew);

  // accumulate generator data to be copied into byte tensor
  auto accum_state = std::make_unique<THGeneratorStateNew>();
  auto rng_data = engine_.data();
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
  if (next_double_normal_sample_) {
    accum_state->legacy_pod.normal_is_valid = true;
    accum_state->legacy_pod.normal_y = *next_double_normal_sample_;
  }
  if (next_float_normal_sample_) {
    accum_state->is_next_float_normal_sample_valid = true;
    accum_state->next_float_normal_sample = *next_float_normal_sample_;
  }

  memcpy(rng_state, accum_state.get(), size);
}

void MT19937CPUGenerator::setRNGState(void* target) {
  THGeneratorState* legacy_pod = (THGeneratorState*)target;
  at::mt19937 engine;
  // construct engine_
  // Note that legacy THGeneratorState stored a state array of 64 bit uints, whereas in our
  // redefined mt19937, we have changed to a state array of 32 bit uints. Hence, we are
  // doing a std::copy.
  at::mt19937_data_pod rng_data;
  std::copy(std::begin(legacy_pod->state), std::end(legacy_pod->state), rng_data.state_.begin());
  rng_data.seed_ = legacy_pod->the_initial_seed;
  rng_data.left_ = legacy_pod->left;
  rng_data.seeded_ = legacy_pod->seeded;
  rng_data.next_ = static_cast<uint32_t>(legacy_pod->next);
  engine.set_data(rng_data);
  THArgCheck(engine.is_valid(), 1, "Invalid mt19937 state");
  engine_ = engine;
}

/**
 * Public clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<MT19937CPUGenerator> MT19937CPUGenerator::clone() const {
  return std::shared_ptr<MT19937CPUGenerator>(this->clone_impl());
}

/**
 * Private clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
MT19937CPUGenerator* MT19937CPUGenerator::clone_impl() const {
  auto gen = new MT19937CPUGenerator();
  gen->engine_ = engine_;
  gen->set_next_float_normal_sample(next_float_normal_sample_);
  gen->set_next_double_normal_sample(next_double_normal_sample_);
  return gen;
}

} // namespace at
