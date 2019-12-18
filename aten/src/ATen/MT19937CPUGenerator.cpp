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
  std::cout << "MT19937CPUGenerator ctor()" << std::endl;
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
  std::cout << "MT19937CPUGenerator::random() generated " << res << std::endl;
  return res;
}

// void MT19937CPUGenerator::getRNGState(void* target) {
//     // TODO
// }

// void MT19937CPUGenerator::setRNGState(void* target) {
//     // TODO
// }

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
  gen->set_engine(engine_);
  gen->set_next_float_normal_sample(next_float_normal_sample_);
  gen->set_next_double_normal_sample(next_double_normal_sample_);
  return gen;
}

} // namespace at
