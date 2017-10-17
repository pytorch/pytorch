#include "THCGenerator.hpp"

#include <stdexcept>

#define const_generator_cast(generator) \
  dynamic_cast<const THCGenerator&>(generator)

namespace thpp {

THCGenerator::THCGenerator(THCState* state)
  : state(state)
{
  int num_devices, current_device;
  cudaGetDeviceCount(&num_devices);
  cudaGetDevice(&current_device);
  THCRandom_init(state, num_devices, current_device);
}

THCGenerator::~THCGenerator() {
  THCRandom_shutdown(state);
}

THCGenerator& THCGenerator::copy(const Generator& from) {
  throw std::runtime_error("THCGenerator::copy() not implemented");
}

THCGenerator& THCGenerator::free() {
  THCRandom_shutdown(state);
  return *this;
}

uint64_t THCGenerator::seed() {
  return THCRandom_initialSeed(state);
}

THCGenerator& THCGenerator::manualSeed(uint64_t seed) {
  THCRandom_manualSeed(state, seed);
  return *this;
}

} // namespace thpp
