#include "THCGenerator.hpp"

#define const_generator_cast(generator) \
  dynamic_cast<const THCGenerator&>(generator)

namespace thpp {

THCGenerator::THCGenerator(THCState* state))
  : generator(THCRandom_getGenerator())
  , state(state)
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

unsigned long THCGenerator::seed() {
  return THCRandom_initialSeed(state);
}

THCGenerator& THCGenerator::manualSeed(unsigned long seed) {
  THCRandom_manualSeed(state, seed);
  return *this;
}

} // namespace thpp
