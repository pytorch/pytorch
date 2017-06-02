#include "CUDAGenerator.h"
#include "Context.h"
#include <stdexcept>

#define const_generator_cast(generator) \
  dynamic_cast<const CUDAGenerator&>(generator)

namespace tlib {

CUDAGenerator::CUDAGenerator(Context * context_)
  : context(context_)
{
  int num_devices, current_device;
  cudaGetDeviceCount(&num_devices);
  cudaGetDevice(&current_device);
  THCRandom_init(context->thc_state, num_devices, current_device);
}

CUDAGenerator::~CUDAGenerator() {
  THCRandom_shutdown(context->thc_state);
}

CUDAGenerator& CUDAGenerator::copy(const Generator& from) {
  throw std::runtime_error("CUDAGenerator::copy() not implemented");
}

CUDAGenerator& CUDAGenerator::free() {
  THCRandom_shutdown(context->thc_state);
  return *this;
}

unsigned long CUDAGenerator::seed() {
  return THCRandom_initialSeed(context->thc_state);
}

CUDAGenerator& CUDAGenerator::manualSeed(unsigned long seed) {
  THCRandom_manualSeed(context->thc_state, seed);
  return *this;
}

} // namespace thpp
