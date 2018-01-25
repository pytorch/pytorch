#include "ATen/Config.h"

#if AT_CUDA_ENABLED()

#include "ATen/CUDAGenerator.h"
#include "ATen/Context.h"
#include "THCTensorRandom.h"
#include <stdexcept>

#define const_generator_cast(generator) \
  dynamic_cast<const CUDAGenerator&>(generator)

THCGenerator* THCRandom_getGenerator(THCState* state);

namespace at {

CUDAGenerator::CUDAGenerator(Context * context_)
  : context(context_)
{
  // there's no reason to call THCRandom_init, because it is called
  // during THCudaInit, which is called before this initializer
  generator = THCRandom_getGenerator(context->thc_state);
}

CUDAGenerator::~CUDAGenerator() {
  // no-op Generator state is global to the program
}

CUDAGenerator& CUDAGenerator::copy(const Generator& from) {
  throw std::runtime_error("CUDAGenerator::copy() not implemented");
}

CUDAGenerator& CUDAGenerator::free() {
  THCRandom_shutdown(context->thc_state);
  return *this;
}

uint64_t CUDAGenerator::seed() {
  return THCRandom_initialSeed(context->thc_state);
}

uint64_t CUDAGenerator::initialSeed() {
  return THCRandom_initialSeed(context->thc_state);
}

CUDAGenerator& CUDAGenerator::manualSeed(uint64_t seed) {
  THCRandom_manualSeed(context->thc_state, seed);
  return *this;
}

void * CUDAGenerator::unsafeGetTH() {
  return (void *) generator;
}

} // namespace at
#endif //AT_CUDA_ENABLED
