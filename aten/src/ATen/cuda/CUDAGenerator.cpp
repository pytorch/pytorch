#include <ATen/Config.h>

#include <ATen/CUDAGenerator.h>
#include <ATen/Context.h>
#include <THC/THCTensorRandom.h>
#include <stdexcept>

// There is only one CUDAGenerator instance. Calls to seed(), manualSeed(),
// initialSeed(), and unsafeGetTH() refer to the THCGenerator on the current
// device.

THCGenerator* THCRandom_getGenerator(THCState* state);

namespace at {

CUDAGenerator::CUDAGenerator(Context * context_)
  : CloneableGenerator(Device(DeviceType::CUDA)), context(context_)
{
}

CUDAGenerator::~CUDAGenerator() {
  // no-op Generator state is global to the program
}

CUDAGenerator& CUDAGenerator::copy(const Generator& from) {
  throw std::runtime_error("CUDAGenerator::copy() not implemented");
}

CUDAGenerator& CUDAGenerator::free() {
  THCRandom_shutdown(context->getTHCState());
  return *this;
}

uint64_t CUDAGenerator::seed() {
  return THCRandom_initialSeed(context->getTHCState());
}

uint64_t CUDAGenerator::getCurrentSeed() const {
  return THCRandom_initialSeed(context->getTHCState());
}

void CUDAGenerator::setCurrentSeed(uint64_t seed) {
  THCRandom_manualSeed(context->getTHCState(), seed);
}

CUDAGenerator& CUDAGenerator::manualSeedAll(uint64_t seed) {
  THCRandom_manualSeedAll(context->getTHCState(), seed);
  return *this;
}

void * CUDAGenerator::unsafeGetTH() {
  return (void*)THCRandom_getGenerator(context->getTHCState());
}

CloneableGenerator<CUDAGenerator, Generator>* CUDAGenerator::clone_impl() const {
  return new CUDAGenerator(context);
}

} // namespace at
