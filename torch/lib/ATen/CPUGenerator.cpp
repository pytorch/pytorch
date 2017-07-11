#include "ATen/CPUGenerator.h"

#define const_generator_cast(generator) \
  dynamic_cast<const CPUGenerator&>(generator)

namespace at {

CPUGenerator::CPUGenerator(Context * context_)
  : context(context_), generator(THGenerator_new())
{}

CPUGenerator::~CPUGenerator() {
  if (generator)
    THGenerator_free(generator);
}

CPUGenerator& CPUGenerator::copy(const Generator& from) {
  THGenerator_copy(generator, const_generator_cast(from).generator);
  return *this;
}

CPUGenerator& CPUGenerator::free() {
  THGenerator_free(generator);
  return *this;
}

unsigned long CPUGenerator::seed() {
  return THRandom_seed(generator);
}

CPUGenerator& CPUGenerator::manualSeed(unsigned long seed) {
  THRandom_manualSeed(generator, seed);
  return *this;
}

} // namespace at
