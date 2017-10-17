#include "THGenerator.hpp"

#define const_generator_cast(generator) \
  dynamic_cast<const THGenerator&>(generator)

namespace thpp {

THGenerator::THGenerator()
  : generator(THGenerator_new())
{}

THGenerator::~THGenerator() {
  if (generator)
    THGenerator_free(generator);
}

THGenerator& THGenerator::copy(const Generator& from) {
  THGenerator_copy(generator, const_generator_cast(from).generator);
  return *this;
}

THGenerator& THGenerator::free() {
  THGenerator_free(generator);
  return *this;
}

uint64_t THGenerator::seed() {
  return THRandom_seed(generator);
}

THGenerator& THGenerator::manualSeed(uint64_t seed) {
  THRandom_manualSeed(generator, seed);
  return *this;
}

} // namespace thpp
