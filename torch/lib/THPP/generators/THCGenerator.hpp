#pragma once

#include <THC/THC.h>

namespace thpp {

struct thc_generator_traits {
  using generator_type = Generator;
};

} // namespace thpp

#include "../Generator.hpp"

namespace thpp {

template<typename real>
struct THCGenerator : public Generator {
  using generator_type = typename thc_generator_traits::generator_type;

  THCGenerator(THCState* state);
  virtual ~THCGenerator();

  virtual THCGenerator& copy(const Generator& from) override;
  virtual THCGenerator& free() override;

  virtual THCGenerator& seed() override;
  virtual THCGenerator& manualSeed(unsigned long seed) override;

protected:
  generator_type *generator;
  THCState *state;
};

}
