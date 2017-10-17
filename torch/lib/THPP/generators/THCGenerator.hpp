#pragma once

#include <THC/THC.h>

#include "../Generator.hpp"

namespace thpp {

struct THCGenerator : public Generator {
  THCGenerator(THCState* state);
  virtual ~THCGenerator();

  virtual THCGenerator& copy(const Generator& from) override;
  virtual THCGenerator& free() override;

  virtual uint64_t seed() override;
  virtual THCGenerator& manualSeed(uint64_t seed) override;

protected:
  THCState *state;
};

}
