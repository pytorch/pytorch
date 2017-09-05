#pragma once

#include <THC/THC.h>

#include "../Generator.hpp"

namespace thpp {

struct THCGenerator : public Generator {
  THCGenerator(THCState* state);
  virtual ~THCGenerator();

  virtual THCGenerator& copy(const Generator& from) override;
  virtual THCGenerator& free() override;

  virtual unsigned long seed() override;
  virtual THCGenerator& manualSeed(unsigned long seed) override;

protected:
  THCState *state;
};

}
