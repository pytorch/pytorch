#pragma once

#include <TH/TH.h>

#include "ATen/Generator.h"

namespace at {

class Context;
struct CPUGenerator : public Generator {
  CPUGenerator(Context * context);
  virtual ~CPUGenerator();

  virtual CPUGenerator& copy(const Generator& from) override;
  virtual CPUGenerator& free() override;

  virtual uint64_t seed() override;
  virtual uint64_t initialSeed() override;
  virtual CPUGenerator& manualSeed(uint64_t seed) override;
  virtual void * unsafeGetTH() override;

//TODO(zach): figure out friends later
public:
  Context * context;
  THGenerator * generator;
};

}
