#pragma once

#include <THC/THC.h>

#include "ATen/Generator.h"

namespace at {

class Context;
struct CUDAGenerator : public Generator {
  CUDAGenerator(Context * context);
  virtual ~CUDAGenerator();

  virtual CUDAGenerator& copy(const Generator& from) override;
  virtual CUDAGenerator& free() override;

  virtual uint64_t seed() override;
  virtual uint64_t initialSeed() override;
  virtual CUDAGenerator& manualSeed(uint64_t seed) override;
  virtual void * unsafeGetTH() override;

//TODO(zach): figure out friends later
public:
  Context * context;
  THCGenerator * generator;
};

}
