#pragma once

#include <$header>

#include "ATen/Generator.h"

namespace at {

class Context;
struct ${name}Generator : public Generator {
  ${name}Generator(Context * context);
  virtual ~${name}Generator();

  virtual ${name}Generator& copy(const Generator& from) override;
  virtual ${name}Generator& free() override;

  virtual uint64_t seed() override;
  virtual ${name}Generator& manualSeed(uint64_t seed) override;

//TODO(zach): figure out friends later
public:
  Context * context;
  ${th_generator}
};

}
