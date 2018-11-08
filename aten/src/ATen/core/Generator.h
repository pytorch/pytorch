#pragma once

#include <ATen/core/ATenGeneral.h>
#include <stdint.h>

namespace at {

struct CAFFE2_API Generator {
  Generator() {};
  Generator(const Generator& other) = delete;
  Generator(Generator&& other) = delete;
  virtual ~Generator() {};

  virtual Generator& copy(const Generator& other) = 0;
  virtual Generator& free() = 0;

  virtual uint64_t seed() = 0;
  virtual uint64_t initialSeed() = 0;
  virtual Generator& manualSeed(uint64_t seed) = 0;
  virtual Generator& manualSeedAll(uint64_t seed) = 0;
  virtual void * unsafeGetTH() = 0;
};

} // namespace at
