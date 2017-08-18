#pragma once

namespace at {

struct Generator {
  Generator() {};
  Generator(const Generator& other) = delete;
  Generator(Generator&& other) = delete;
  virtual ~Generator() {};

  virtual Generator& copy(const Generator& other) = 0;
  virtual Generator& free() = 0;

  virtual unsigned long seed() = 0;
  virtual Generator& manualSeed(unsigned long seed) = 0;
};

} // namespace at
