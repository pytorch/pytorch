#pragma once

#include <TH/TH.h>

#include "../Generator.hpp"
#include "../tensors/THTensor.hpp"

namespace thpp {

struct th_generator_traits {
  using generator_type = THGenerator;
};

} // namespace thpp

namespace thpp {

struct THGenerator : public Generator {
  template<typename U>
  friend struct THTensor;

  using generator_type = th_generator_traits::generator_type;

  THGenerator();
  virtual ~THGenerator();

  virtual THGenerator& copy(const Generator& from) override;
  virtual THGenerator& free() override;

  virtual uint64_t seed() override;
  virtual THGenerator& manualSeed(uint64_t seed) override;

protected:
  generator_type *generator;
};

}
