#pragma once

#include <ATen/CPUGenerator.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <c10/util/Optional.h>

#undef NAMESPACE_END

namespace at {

struct CAFFE2_API MT19937CPUGenerator : public CPUGenerator {
  // Constructors
  MT19937CPUGenerator(uint64_t seed_in = default_rng_seed_val);
  ~MT19937CPUGenerator() = default;

  // MT19937CPUGenerator methods
  std::shared_ptr<MT19937CPUGenerator> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  uint32_t random() override;
//   void getRNGState(void* target) override;
//   void setRNGState(void* target) override;

private:
  MT19937CPUGenerator* clone_impl() const override;
  at::mt19937 engine_;
};

} // namespace at
