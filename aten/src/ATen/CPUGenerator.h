#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <ATen/core/PhiloxRNGEngine.h>

namespace at {

struct CAFFE2_API CPUGenerator : public CloneableGenerator<CPUGenerator, Generator> {
  // Constructors
  CPUGenerator(uint64_t seed_in = default_rng_seed_val);

  // CPUGenerator methods
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  static DeviceType device_type();
  uint32_t random();
  uint64_t random64();

private:
  uint64_t current_seed_;
  at::mt19937 engine_;
  CloneableGenerator<CPUGenerator, Generator>* clone_impl() const override;
};

namespace detail {

CAFFE2_API std::unique_ptr<CPUGenerator>& getDefaultCPUGenerator();
CAFFE2_API std::unique_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail

}
