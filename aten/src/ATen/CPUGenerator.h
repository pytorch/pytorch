#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <ATen/core/PhiloxRNGEngine.h>

namespace at {

struct CAFFE2_API CPUGenerator : public CloneableGenerator<CPUGenerator, Generator> {
  // Constructors
  CPUGenerator(uint64_t seed_in = default_rng_seed_val);
  CPUGenerator(mt19937 engine_in);

  // CPUGenerator methods
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  static DeviceType device_type();
  uint32_t random();
  uint64_t random64();
  bool is_normal_cache_available();
  float normal_cache_float();
  double normal_cache_double();
  void set_normal_cache_float(float randn);
  void set_normal_cache_double(double randn);

private:
  CloneableGenerator<CPUGenerator, Generator>* clone_impl() const override;
  at::mt19937 engine_;
  bool is_normal_cache_available_;
  float normal_cache_float_;
  double normal_cache_double_;
};

namespace detail {

CAFFE2_API std::unique_ptr<CPUGenerator>& getDefaultCPUGenerator();
CAFFE2_API std::unique_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val = default_rng_seed_val);
CAFFE2_API uint64_t getNonDeterministicRandom();

} // namespace detail

}
