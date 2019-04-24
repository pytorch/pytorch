#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <c10/util/Optional.h>

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
  c10::optional<float> next_float_normal_sample();
  c10::optional<double> next_double_normal_sample();
  void set_next_float_normal_sample(c10::optional<float> randn);
  void set_next_double_normal_sample(c10::optional<double> randn);
  at::mt19937 engine();
  void set_engine(at::mt19937 engine);

private:
  CloneableGenerator<CPUGenerator, Generator>* clone_impl() const override;
  at::mt19937 engine_;
  c10::optional<float> next_float_normal_sample_;
  c10::optional<double> next_double_normal_sample_;
};

namespace detail {

CAFFE2_API CPUGenerator* getDefaultCPUGenerator();
CAFFE2_API std::unique_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val = default_rng_seed_val);
CAFFE2_API uint64_t getNonDeterministicRandom();

} // namespace detail

}
