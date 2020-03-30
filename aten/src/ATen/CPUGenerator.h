#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <c10/util/Optional.h>
#include <c10/core/GeneratorImpl.h>

namespace at {

struct CAFFE2_API CPUGenerator : public c10::GeneratorImpl {
  // Constructors
  CPUGenerator(uint64_t seed_in = default_rng_seed_val);
  ~CPUGenerator() = default;

  // CPUGenerator methods
  std::shared_ptr<CPUGenerator> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
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
  CPUGenerator* clone_impl() const override;
  at::mt19937 engine_;
  c10::optional<float> next_float_normal_sample_;
  c10::optional<double> next_double_normal_sample_;
};

namespace detail {

CAFFE2_API const Generator& getDefaultCPUGenerator();
CAFFE2_API Generator createCPUGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail

}
