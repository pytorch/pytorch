#pragma once

#include <ATen/core/Generator.h>
#include <c10/util/Optional.h>

namespace at {

struct CAFFE2_API CPUGenerator : public Generator {
  // Constructors
  CPUGenerator(uint64_t seed_in = default_rng_seed_val);
  virtual ~CPUGenerator() = default;

  // CPUGenerator methods
  std::shared_ptr<CPUGenerator> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  static DeviceType device_type();
  virtual uint32_t random() = 0;
  virtual uint64_t random64();
  c10::optional<float> next_float_normal_sample();
  c10::optional<double> next_double_normal_sample();
  void set_next_float_normal_sample(c10::optional<float> randn);
  void set_next_double_normal_sample(c10::optional<double> randn);
  virtual void getRNGState(void* target) = 0;
  virtual void setRNGState(void* target) = 0;

protected:
  virtual CPUGenerator* clone_impl() const override = 0;
  c10::optional<float> next_float_normal_sample_;
  c10::optional<double> next_double_normal_sample_;
};

namespace detail {

CAFFE2_API CPUGenerator* getDefaultCPUGenerator();
CAFFE2_API std::shared_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail

}
