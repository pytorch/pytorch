#pragma once

#include <ATen/core/Generator.h>
#include <c10/util/Optional.h>

namespace at {

struct CAFFE2_API CPUGenerator : public Generator {
  // Constructors
  CPUGenerator(uint64_t seed_in = default_rng_seed_val);
  virtual ~CPUGenerator() = default;

  // CPUGenerator methods
  static DeviceType device_type();
  std::shared_ptr<CPUGenerator> clone() const;
  c10::optional<float> next_float_normal_sample();
  c10::optional<double> next_double_normal_sample();
  void set_next_float_normal_sample(c10::optional<float> randn);
  void set_next_double_normal_sample(c10::optional<double> randn);

  virtual void set_current_seed(uint64_t seed) = 0;
  virtual uint64_t current_seed() const = 0;
  virtual uint64_t seed() = 0;
  virtual uint32_t random() = 0;
  virtual uint64_t random64();
  virtual void get_rng_state(void* target, size_t size) = 0;
  virtual void set_rng_state(void* target, size_t size) = 0;
  virtual size_t get_rng_state_size() = 0;

protected:
  c10::optional<float> next_float_normal_sample_;
  c10::optional<double> next_double_normal_sample_;
};

namespace detail {

CAFFE2_API CPUGenerator* getDefaultCPUGenerator();
CAFFE2_API std::shared_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail

}
