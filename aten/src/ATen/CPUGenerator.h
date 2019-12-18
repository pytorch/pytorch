#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <c10/util/Optional.h>

namespace at {

struct CAFFE2_API CPUGenerator : public Generator {
  // Constructors
  CPUGenerator(uint64_t seed_in = default_rng_seed_val);
  virtual ~CPUGenerator() = default;

  // CPUGenerator methods
  std::shared_ptr<CPUGenerator> clone() const;
  virtual void set_current_seed(uint64_t seed) = 0;
  virtual uint64_t current_seed() const = 0;
  virtual uint64_t seed() = 0;
  static DeviceType device_type();
  /**
   * Gets a random 32 bit unsigned integer from the engine
   *
   * See Note [Acquire lock when using random generators]
   */
  virtual uint32_t random() = 0;
  virtual uint64_t random64();
  c10::optional<float> next_float_normal_sample();
  c10::optional<double> next_double_normal_sample();
  void set_next_float_normal_sample(c10::optional<float> randn);
  void set_next_double_normal_sample(c10::optional<double> randn);
  at::mt19937 engine();
  void set_engine(at::mt19937 engine);

protected:
  // CPUGenerator* clone_impl() const override;
  at::mt19937 engine_;
  c10::optional<float> next_float_normal_sample_;
  c10::optional<double> next_double_normal_sample_;
};

namespace detail {

CAFFE2_API CPUGenerator* getDefaultCPUGenerator();
CAFFE2_API std::shared_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail

}
