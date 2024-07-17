#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <c10/core/GeneratorImpl.h>
#include <optional>

namespace at {

struct TORCH_API CPUGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  CPUGeneratorImpl(uint64_t seed_in = default_rng_seed_val);
  ~CPUGeneratorImpl() override = default;

  // CPUGeneratorImpl methods
  std::shared_ptr<CPUGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  static c10::DeviceType device_type();
  uint32_t random();
  uint64_t random64();
  std::optional<float> next_float_normal_sample();
  std::optional<double> next_double_normal_sample();
  void set_next_float_normal_sample(std::optional<float> randn);
  void set_next_double_normal_sample(std::optional<double> randn);
  at::mt19937 engine();
  void set_engine(at::mt19937 engine);

 private:
  CPUGeneratorImpl* clone_impl() const override;
  at::mt19937 engine_;
  std::optional<float> next_float_normal_sample_;
  std::optional<double> next_double_normal_sample_;
};

namespace detail {

TORCH_API const Generator& getDefaultCPUGenerator();
TORCH_API Generator
createCPUGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail

} // namespace at
