#pragma once

#include <ATen/core/Generator.h>

namespace at {

struct TORCH_XPU_API XPUGeneratorImpl : public GeneratorImpl {
  // Constructors
  XPUGeneratorImpl(DeviceIndex device_index = -1);
  ~XPUGeneratorImpl() override = default;

  // XPUGeneratorImpl methods
  std::shared_ptr<XPUGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static c10::DeviceType device_type();

 private:
  XPUGeneratorImpl* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
};

namespace xpu::detail {

TORCH_XPU_API const Generator& getDefaultXPUGenerator(DeviceIndex device = -1);

TORCH_XPU_API Generator createXPUGenerator(DeviceIndex device = -1);

} // namespace xpu::detail
} // namespace at
