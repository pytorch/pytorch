#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/TensorBase.h>
#include <ATen/xpu/PhiloxXpuState.h>
#include <unordered_set>

namespace at {

namespace xpu {
struct XPUGraph;
}

struct XPUGeneratorState : public c10::intrusive_ptr_target {
  uint64_t seed_;
  uint64_t philox_offset_per_thread_;
  uint32_t offset_intragraph_;
  bool capturing_{};
  at::TensorBase seed_extragraph_{};
  at::TensorBase offset_extragraph_{};

  XPUGeneratorState(
      uint64_t seed = default_rng_seed_val,
      uint64_t philox_offset_per_thread = 0,
      uint32_t offset_intragraph = 0)
      : seed_(seed),
        philox_offset_per_thread_(philox_offset_per_thread),
        offset_intragraph_(offset_intragraph) {}

  void increase(uint64_t increment);

  c10::intrusive_ptr<XPUGeneratorState> clone();
};

struct TORCH_XPU_API XPUGeneratorImpl : public GeneratorImpl {
  // Constructors
  XPUGeneratorImpl(DeviceIndex device_index = -1);
  XPUGeneratorImpl(
      DeviceIndex device_index,
      c10::intrusive_ptr<XPUGeneratorState> state_);
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

  PhiloxXpuState philox_xpu_state(uint64_t increment);
  // will remove once all ops are refactored to use philox_xpu_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static c10::DeviceType device_type();

 private:
  XPUGeneratorImpl* clone_impl() const override;
  c10::intrusive_ptr<XPUGeneratorState> state_;
};

namespace xpu::detail {

TORCH_XPU_API const Generator& getDefaultXPUGenerator(DeviceIndex device = -1);

TORCH_XPU_API Generator createXPUGenerator(DeviceIndex device = -1);

} // namespace xpu::detail
} // namespace at
