#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/TensorBase.h>
#include <ATen/xpu/PhiloxXpuState.h>
#include <c10/util/flat_hash_map.h>
#include <mutex>

namespace at {

namespace xpu {
struct XPUGraphImpl;
}

struct XPUGeneratorCaptureState : public c10::intrusive_ptr_target {
  uint64_t offset_intragraph_{0};
  at::TensorBase seed_extragraph_;
  at::TensorBase offset_extragraph_;

  bool is_initialized() const {
    return seed_extragraph_.defined();
  }
  void initialize();
  void increase(uint64_t increment);
  uint64_t finalize();
  void setup_for_replay(uint64_t seed, uint64_t offset);
};

struct XPUGeneratorState : public c10::intrusive_ptr_target {
  uint64_t seed_;
  uint64_t philox_offset_per_thread_;
  ska::flat_hash_map<size_t, c10::intrusive_ptr<XPUGeneratorCaptureState>>
      capture_states_;
  mutable std::mutex mutex_;

  XPUGeneratorState(
      uint64_t seed = default_rng_seed_val,
      uint64_t philox_offset_per_thread = 0)
      : seed_(seed), philox_offset_per_thread_(philox_offset_per_thread) {}

  void increase(uint64_t increment);
  XPUGeneratorCaptureState* get_capture_state(
      size_t capture_id,
      bool create_if_not_found = false);
  void remove_capture_state(size_t capture_id);
  uint64_t capture_epilogue(size_t capture_id);
  void replay_prologue(size_t capture_id, uint64_t wholegraph_increment);

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
  void graphsafe_set_state(
      const c10::intrusive_ptr<GeneratorImpl>& state) override;
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;

  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;

  void register_graph(xpu::XPUGraphImpl* graph);
  PhiloxXpuState philox_xpu_state(uint64_t increment);
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
