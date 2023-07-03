//  Copyright © 2022 Apple Inc.

#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/util/Optional.h>

namespace at {
namespace mps {
namespace detail {

static const uint32_t PHILOX_STATE_N = 7;
struct rng_data_pod {
  std::array<uint32_t, PHILOX_STATE_N> state{1};
  uint64_t seed = default_rng_seed_val;
};

TORCH_API const Generator& getDefaultMPSGenerator();
TORCH_API Generator createMPSGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail
} // namespace mps

struct TORCH_API MPSGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  MPSGeneratorImpl(uint64_t seed_in = default_rng_seed_val);
  ~MPSGeneratorImpl() override = default;

  // MPSGeneratorImpl methods
  std::shared_ptr<MPSGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void update_philox_counters();

  void set_engine(at::Philox4_32 engine) { engine_ = engine; };
  at::Philox4_32 engine() { return engine_; };
  uint32_t* state_data() { return data_.state.data(); }
  static DeviceType device_type() { return DeviceType::MPS; };

private:
  mps::detail::rng_data_pod data_;
  at::Philox4_32 engine_;

  MPSGeneratorImpl* clone_impl() const override;
};

} // namespace at
