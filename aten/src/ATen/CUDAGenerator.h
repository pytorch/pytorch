#pragma once

#include <ATen/core/Generator.h>

namespace at {

struct CAFFE2_API CUDAGenerator : public Generator {
  // Constructors
  CUDAGenerator(DeviceIndex device_index = -1);
  ~CUDAGenerator() = default;

  // CUDAGenerator methods
  std::shared_ptr<CUDAGenerator> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread();
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static DeviceType device_type();

private:
  CUDAGenerator* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
};

}
