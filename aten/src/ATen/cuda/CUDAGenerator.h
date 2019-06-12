#pragma once

#include <ATen/core/Generator.h>

namespace at {

struct CUDAGenerator : public Generator {
  // Constructors
  CAFFE2_API CUDAGenerator(DeviceIndex device_index = -1);
  CAFFE2_API ~CUDAGenerator() = default;

  // CUDAGenerator methods
  CAFFE2_API std::shared_ptr<CUDAGenerator> clone() const;
  CAFFE2_API void set_current_seed(uint64_t seed) override;
  CAFFE2_API uint64_t current_seed() const override;
  CAFFE2_API void set_philox_offset_per_thread(uint64_t offset);
  CAFFE2_API uint64_t philox_offset_per_thread();
  CAFFE2_API std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  CAFFE2_API static DeviceType device_type();

private:
  CUDAGenerator* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
};

}
