#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/PhiloxRNGEngine.h>

namespace at {

struct CPUGenerator;

namespace detail {

// Global generator state and constants
static std::once_flag cpu_device_flag;
static std::unique_ptr<CPUGenerator> default_gen_cpu;

// Internal API
CAFFE2_API CPUGenerator& getDefaultCPUGenerator();
CAFFE2_API std::unique_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail

struct CAFFE2_API CPUGenerator : public Generator {
  // Constructors
  CPUGenerator(uint64_t seed_in = default_rng_seed_val);
  CPUGenerator(const CPUGenerator& other);
  CPUGenerator(CPUGenerator&& other);
  CPUGenerator& operator=(CPUGenerator& other);

  void setCurrentSeed(uint64_t seed) override;
  uint32_t random();
  uint64_t random64();

private:
  Philox4_32_10 engine_;
  
  // constructor forwarding to grab mutex before any copying or moving
  CPUGenerator(const CPUGenerator &other,
               const std::lock_guard<std::mutex> &other_mutex);
  CPUGenerator(const CPUGenerator &&other,
               const std::lock_guard<std::mutex> &other_mutex);
};

}
