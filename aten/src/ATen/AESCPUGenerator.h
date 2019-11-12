#pragma once

#include <ATen/CPUGenerator.h>
#include <c10/util/Optional.h>
#include <cryptopp/randpool.h>

namespace at {

struct CAFFE2_API AESCPUGenerator : public CPUGenerator {
  // Constructors
  AESCPUGenerator(uint64_t seed_in = default_rng_seed_val);
  ~AESCPUGenerator() = default;

  // AESCPUGenerator methods
  std::shared_ptr<AESCPUGenerator> clone() const;
  uint32_t random() override;
  void getRNGState(void* target) override;
  void setRNGState(void* target) override;

private:
  AESCPUGenerator* clone_impl() const override;
  CryptoPP::RandomPool engine_;
};

} // namespace at
