#pragma once

#include <ATen/CPUGenerator.h>
#include <c10/util/Optional.h>
#include <cryptopp/rdrand.h>

namespace at {

struct CAFFE2_API RDRANDCPUGenerator : public CPUGenerator {
  // Constructors
  RDRANDCPUGenerator(uint64_t seed_in = default_rng_seed_val);
  ~RDRANDCPUGenerator() = default;

  // RDRANDCPUGenerator methods
  std::shared_ptr<RDRANDCPUGenerator> clone() const;
  uint32_t random() override;
  void getRNGState(void* target) override;
  void setRNGState(void* target) override;

private:
  RDRANDCPUGenerator* clone_impl() const override;
  CryptoPP::RDRAND engine_;
};

} // namespace at
