#include <ATen/RDRANDCPUGenerator.h>
#include <c10/util/C++17.h>
#include <algorithm>
#include <iostream>

namespace at {

/**
 * RDRANDCPUGenerator class implementation
 */
RDRANDCPUGenerator::RDRANDCPUGenerator(uint64_t seed_in)
  : CPUGenerator(seed_in) {
  std::cout << "RDRANDCPUGenerator ctor" << std::endl;
}

/**
 * Gets a random 32 bit unsigned integer from the engine
 * 
 * See Note [Acquire lock when using random generators]
 */
uint32_t RDRANDCPUGenerator::random() {
  const auto res = engine_.GenerateWord32();
  std::cout << "RDRANDCPUGenerator::random() generated " << res << std::endl;
  return res;
}

void RDRANDCPUGenerator::getRNGState(void* target) {
  std::cout << "RDRANDCPUGenerator::getRNGState() not implemented" << std::endl;
}

void RDRANDCPUGenerator::setRNGState(void* target) {
  std::cout << "RDRANDCPUGenerator::setRNGState() not implemented" << std::endl;
}

/**
 * Public clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<RDRANDCPUGenerator> RDRANDCPUGenerator::clone() const {
  return std::shared_ptr<RDRANDCPUGenerator>(this->clone_impl());
}

/**
 * Private clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
RDRANDCPUGenerator* RDRANDCPUGenerator::clone_impl() const {
  auto gen = new RDRANDCPUGenerator();
  return gen;
}

} // namespace at
