#include <ATen/AESCPUGenerator.h>
#include <c10/util/C++17.h>
#include <algorithm>
#include <iostream>

namespace at {

/**
 * AESCPUGenerator class implementation
 */
AESCPUGenerator::AESCPUGenerator(uint64_t seed_in)
  : CPUGenerator(seed_in) {
  std::cout << "AESCPUGenerator ctor" << std::endl;
}

/**
 * Gets a random 32 bit unsigned integer from the engine
 * 
 * See Note [Acquire lock when using random generators]
 */
uint32_t AESCPUGenerator::random() {
  const auto res = engine_.GenerateWord32();
  std::cout << "AESCPUGenerator::random() generated " << res << std::endl;
  return res;
}

void AESCPUGenerator::getRNGState(void* target) {
  std::cout << "AESCPUGenerator::getRNGState() not implemented" << std::endl;
}

void AESCPUGenerator::setRNGState(void* target) {
  std::cout << "AESCPUGenerator::setRNGState() not implemented" << std::endl;
}

/**
 * Public clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<AESCPUGenerator> AESCPUGenerator::clone() const {
  return std::shared_ptr<AESCPUGenerator>(this->clone_impl());
}

/**
 * Private clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
AESCPUGenerator* AESCPUGenerator::clone_impl() const {
  auto gen = new AESCPUGenerator();
  return gen;
}

} // namespace at
