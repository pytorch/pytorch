#include <ATen/core/Generator.h>

namespace at {

/**
 * Generator class implementation
 */
Generator::Generator(Device device_in) : device_(device_in) {}

/**
 * Clone this generator. Note that clone() is the only
 * method for copying for Generators in ATen.
 */
std::unique_ptr<Generator> Generator::clone() const {
  return std::unique_ptr<Generator>(static_cast<Generator*>(this->clone_impl()));
}

/**
 * Gets the device of a generator.
 */
Device Generator::device() const {
  return device_;
}

Generator& Generator::manualSeedAll(uint64_t seed) {
  AT_ERROR("manualSeedAll is a CUDA only function. It will be deprecated soon.");
}

} // namespace at