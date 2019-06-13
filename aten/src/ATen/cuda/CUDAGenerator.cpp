#include <ATen/CUDAGenerator.h>

namespace at {

/**
 * CUDAGenerator class implementation
 */
CUDAGenerator::CUDAGenerator(DeviceIndex device_index)
  : Generator{Device(DeviceType::CUDA, device_index)} { }

/**
 * Sets the seed to be used by curandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 * 
 * See Note [Acquire lock when using random generators]
 */
void CUDAGenerator::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

/**
 * Gets the current seed of CUDAGenerator.
 */
uint64_t CUDAGenerator::current_seed() const {
  return seed_;
}

/**
 * Gets a nondeterminstic random number from /dev/urandom or time,
 * seeds the CPUGenerator with it and then returns that number.
 * 
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and CUDA
 */
uint64_t CUDAGenerator::seed() {
  auto random = detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 * 
 * See Note [Acquire lock when using random generators]
 */
void CUDAGenerator::set_philox_offset_per_thread(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGenerator.
 */
uint64_t CUDAGenerator::philox_offset_per_thread() {
  return philox_offset_per_thread_;
}

/**
 * Gets the seed and philox offset value to be used in
 * curandStatePhilox4_32_10
 * 
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate. 
 * 
 * Increment should be at least the number of curand() random numbers used in
 * each thread. It is the user's responsibility to make sure that the increment
 * for philox is never smaller than the number of curand() calls. Increment
 * value > the number of curand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 * 
 * See Note [Acquire lock when using random generators]
 */
std::pair<uint64_t, uint64_t> CUDAGenerator::philox_engine_inputs(uint64_t increment) {
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

/*
 * Gets the DeviceType of CUDAGenerator.
 * Used for type checking during run time.
 */
DeviceType CUDAGenerator::device_type() {
  return DeviceType::CUDA;
}

/**
 * Public clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<CUDAGenerator> CUDAGenerator::clone() const {
  return std::shared_ptr<CUDAGenerator>(this->clone_impl());
}

/**
 * Private clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
CUDAGenerator* CUDAGenerator::clone_impl() const {
  auto gen = new CUDAGenerator(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

} // namespace at
