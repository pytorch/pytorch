#include <ATen/CPUGenerator.h>
#include <c10/util/C++17.h>

namespace at {

namespace detail {

/*
* call once pattern to initialize default generator once
*/
CPUGenerator& getDefaultCPUGenerator() {
  std::call_once(cpu_device_flag, [&] {
    default_gen_cpu = c10::guts::make_unique<CPUGenerator>(default_rng_seed_val);
  });
  return *default_gen_cpu;
}

/*
* Utility to create a CPUGenerator. Returns a unique_ptr
*/
std::unique_ptr<CPUGenerator> createCPUGenerator(uint64_t seed_val) {
  return c10::guts::make_unique<CPUGenerator>(seed_val);
}

} //namespace detail

/*
* CPUGenerator class implementation
*/
CPUGenerator::CPUGenerator(uint64_t seed_in)
  : Generator(Device(DeviceType::CPU), seed_in), engine_(Philox4_32_10(seed_in)) {}

CPUGenerator::CPUGenerator(const CPUGenerator& other)
  : CPUGenerator(other, 
                 std::lock_guard<std::mutex>(other.mutex)) {}

CPUGenerator::CPUGenerator(const CPUGenerator &other, 
                           const std::lock_guard<std::mutex> &other_mutex)
  : Generator(other, other_mutex), engine_(other.engine_) {}

CPUGenerator::CPUGenerator(CPUGenerator&& other)
  : CPUGenerator(other,
                 std::lock_guard<std::mutex>(other.mutex)) {}

CPUGenerator::CPUGenerator(const CPUGenerator &&other,
                           const std::lock_guard<std::mutex> &other_mutex)
  : Generator(other, other_mutex), engine_(std::move(other.engine_)) {}

CPUGenerator& CPUGenerator::operator=(CPUGenerator& other) {
  if (this != &other) {
    std::unique_lock<std::mutex> this_lock(mutex, std::defer_lock),
                                 other_lock(other.mutex, std::defer_lock);
    std::lock(this_lock, other_lock);
    device_ = other.device_;
    current_seed_ = other.current_seed_;
    engine_ = other.engine_;
  }
  return *this;
}

/* 
* Manually seeds the engine with the seed input
*/
void CPUGenerator::setCurrentSeed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(mutex);
  current_seed_ = seed;
  engine_ = Philox4_32_10(seed);
}

/* 
* Gets a random 32 bit unsigned integer from the engine
*/
uint32_t CPUGenerator::random() {
  std::lock_guard<std::mutex> lock(mutex);
  return engine_();
}

/* 
* Gets a random 64 bit unsigned integer from the engine
*/
uint64_t CPUGenerator::random64() {
  std::lock_guard<std::mutex> lock(mutex);
  uint64_t hi = (static_cast<uint64_t>(engine_())) << 32;
  uint64_t lo = static_cast<uint64_t>(engine_());
  return hi | lo;
}

} // namespace at
