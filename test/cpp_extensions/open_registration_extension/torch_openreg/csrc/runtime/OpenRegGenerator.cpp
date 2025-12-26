#include "OpenRegGenerator.h"
#include "runtime/OpenRegFunctions.h"

// Default, global generators, one per device.
static std::vector<at::Generator> default_generators;

namespace c10::openreg {

// LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR IMPL
const at::Generator& getDefaultOpenRegGenerator(c10::DeviceIndex device_index) {
  static bool flag [[maybe_unused]] = []() {
    auto deivce_nums = device_count();
    default_generators.resize(deivce_nums);
    for (auto i = 0; i < deivce_nums; i++) {
      default_generators[i] = createOpenRegGenerator(i);
      default_generators[i].seed();
    }
    return true;
  }();

  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < device_count());
  }
  return default_generators[idx];
}
// LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR IMPL

// LITERALINCLUDE START: OPENREG CREATE GENERATOR IMPL
at::Generator createOpenRegGenerator(DeviceIndex device_index) {
  check_device_index(device_index);
  auto gen = at::make_generator<OpenRegGeneratorImpl>(device_index);
  auto openreg_gen = at::check_generator<OpenRegGeneratorImpl>(gen);
  openreg_gen->set_current_seed(default_rng_seed_val);
  return gen;
}
// LITERALINCLUDE END: OPENREG CREATE GENERATOR IMPL

DeviceType OpenRegGeneratorImpl::device_type() {
  return DeviceType::PrivateUse1;
}

} // namespace c10::openreg
