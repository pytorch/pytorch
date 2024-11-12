#include <ATen/Utils.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/xpu/XPUFunctions.h>

namespace at {
namespace xpu::detail {
namespace {

/*
 * Currently, there is one generator pool containing XPU generator per device.
 * Each generator is lazily initialized the first time generator is
 * requested for a device.
 */
c10::once_flag init_flag;
DeviceIndex num_gpus = -1;
std::deque<c10::once_flag> xpu_gens_init_flag;
std::vector<Generator> default_gens_xpu;

void initXPUGenVector() {
  num_gpus = device_count();
  xpu_gens_init_flag.resize(num_gpus);
  default_gens_xpu.resize(num_gpus);
}

} // anonymous namespace

// Get the default generator with a random seed for a specific xpu device.
const Generator& getDefaultXPUGenerator(DeviceIndex device) {
  c10::call_once(init_flag, initXPUGenVector);
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device_index(device);
  c10::call_once(xpu_gens_init_flag[device], [&]() {
    default_gens_xpu[device] = make_generator<XPUGeneratorImpl>(device);
    default_gens_xpu[device].seed();
  });
  return default_gens_xpu[device];
}

// Create a generator with a fixed seed for a specific xpu device.
Generator createXPUGenerator(DeviceIndex device) {
  c10::call_once(init_flag, initXPUGenVector);
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device_index(device);
  auto gen = make_generator<XPUGeneratorImpl>(device);
  auto xpu_gen = check_generator<XPUGeneratorImpl>(gen);
  xpu_gen->set_current_seed(default_rng_seed_val);
  xpu_gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace xpu::detail

XPUGeneratorImpl::XPUGeneratorImpl(DeviceIndex device_index)
    : GeneratorImpl{
          Device(DeviceType::XPU, device_index),
          DispatchKeySet(c10::DispatchKey::XPU)} {}

void XPUGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  set_philox_offset_per_thread(0);
}

void XPUGeneratorImpl::set_offset(uint64_t offset) {
  set_philox_offset_per_thread(offset);
}

uint64_t XPUGeneratorImpl::get_offset() const {
  return philox_offset_per_thread_;
}

uint64_t XPUGeneratorImpl::current_seed() const {
  return seed_;
}

uint64_t XPUGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

c10::intrusive_ptr<c10::TensorImpl> XPUGeneratorImpl::get_state() const {
  // The RNG state comprises the seed, and an offset used for Philox.
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(uint64_t);
  static const size_t total_size = seed_size + offset_size;

  // The internal state is returned as a CPU byte tensor.
  auto state_tensor = at::detail::empty_cpu(
      {static_cast<int64_t>(total_size)},
      ScalarType::Byte,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt);
  auto rng_state = state_tensor.data_ptr<uint8_t>();
  auto current_seed = this->current_seed();
  auto offset = this->philox_offset_per_thread();
  memcpy(rng_state, &current_seed, seed_size);
  memcpy(rng_state + seed_size, &offset, offset_size);

  return state_tensor.getIntrusivePtr();
}

void XPUGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(uint64_t);
  static const size_t total_size = seed_size + offset_size;

  at::detail::check_rng_state(new_state);
  auto new_state_size = new_state.numel();
  TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");

  uint64_t input_seed;
  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
  uint64_t philox_offset;
  memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
  this->set_philox_offset_per_thread(philox_offset);
}

void XPUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  philox_offset_per_thread_ = offset;
}

uint64_t XPUGeneratorImpl::philox_offset_per_thread() const {
  return philox_offset_per_thread_;
}

std::pair<uint64_t, uint64_t> XPUGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  increment = ((increment + 3) / 4) * 4;
  TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

DeviceType XPUGeneratorImpl::device_type() {
  return DeviceType::XPU;
}

std::shared_ptr<XPUGeneratorImpl> XPUGeneratorImpl::clone() const {
  return std::shared_ptr<XPUGeneratorImpl>(this->clone_impl());
}

XPUGeneratorImpl* XPUGeneratorImpl::clone_impl() const {
  auto gen = new XPUGeneratorImpl(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

} // namespace at
