#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <ATen/xpu/XPUGraphsUtils.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/xpu/XPUFunctions.h>

constexpr uint64_t PHILOX_ROUND_SIZE = 4;

namespace at {
namespace xpu::detail {
namespace {

/*
 * Currently, there is one generator pool containing XPU generator per device.
 * Each generator is lazily initialized the first time generator is
 * requested for a device.
 */
DeviceIndex num_gpus = -1;
std::deque<c10::once_flag> xpu_gens_init_flag;
std::vector<Generator> default_gens_xpu;

void initXPUGenVector() {
  static bool init_flag [[maybe_unused]] = []() {
    num_gpus = device_count();
    xpu_gens_init_flag.resize(num_gpus);
    default_gens_xpu.resize(num_gpus);
    return true;
  }();
}

} // anonymous namespace

// Get the default generator with a random seed for a specific xpu device.
const Generator& getDefaultXPUGenerator(DeviceIndex device) {
  initXPUGenVector();
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
  initXPUGenVector();
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

// Creates a clone of this XPU Generator State.
c10::intrusive_ptr<XPUGeneratorState> XPUGeneratorState::clone() {
  return make_intrusive<XPUGeneratorState>(
      seed_, philox_offset_per_thread_, offset_intragraph_);
}

// Function to increase the internal offset based on the specified increment.
void XPUGeneratorState::increase(uint64_t increment) {
  increment = ((increment + PHILOX_ROUND_SIZE - 1) / PHILOX_ROUND_SIZE) *
      PHILOX_ROUND_SIZE;
  if (at::xpu::currentStreamCaptureStatus() !=
      at::xpu::CaptureStatus::Executing) {
    TORCH_INTERNAL_ASSERT(
        capturing_,
        "Attempt to increase offset for a XPU generator not in capture mode.");
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
        "Increment causes overflow in the offset value.");
    offset_intragraph_ += increment;
  } else {
    TORCH_INTERNAL_ASSERT(
        !capturing_,
        "Offset increment outside graph capture encountered unexpectedly.");
    TORCH_INTERNAL_ASSERT(
        philox_offset_per_thread_ % 4 == 0,
        "RNG offset must be a multiple of 4.");
    philox_offset_per_thread_ += increment;
  }
}

XPUGeneratorImpl::XPUGeneratorImpl(DeviceIndex device_index)
    : GeneratorImpl{
          Device(DeviceType::XPU, device_index),
          DispatchKeySet(c10::DispatchKey::XPU)} {
  at::xpu::assertNotCapturing("Cannot construct a new XPUGeneratorImpl");
  state_ = make_intrusive<XPUGeneratorState>();
}

XPUGeneratorImpl::XPUGeneratorImpl(
    DeviceIndex device_index,
    intrusive_ptr<XPUGeneratorState> state)
    : GeneratorImpl{Device(DeviceType::XPU, device_index), DispatchKeySet(c10::DispatchKey::XPU)},
      state_(std::move(state)) {}

void XPUGeneratorImpl::set_current_seed(uint64_t seed) {
  if (C10_LIKELY(
          at::xpu::currentStreamCaptureStatus() ==
          at::xpu::CaptureStatus::Executing)) {
    state_->seed_ = seed;
    state_->philox_offset_per_thread_ = 0;
  } else {
    TORCH_CHECK(
        state_->seed_ == seed,
        "XPUGeneratorImpl::set_current_seed can be called during stream capture only if new seed is the same as the original seed.");
  }
}

void XPUGeneratorImpl::set_offset(uint64_t offset) {
  at::xpu::assertNotCapturing("Cannot call XPUGeneratorImpl::set_offset");
  set_philox_offset_per_thread(offset);
}

uint64_t XPUGeneratorImpl::get_offset() const {
  at::xpu::assertNotCapturing("Cannot call XPUGeneratorImpl::get_offset");
  return state_->philox_offset_per_thread_;
}

uint64_t XPUGeneratorImpl::current_seed() const {
  at::xpu::assertNotCapturing("Cannot call XPUGeneratorImpl::current_seed");
  return state_->seed_;
}

uint64_t XPUGeneratorImpl::seed() {
  at::xpu::assertNotCapturing("Cannot call XPUGeneratorImpl::seed");
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
  at::xpu::assertNotCapturing(
      "Please ensure to utilize the XPUGeneratorImpl::set_state_index method during capturing.");
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(uint64_t);
  static const size_t total_size = seed_size + offset_size;

  at::detail::check_rng_state(new_state);

  bool no_philox_seed = false;
  auto new_state_size = new_state.numel();
  if (new_state_size == total_size - offset_size) {
    no_philox_seed = true;
  } else {
    TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
  }

  uint64_t input_seed = 0;
  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
  uint64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
  }
  this->set_philox_offset_per_thread(philox_offset);
}

void XPUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  state_->philox_offset_per_thread_ = offset;
}

uint64_t XPUGeneratorImpl::philox_offset_per_thread() const {
  return state_->philox_offset_per_thread_;
}

PhiloxXpuState XPUGeneratorImpl::philox_xpu_state(uint64_t increment) {
  if (at::xpu::currentStreamCaptureStatus() !=
      at::xpu::CaptureStatus::Executing) {
    uint32_t offset = state_->offset_intragraph_;
    state_->increase(increment);
    return PhiloxXpuState(
        state_->seed_extragraph_.data_ptr<int64_t>(),
        state_->offset_extragraph_.data_ptr<int64_t>(),
        offset);
  } else {
    uint64_t offset = state_->philox_offset_per_thread_;
    state_->increase(increment);
    return PhiloxXpuState(state_->seed_, offset);
  }
}

std::pair<uint64_t, uint64_t> XPUGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  at::xpu::assertNotCapturing(
      "Refactor this op to use XPUGeneratorImpl::philox_xpu_state. Cannot call XPUGeneratorImpl::philox_engine_inputs");
  uint64_t offset = state_->philox_offset_per_thread_;
  state_->increase(increment);
  return std::make_pair(state_->seed_, offset);
}

DeviceType XPUGeneratorImpl::device_type() {
  return DeviceType::XPU;
}

std::shared_ptr<XPUGeneratorImpl> XPUGeneratorImpl::clone() const {
  return std::shared_ptr<XPUGeneratorImpl>(this->clone_impl());
}

XPUGeneratorImpl* XPUGeneratorImpl::clone_impl() const {
  at::xpu::assertNotCapturing("Cannot call XPUGeneratorImpl::clone_impl");
  auto gen = new XPUGeneratorImpl(this->device().index(), state_->clone());
  return gen;
}

} // namespace at
