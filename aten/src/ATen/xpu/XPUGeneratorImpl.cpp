#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <ATen/xpu/XPUGraph.h>
#include <ATen/xpu/XPUGraphsUtils.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <memory>
#include <vector>

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
  std::lock_guard<std::mutex> lock(mutex_);
  return make_intrusive<XPUGeneratorState>(seed_, philox_offset_per_thread_);
}

// Function to increase the internal offset based on the specified increment.
void XPUGeneratorState::increase(uint64_t increment) {
  increment = ((increment + PHILOX_ROUND_SIZE - 1) / PHILOX_ROUND_SIZE) *
      PHILOX_ROUND_SIZE;
  auto capture_id = at::xpu::currentStreamCaptureId();
  if (capture_id.has_value()) {
    get_capture_state(*capture_id, true)->increase(increment);
  } else {
    TORCH_INTERNAL_ASSERT(
        philox_offset_per_thread_ % 4 == 0,
        "RNG offset must be a multiple of 4.");
    philox_offset_per_thread_ += increment;
  }
}

void XPUGeneratorCaptureState::initialize() {
  if (is_initialized()) {
    return;
  }

  static auto& allocation_queues =
      *new std::vector<std::unique_ptr<sycl::queue>>(c10::xpu::device_count());
  static std::mutex allocation_mutex;
  std::lock_guard<std::mutex> lock(allocation_mutex);
  auto device = c10::xpu::current_device();
  auto& queue = allocation_queues[device];
  if (!queue) {
    queue = std::make_unique<sycl::queue>(
        c10::xpu::get_device_context(),
        c10::xpu::get_raw_device(device),
        sycl::property_list{sycl::property::queue::in_order{}});
  }
  auto stream = c10::xpu::getStreamFromExternal(queue.get(), device);
  c10::StreamGuard stream_guard(stream);
  c10::InferenceMode inference_guard(false);
  auto options = at::TensorOptions().device(at::kXPU).dtype(at::kLong);
  seed_extragraph_ = at::empty({1}, options);
  offset_extragraph_ = at::empty({1}, options);
  seed_extragraph_.storage().unsafeGetStorageImpl()->set_resizable(false);
  offset_extragraph_.storage().unsafeGetStorageImpl()->set_resizable(false);
  stream.synchronize();
  offset_intragraph_ = 0;
}

void XPUGeneratorCaptureState::increase(uint64_t increment) {
  TORCH_INTERNAL_ASSERT(
      offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
  TORCH_INTERNAL_ASSERT(
      offset_intragraph_ <= std::numeric_limits<uint64_t>::max() - increment,
      "Increment causes overflow in the offset value.");
  offset_intragraph_ += increment;
}

uint64_t XPUGeneratorCaptureState::finalize() {
  auto increment = offset_intragraph_;
  offset_intragraph_ = 0;
  return increment;
}

void XPUGeneratorCaptureState::setup_for_replay(
    uint64_t seed,
    uint64_t offset) {
  TORCH_INTERNAL_ASSERT(is_initialized(), "Capture state not initialized");
  seed_extragraph_.fill_(static_cast<int64_t>(seed));
  offset_extragraph_.fill_(static_cast<int64_t>(offset));
  auto stream = c10::xpu::getCurrentXPUStream();
  c10::xpu::XPUCachingAllocator::recordStream(
      seed_extragraph_.storage().data_ptr(), stream);
  c10::xpu::XPUCachingAllocator::recordStream(
      offset_extragraph_.storage().data_ptr(), stream);
}

XPUGeneratorCaptureState* XPUGeneratorState::get_capture_state(
    size_t capture_id,
    bool create_if_not_found) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = capture_states_.find(capture_id);
    if (it != capture_states_.end()) {
      return it->second.get();
    }
    if (!create_if_not_found) {
      return nullptr;
    }
  }
  auto* graph = xpu::get_graph_from_capture_id(capture_id);
  TORCH_CHECK(
      graph != nullptr,
      "RNG op during graph capture but could not find the XPUGraph object.");
  auto capture_state = make_intrusive<XPUGeneratorCaptureState>();
  capture_state->initialize();
  graph->register_generator_state(
      c10::intrusive_ptr<XPUGeneratorState>::reclaim_copy(this));
  std::lock_guard<std::mutex> lock(mutex_);
  auto result = capture_states_.emplace(capture_id, std::move(capture_state));
  return result.first->second.get();
}

void XPUGeneratorState::remove_capture_state(size_t capture_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  capture_states_.erase(capture_id);
}

uint64_t XPUGeneratorState::capture_epilogue(size_t capture_id) {
  auto* capture_state = get_capture_state(capture_id);
  return capture_state ? capture_state->finalize() : 0;
}

void XPUGeneratorState::replay_prologue(
    size_t capture_id,
    uint64_t wholegraph_increment) {
  at::xpu::assertNotCapturing(
      "Cannot prepare for replay during capturing stage.");
  if (wholegraph_increment == 0) {
    return;
  }
  uint64_t replay_seed = 0;
  uint64_t replay_offset = 0;
  c10::intrusive_ptr<XPUGeneratorCaptureState> capture_state;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = capture_states_.find(capture_id);
    TORCH_INTERNAL_ASSERT(
        it != capture_states_.end(),
        "replay_prologue called but no capture state found for this capture_id");
    capture_state = it->second;
    replay_seed = seed_;
    replay_offset = philox_offset_per_thread_;
    philox_offset_per_thread_ += wholegraph_increment;
  }
  capture_state->setup_for_replay(replay_seed, replay_offset);
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
  std::lock_guard<std::mutex> lock(state_->mutex_);
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
  std::lock_guard<std::mutex> lock(state_->mutex_);
  return state_->philox_offset_per_thread_;
}

uint64_t XPUGeneratorImpl::current_seed() const {
  std::lock_guard<std::mutex> lock(state_->mutex_);
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
  constexpr size_t seed_size = sizeof(uint64_t);
  constexpr size_t offset_size = sizeof(uint64_t);
  constexpr size_t total_size = seed_size + offset_size;

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
  constexpr size_t seed_size = sizeof(uint64_t);
  constexpr size_t offset_size = sizeof(uint64_t);
  constexpr size_t total_size = seed_size + offset_size;

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

void XPUGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<GeneratorImpl>& gen) {
  c10::intrusive_ptr<XPUGeneratorImpl> xpu_gen =
      dynamic_intrusive_pointer_cast<XPUGeneratorImpl>(gen);
  TORCH_CHECK(xpu_gen, "Expected a XPU Generator");
  state_ = xpu_gen->state_;
}

c10::intrusive_ptr<c10::GeneratorImpl> XPUGeneratorImpl::graphsafe_get_state()
    const {
  auto gen = make_intrusive<XPUGeneratorImpl>(device().index(), state_);
  return gen;
}

void XPUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  auto capture_id = at::xpu::currentStreamCaptureId();
  if (!capture_id.has_value()) {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    state_->philox_offset_per_thread_ = offset;
  } else {
    state_->get_capture_state(*capture_id, true)->offset_intragraph_ = offset;
  }
}

uint64_t XPUGeneratorImpl::philox_offset_per_thread() const {
  auto capture_id = at::xpu::currentStreamCaptureId();
  if (!capture_id.has_value()) {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    return state_->philox_offset_per_thread_;
  } else {
    return state_->get_capture_state(*capture_id, true)->offset_intragraph_;
  }
}

void XPUGeneratorImpl::register_graph(xpu::XPUGraphImpl* graph) {
  at::xpu::assertNotCapturing(
      "Cannot register the state during capturing stage.");
  graph->register_generator_state(state_);
}

// 1, During graph capture, constructs a PhiloxXpuState
//    [extragraph seed ptr, extragraph offset ptr, intragraph offset] on host
// 2, Before each replay, the extragraph seed and offset tensors will be updated
//    extragraph offset = philox_offset_per_thread_ + intragraph offset
// 3, During replay, kernel will compute final offset = *extragraph offset ptr +
// intragraph offset
PhiloxXpuState XPUGeneratorImpl::philox_xpu_state(uint64_t increment) {
  auto capture_id = at::xpu::currentStreamCaptureId();
  if (capture_id.has_value()) {
    auto* capture_state = state_->get_capture_state(*capture_id, true);
    uint64_t offset = capture_state->offset_intragraph_;
    state_->increase(increment);
    return PhiloxXpuState(
        capture_state->seed_extragraph_.data_ptr<int64_t>(),
        capture_state->offset_extragraph_.data_ptr<int64_t>(),
        offset);
  } else {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    uint64_t offset = state_->philox_offset_per_thread_;
    state_->increase(increment);
    return PhiloxXpuState(state_->seed_, offset);
  }
}

std::pair<uint64_t, uint64_t> XPUGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  at::xpu::assertNotCapturing(
      "Refactor this op to use XPUGeneratorImpl::philox_xpu_state. Cannot call XPUGeneratorImpl::philox_engine_inputs");
  std::lock_guard<std::mutex> lock(state_->mutex_);
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
