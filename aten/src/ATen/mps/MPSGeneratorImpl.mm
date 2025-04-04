//  Copyright © 2022 Apple Inc.

#include <ATen/Utils.h>
#include <ATen/mps/MPSGeneratorImpl.h>
#include <algorithm>

namespace at {
namespace mps::detail {

const Generator& getDefaultMPSGenerator() {
  static auto default_gen_mps = createMPSGenerator(c10::detail::getNonDeterministicRandom());
  return default_gen_mps;
}

Generator createMPSGenerator(uint64_t seed_val) {
  auto gen = make_generator<MPSGeneratorImpl>(seed_val);
  gen.set_current_seed(seed_val);
  return gen;
}

} // namespace mps::detail

MPSGeneratorImpl::MPSGeneratorImpl(uint64_t seed_in)
    : c10::GeneratorImpl{Device(DeviceType::MPS, 0), DispatchKeySet(c10::DispatchKey::MPS)},
      data_({.seed = seed_in}),
      engine_(seed_in, 0, 0) {}

void MPSGeneratorImpl::set_current_seed(uint64_t seed) {
  data_.seed = seed;
  data_.state.fill(1);
  // the two last state values are the Philox keys
  // TODO: make "key" in PhiloxRNGEngine.h public so we don't duplicate code here
  data_.state[5] = static_cast<uint32_t>(seed);
  data_.state[6] = static_cast<uint32_t>(seed >> 32);
  engine_.reset_state(seed);
}

void MPSGeneratorImpl::set_offset(uint64_t offset) {
  engine_.set_offset(offset);
}

uint64_t MPSGeneratorImpl::get_offset() const {
  return engine_.get_offset();
}

uint64_t MPSGeneratorImpl::current_seed() const {
  return data_.seed;
}

uint64_t MPSGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom();
  this->set_current_seed(random);
  return random;
}

// See Note [Acquire lock when using random generators]
void MPSGeneratorImpl::update_philox_counters() {
  // calling engine_() would call operator() of philox_engine class to
  // get each of the four newly generated counter values (see PhiloxRNGEngine.h).
  for (int i = 1; i <= 4; i++) {
    data_.state[i] = engine_();
  }
}

c10::intrusive_ptr<c10::TensorImpl> MPSGeneratorImpl::get_state() const {
  constexpr size_t states_size = mps::detail::PHILOX_STATE_N * sizeof(uint32_t);
  constexpr size_t seed_size = sizeof(uint64_t);
  constexpr size_t offset_size = sizeof(uint64_t);
  constexpr size_t total_size = states_size + seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu(
      {(int64_t)total_size}, ScalarType::Byte, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  auto rng_state = state_tensor.mutable_data_ptr<uint8_t>();
  auto current_seed = this->current_seed();
  auto current_offset = this->get_offset();

  static_assert(sizeof(decltype(current_seed)) == seed_size, "current_seed size is wrong");
  static_assert(sizeof(decltype(current_offset)) == offset_size, "current_offset size is wrong");

  memcpy(rng_state, this->data_.state.data(), states_size);
  memcpy(rng_state + states_size, &current_seed, seed_size);
  memcpy(rng_state + states_size + seed_size, &current_offset, offset_size);

  return state_tensor.getIntrusivePtr();
}

void MPSGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  constexpr size_t states_size = mps::detail::PHILOX_STATE_N * sizeof(uint32_t);
  constexpr size_t seed_size = sizeof(uint64_t);
  constexpr size_t offset_size = sizeof(uint64_t);
  constexpr size_t total_size = states_size + seed_size + offset_size;

  detail::check_rng_state(new_state);

  auto new_state_size = new_state.numel();
  TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");

  uint64_t input_seed = default_rng_seed_val;
  uint64_t input_offset = 0;
  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state + states_size, seed_size);
  this->set_current_seed(input_seed);
  memcpy(&input_offset, new_rng_state + states_size + seed_size, offset_size);
  this->set_offset(input_offset);
  // state.data must be copied after input_seed to not reset the state in set_current_seed()
  memcpy(this->state_data(), new_rng_state, states_size);
}

std::shared_ptr<MPSGeneratorImpl> MPSGeneratorImpl::clone() const {
  return std::shared_ptr<MPSGeneratorImpl>(this->clone_impl());
}

MPSGeneratorImpl* MPSGeneratorImpl::clone_impl() const {
  auto gen = new MPSGeneratorImpl();
  gen->set_current_seed(this->data_.seed);
  return gen;
}

} // namespace at
