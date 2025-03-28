#include <c10/macros/Export.h>

#include <aten/src/ATen/mps/MPSDevice.h>
#include <aten/src/ATen/mps/MPSGeneratorImpl.h>
#include <ATen/core/Generator.h>

namespace at {
  namespace native {
    namespace mps {
      TORCH_API void
      copy_blit_mps(void*dst, void const*src, unsigned long size) {

      }
      TORCH_API at::Tensor&
      mps_copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
	return dst;
      }
    } // namespace mps
  } // namespace native

  namespace mps {
    TORCH_API at::Allocator*
    GetMPSAllocator(bool useSharedAllocator) {
      return NULL;
    }

    TORCH_API bool
    eq(at::Tensor const&, at::Tensor const&) {
      return true;
    }

    namespace detail {

      TORCH_API Generator
      createMPSGenerator(uint64_t seed_val) {
	return Generator();
      }
    } // namespace detail

    TORCH_API bool
    is_macos_13_or_newer(at::mps::MacOSVersion version) {
      return true;
    }
  } // namespace mps


  MPSGeneratorImpl::MPSGeneratorImpl(uint64_t seed_in)
    : c10::GeneratorImpl{Device(DeviceType::MPS, 0), DispatchKeySet(c10::DispatchKey::MPS)}
  {
    engine_ = at::Philox4_32(seed_in);

    // Initialize or reset other members as needed
    data_.seed = seed_in;
  }

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

  c10::intrusive_ptr<c10::TensorImpl> MPSGeneratorImpl::get_state() const {
    auto state_tensor = c10::intrusive_ptr<c10::TensorImpl>();

    return state_tensor;
  }

  void MPSGeneratorImpl::set_state(const c10::TensorImpl& new_state) {

  }


  // See Note [Acquire lock when using random generators]
  void MPSGeneratorImpl::update_philox_counters() {

  }

MPSGeneratorImpl* MPSGeneratorImpl::clone_impl() const {
  auto gen = new MPSGeneratorImpl();
  gen->set_current_seed(this->data_.seed);
  return gen;
}

} // namespace at
