#pragma once

#include <stdint.h>
#include <memory>
#include <mutex>
#include <deque>
#include <atomic>
#include <typeinfo>
#include <utility>
#include <cstddef>

#include <c10/util/Exception.h>
#include <c10/util/C++17.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/GeneratorImpl.h>
#include <ATen/CPUGenerator.h>
#include <ATen/CUDAGenerator.h>

/**
 * Note [Generator]
 * ~~~~~~~~~~~~~~~~
 * A Pseudo Random Number Generator (PRNG) is an engine that uses an algorithm to
 * generate a seemingly random sequence of numbers, that may be later be used in creating
 * a random distribution. Such an engine almost always maintains a state and requires a
 * seed to start off the creation of random numbers. Often times, users have
 * found it beneficial to be able to explicitly create, retain, and destroy
 * PRNG states and also be able to have control over the seed value.
 *
 * A Generator in ATen gives users the ability to read, write and modify a PRNG engine.
 * For instance, it does so by letting users seed a PRNG engine, fork the state of the
 * engine, etc.
 *
 * By default, there is one generator per device, and a device's generator is
 * lazily created. A user can use the torch.Generator() api to create their own generator.
 * Currently torch.Generator() can only create a CPUGenerator.
 */

/**
 * Note [Acquire lock when using random generators]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Generator and its derived classes are NOT thread-safe. Please note that most of the
 * places where we have inserted locking for generators are historically based, and we
 * haven't actually checked that everything is truly thread safe (and it probably isn't).
 * Please use the public mutex_ when using any methods from these classes, except for the
 * read-only methods. You can learn about the usage by looking into the unittests
 * (aten/src/ATen/cpu_generator_test.cpp) and other places where we have used lock_guard.
 * 
 * TODO: Look into changing the threading semantics of Generators in ATen (e.g., making
 * them non-thread safe and instead making the generator state splittable, to accommodate
 * forks into other threads).
 */

namespace at {

struct CAFFE2_API Generator {
  Generator() {}

  explicit Generator(std::shared_ptr<c10::GeneratorImpl> gen_impl)
   : impl_(std::move(gen_impl)) {
    if (impl_.get() == nullptr) {
      throw std::runtime_error("GeneratorImpl with nullptr is not supported");
    }
  }

  // TODO(pbelevich): delete this after replace Generator generator = nullptr with c10::optional<at::Generator> = c10::nullopt
  Generator(std::nullptr_t gen_impl) {}

  explicit Generator(Device device) {
#ifdef USE_CUDA
    if (device.type() == at::kCPU) {
      this->impl_ = std::make_shared<CPUGenerator>();
    } else if (device.type() == at::kCUDA){
      this->impl_ = std::make_shared<CUDAGenerator>(device.index());
    } else {
      AT_ERROR("Device type ", c10::DeviceTypeName(device.type()),
              " is not supported for torch.Generator() api.");
    }
#else
    TORCH_CHECK(device.type() == at::kCPU,
                "Device type ", c10::DeviceTypeName(device.type()),
                " is not supported for Generator API.");
    this->impl_ = std::make_shared<CPUGenerator>();
#endif
  }

  bool operator==(const Generator& rhs) const {
    return (!(this->impl_) && !(rhs.impl_)) || (this->impl_ == rhs.impl_);
  }

  bool operator!=(const Generator& rhs) const {
    return !((*this) == rhs);
  }

  bool defined() const {
    return (bool)impl_;
  }

  c10::GeneratorImpl* operator->() const { return impl_.get(); }

  template<typename T>
  T* get() const { return static_cast<T*>(impl_.get()); }

  Generator clone() const {
    return Generator(impl_->clone());
  }

 private:
  std::shared_ptr<c10::GeneratorImpl> impl_;
};

template<class Impl, class... Args>
Generator make_generator(Args&&... args) {
  return Generator(std::make_shared<Impl>(args...));
}

/**
 * Utility function to static cast input Generator* to
 * the backend generator type (CPU/CUDAGenerator etc.)
 */
template <typename T>
static inline T * check_generator(Generator expr) {
  if (T::device_type() == expr->device().type()) {
    return expr.get<T>();
  }
  AT_ERROR("Expected a '", T::device_type(), "' device type for generator but found '", expr->device().type(), "'");
}

/**
 * Utility function used in tensor implementations, which
 * supplies the default generator to tensors, if an input generator
 * is not supplied. The input Generator* is also static casted to
 * the backend generator type (CPU/CUDAGenerator etc.)
 */
template <typename T>
static inline T* get_generator_or_default(const Generator& expr, const Generator& defaultValue) {
  T* result = expr.defined() ? check_generator<T>(expr) : check_generator<T>(defaultValue);
  if (result == nullptr) {
    AT_ERROR("Expected a '", T::device_type(), "' device type for generator but found 'nullptr'");
  }
  return result;
}

namespace detail {

CAFFE2_API const Generator& getDefaultCPUGenerator();
CAFFE2_API Generator createCPUGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace detail

namespace cuda {
namespace detail {

  TORCH_CUDA_API const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1);
  TORCH_CUDA_API Generator createCUDAGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace cuda

} // namespace at
