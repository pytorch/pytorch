#include <c10/core/GeneratorImpl.h>
#include <chrono>
#include <random>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace c10 {

/**
 * GeneratorImpl class implementation
 */
GeneratorImpl::GeneratorImpl(Device device_in, DispatchKeySet key_set)
 : device_{device_in}, key_set_(key_set) {}

/**
 * Clone this generator. Note that clone() is the only
 * method for copying for Generators in ATen.
 */
c10::intrusive_ptr<GeneratorImpl> GeneratorImpl::clone() const {
  auto res = this->clone_impl();
  c10::raw::intrusive_ptr::incref(res);
  return c10::intrusive_ptr<GeneratorImpl>::reclaim(res);
}

/**
 * Gets the device of a generator.
 */
Device GeneratorImpl::device() const {
  return device_;
}

namespace detail {

/**
 * Gets a random number for /dev/urandom
 * Note this is a legacy method (from THRandom.cpp)
 * FIXME: use std::random_device with entropy information
 */
#ifndef _WIN32
static uint64_t readURandomLong()
{
  int randDev = open("/dev/urandom", O_RDONLY);
  uint64_t randValue;
  TORCH_CHECK(randDev >= 0, "Unable to open /dev/urandom");
  ssize_t readBytes = read(randDev, &randValue, sizeof(randValue));
  TORCH_CHECK(readBytes >= (ssize_t) sizeof(randValue), "Unable to read from /dev/urandom");
  close(randDev);
  return randValue;
}
#endif // _WIN32

/**
 * Gets a non deterministic random number number from either the
 * /dev/urandom or the current time. For CUDA, gets random from
 * std::random_device and adds a transformation on it.
 *
 * FIXME: The behavior in this function is from legacy code (THRandom_seed/THCRandom_seed)
 * and is probably not the right thing to do, even though our tests pass.
 * Figure out if tests get perturbed
 * - when the same algorithm is used for all backends. Note that the current behavior is
 *   different for CPU, CUDA and Windows CPU.
 * - when using C++11 std objects, such as std::random_device
 * - when constructing a 64 bit seed properly, rather than static casting
 *   a 32 bit number to 64 bit.
 */
uint64_t getNonDeterministicRandom(bool is_cuda) {
  uint64_t s;
  if (!is_cuda) {
    #ifdef _WIN32
      s = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    #else
      s = readURandomLong();
    #endif
  } else {
    std::random_device rd;
    // limit to 53 bits to ensure unique representation in double
    s = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
  }
  return s;
}

} // namespace detail
} // namespace c10
