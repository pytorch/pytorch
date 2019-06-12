#include "caffe2/core/context.h"

#include <atomic>
#if defined(_MSC_VER)
#include <process.h>
#endif

namespace caffe2 {

uint32_t RandomNumberSeed() {
  // Originally copied from folly::randomNumberSeed (at 418ad4)
  // modified to use chrono instead of sys/time.h
  static std::atomic<uint32_t> seedInput(0);
  auto tv = std::chrono::system_clock::now().time_since_epoch();
  uint64_t usec = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(tv).count());
  uint32_t tv_sec = usec / 1000000;
  uint32_t tv_usec = usec % 1000000;
  const uint32_t kPrime0 = 51551;
  const uint32_t kPrime1 = 61631;
  const uint32_t kPrime2 = 64997;
  const uint32_t kPrime3 = 111857;
  return kPrime0 * (seedInput++) + kPrime1 * static_cast<uint32_t>(getpid()) +
      kPrime2 * tv_sec + kPrime3 * tv_usec;
}

namespace {
inline void CopyBytesImpl(size_t nbytes, const void* src, void* dst) {
  if (nbytes == 0) {
    return;
  }
  CAFFE_ENFORCE(src);
  CAFFE_ENFORCE(dst);
  memcpy(dst, src, nbytes);
}

void CopyBytesWrapper(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device) {
  CopyBytesImpl(nbytes, src, dst);
}
} // namespace

void CPUContext::CopyBytesSameDevice(
    size_t nbytes,
    const void* src,
    void* dst) {
  CopyBytesImpl(nbytes, src, dst);
}

} // namespace caffe2

namespace at {

REGISTER_CONTEXT(DeviceType::CPU, caffe2::CPUContext);

REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CPU,
    DeviceType::CPU,
    caffe2::CopyBytesWrapper);
} // namespace at
