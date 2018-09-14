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

REGISTER_CONTEXT(CPU, CPUContext);

BaseStaticContext* GetCPUStaticContext() {
  static CPUStaticContext context;
  return &context;
}

REGISTER_STATIC_CONTEXT(CPU, GetCPUStaticContext());

struct ExtractDeviceOptionCPU : public ExtractDeviceOptionFn {
  void operator()(DeviceOption* device, const void* data) override {
    device->set_device_type(PROTO_CPU);
  }
};

ExtractDeviceOptionFn* GetExtractDeviceOptionCPU() {
  static ExtractDeviceOptionCPU fn;
  return &fn;
}

REGISTER_DEVICE_OPTION_FN(CPU, GetExtractDeviceOptionCPU);

} // namespace caffe2
