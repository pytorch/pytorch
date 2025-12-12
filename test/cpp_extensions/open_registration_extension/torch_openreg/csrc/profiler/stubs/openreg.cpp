#include <sstream>

#include <c10/core/DeviceGuard.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

#include <include/openreg.h>

#include "runtime/OpenRegFunctions.h"
#include "runtime/OpenRegStream.h"


namespace torch::profiler::impl {
namespace {

static void openregCheck(orError_t result, const char* file, int line) {
  if (result != orSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": ";
    if (result == orErrorNotReady) {
      ss << "OpenReg operation not ready";
    } else {
      ss << "OpenReg error: " << result;
    }
    TORCH_CHECK(false, ss.str());
  }
}
#define TORCH_OPENREG_CHECK(result) openregCheck(result, __FILE__, __LINE__);

struct OpenRegMethods : public ProfilerStubs {
  void record(
      c10::DeviceIndex* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const override {
    auto stream = c10::openreg::getCurrentOpenRegStream();

    // Get current device if requested
    if (device) {
      *device = c10::openreg::current_device();
    }

    // Create OpenReg event
    orEvent_t openreg_event_ptr{nullptr};
    TORCH_OPENREG_CHECK(orEventCreateWithFlags(&openreg_event_ptr, orEventEnableTiming));
    *event = std::shared_ptr<orEvent>(openreg_event_ptr, [](orEvent_t ptr) {
      orEventDestroy(ptr);
    });

    // Record CPU timestamp if requested
    if (cpu_ns) {
      *cpu_ns = c10::getTime();
    }

    // Record event on stream
    TORCH_OPENREG_CHECK(orEventRecord(openreg_event_ptr, stream.stream()));
  }

  float elapsed(
      const ProfilerVoidEventStub* event_,
      const ProfilerVoidEventStub* event2_) const override {
    // Cast to shared_ptr<orEvent> - similar to CUDA's ProfilerEventStub cast
    auto event = reinterpret_cast<const std::shared_ptr<orEvent>*>(event_);
    auto event2 = reinterpret_cast<const std::shared_ptr<orEvent>*>(event2_);

    // Check if events are valid
    if (!event || !(*event) || !event2 || !(*event2)) {
      return 0.0f;
    }

    TORCH_OPENREG_CHECK(orEventSynchronize(event->get()));
    TORCH_OPENREG_CHECK(orEventSynchronize(event2->get()));

    float ms = 0;
    TORCH_OPENREG_CHECK(orEventElapsedTime(&ms, event->get(), event2->get()));
    // Convert milliseconds to microseconds
    return ms * 1000.0;
  }

  void mark(const char* name) const override {
    // OpenReg doesn't have built-in annotation support like NVTX
    // This is a no-op for KINETO_PRIVATEUSE1_FALLBACK mode
    // PRIVATEUSE1 mode will use OpenReg defined `enter()` and `exit()` instead
  }

  void rangePush(const char* name) const override {
    // OpenReg doesn't have built-in annotation support like NVTX
    // This is a no-op for KINETO_PRIVATEUSE1_FALLBACK mode
    // PRIVATEUSE1 mode will use OpenReg defined `enter()` and `exit()` instead
  }

  void rangePop() const override {
    // OpenReg doesn't have built-in annotation support like NVTX
    // This is a no-op for KINETO_PRIVATEUSE1_FALLBACK mode
    // PRIVATEUSE1 mode will use OpenReg defined `enter()` and `exit()` instead
  }

  void onEachDevice(std::function<void(int)> op) const override {
    c10::DeviceGuard device_guard(c10::DeviceType::PrivateUse1);
    int device_count = c10::openreg::device_count();
    for (const auto i : c10::irange(device_count)) {
      device_guard.set_index(i);
      op(i);
    }
  }

  void synchronize() const override {
    TORCH_OPENREG_CHECK(orDeviceSynchronize());
  }

  bool enabled() const override {
    return true;
  }
};

struct RegisterOpenRegMethods {
  RegisterOpenRegMethods() {
    static OpenRegMethods methods;
    registerPrivateUse1Methods(&methods);
  }
};
RegisterOpenRegMethods reg;

} // namespace
} // namespace torch::profiler::impl
