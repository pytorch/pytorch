#include <torch/csrc/profiler/stubs/base.h>
#include <chrono>

namespace torch_vulkan {

// No-op profiler stubs for PrivateUse1 (Vulkan).
// Enables torch.profiler to run without crashing; actual GPU timing
// can be added later with Vulkan timestamp queries.
struct VulkanProfilerStubs : public torch::profiler::impl::ProfilerStubs {
    void record(
        c10::DeviceIndex* /*device*/,
        torch::profiler::impl::ProfilerVoidEventStub* /*event*/,
        int64_t* cpu_ns) const override {
        // Record CPU timestamp as stand-in
        if (cpu_ns) {
            *cpu_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        }
    }

    float elapsed(
        const torch::profiler::impl::ProfilerVoidEventStub* /*event*/,
        const torch::profiler::impl::ProfilerVoidEventStub* /*event2*/) const override {
        return 0.0f;
    }

    void mark(const char* /*name*/) const override {}
    void rangePush(const char* /*name*/) const override {}
    void rangePop() const override {}

    bool enabled() const override { return false; }

    void onEachDevice(std::function<void(int)> op) const override {
        // Single device (vulkan:0)
        op(0);
    }

    void synchronize() const override {
        // No-op: single-stream backend, already synchronized
    }
};

static VulkanProfilerStubs vulkan_profiler_stubs;

// Called during module init to register stubs
void register_profiler_stubs() {
    torch::profiler::impl::registerPrivateUse1Methods(&vulkan_profiler_stubs);
}

} // namespace torch_vulkan
