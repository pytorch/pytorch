#include <torch/csrc/profiler/api.h>

namespace torch {
namespace profiler {
namespace impl {

ExperimentalConfig::ExperimentalConfig(
    std::vector<std::string> profiler_metrics,
    bool profiler_measure_per_kernel)
    : profiler_metrics{profiler_metrics},
      profiler_measure_per_kernel{profiler_measure_per_kernel} {}

ExperimentalConfig::operator bool() const {
  return !profiler_metrics.empty();
}

bool ProfilerConfig::disabled() const {
  return state == torch::profiler::impl::ProfilerState::Disabled;
}

bool ProfilerConfig::global() const {
  return state == torch::profiler::impl::ProfilerState::KINETO_ONDEMAND;
}

namespace {
enum ProfilerIValueIdx {
  STATE = 0,
  REPORT_INPUT_SHAPES,
  PROFILE_MEMORY,
  NUM_PROFILER_CFG_IVALUE_IDX // must be last in list
};
} // namespace

at::IValue ProfilerConfig::toIValue() const {
  c10::impl::GenericList eventIValueList(at::AnyType::get());
  eventIValueList.reserve(NUM_PROFILER_CFG_IVALUE_IDX);
  eventIValueList.emplace_back(static_cast<int64_t>(state));
  eventIValueList.emplace_back(report_input_shapes);
  eventIValueList.emplace_back(profile_memory);
  return eventIValueList;
}

ProfilerConfig ProfilerConfig::fromIValue(
    const at::IValue& profilerConfigIValue) {
  TORCH_INTERNAL_ASSERT(
      profilerConfigIValue.isList(),
      "Expected IValue to contain type c10::impl::GenericList");
  auto ivalues = profilerConfigIValue.toList();
  TORCH_INTERNAL_ASSERT(
      ivalues.size() == NUM_PROFILER_CFG_IVALUE_IDX,
      c10::str(
          "Expected exactly ",
          NUM_PROFILER_CFG_IVALUE_IDX,
          " ivalues to resconstruct ProfilerConfig."));
  return ProfilerConfig(
      static_cast<ProfilerState>(ivalues.get(ProfilerIValueIdx::STATE).toInt()),
      ivalues.get(ProfilerIValueIdx::REPORT_INPUT_SHAPES).toBool(),
      ivalues.get(ProfilerIValueIdx::PROFILE_MEMORY).toBool());
}

bool profilerEnabled() {
  auto state_ptr = ProfilerThreadLocalStateBase::getTLS();
  return state_ptr && !state_ptr->config().disabled();
}

TORCH_API ActiveProfilerType profilerType() {
  auto state_ptr = ProfilerThreadLocalStateBase::getTLS();
  return state_ptr == nullptr ? ActiveProfilerType::NONE
                              : state_ptr->profilerType();
}

torch::profiler::impl::ProfilerConfig getProfilerConfig() {
  auto state_ptr = ProfilerThreadLocalStateBase::getTLS();
  TORCH_CHECK(
      state_ptr,
      "Tried to access profiler config, but profiler is not enabled!");
  return state_ptr->config();
}

ProfilerStubs::~ProfilerStubs() = default;

namespace {
struct DefaultCUDAStubs : public ProfilerStubs {
  void record(
      int* /*device*/,
      ProfilerEventStub* /*event*/,
      int64_t* /*cpu_ns*/) const override {
    fail();
  }
  float elapsed(
      const ProfilerEventStub* /*event*/,
      const ProfilerEventStub* /*event2*/) const override {
    fail();
    return 0.f;
  }
  void mark(const char* /*name*/) const override {
    fail();
  }
  void rangePush(const char* /*name*/) const override {
    fail();
  }
  void rangePop() const override {
    fail();
  }
  bool enabled() const override {
    return false;
  }
  void onEachDevice(std::function<void(int)> /*op*/) const override {
    fail();
  }
  void synchronize() const override {
    fail();
  }
  ~DefaultCUDAStubs() override = default;

 private:
  void fail() const {
    AT_ERROR("CUDA used in profiler but not enabled.");
  }
};

const DefaultCUDAStubs default_cuda_stubs;
constexpr const DefaultCUDAStubs* default_cuda_stubs_addr = &default_cuda_stubs;
// Constant initialization, so it is guaranteed to be initialized before
// static initialization calls which may invoke registerCUDAMethods
inline const ProfilerStubs*& cuda_stubs() {
  static const ProfilerStubs* stubs_ =
      static_cast<const ProfilerStubs*>(default_cuda_stubs_addr);
  return stubs_;
}

struct DefaultITTStubs : public ProfilerStubs {
  void record(
      int* /*device*/,
      ProfilerEventStub* /*event*/,
      int64_t* /*cpu_ns*/) const override {
    fail();
  }
  float elapsed(
      const ProfilerEventStub* /*event*/,
      const ProfilerEventStub* /*event2*/) const override {
    fail();
    return 0.f;
  }
  void mark(const char* /*name*/) const override {
    fail();
  }
  void rangePush(const char* /*name*/) const override {
    fail();
  }
  void rangePop() const override {
    fail();
  }
  bool enabled() const override {
    return false;
  }
  void onEachDevice(std::function<void(int)> /*op*/) const override {
    fail();
  }
  void synchronize() const override {
    fail();
  }
  ~DefaultITTStubs() override = default;

 private:
  void fail() const {
    AT_ERROR("ITT used in profiler but not enabled.");
  }
};

const DefaultITTStubs default_itt_stubs;
constexpr const DefaultITTStubs* default_itt_stubs_addr = &default_itt_stubs;
// Constant initialization, so it is guaranteed to be initialized before
// static initialization calls which may invoke registerITTMethods
inline const ProfilerStubs*& itt_stubs() {
  static const ProfilerStubs* stubs_ =
      static_cast<const ProfilerStubs*>(default_itt_stubs_addr);
  return stubs_;
}
} // namespace

const ProfilerStubs* cudaStubs() {
  return cuda_stubs();
}

void registerCUDAMethods(ProfilerStubs* stubs) {
  cuda_stubs() = stubs;
}

const ProfilerStubs* ittStubs() {
  return itt_stubs();
}

void registerITTMethods(ProfilerStubs* stubs) {
  itt_stubs() = stubs;
}

} // namespace impl
} // namespace profiler
} // namespace torch
