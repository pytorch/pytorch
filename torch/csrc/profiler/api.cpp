#include <torch/csrc/profiler/api.h>

#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

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
