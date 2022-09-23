#pragma once

#include <functional>
#include <memory>

#include <torch/csrc/Export.h>

struct CUevent_st;

namespace torch {
namespace profiler {
namespace impl {

// ----------------------------------------------------------------------------
// -- Annotation --------------------------------------------------------------
// ----------------------------------------------------------------------------
using ProfilerEventStub = std::shared_ptr<CUevent_st>;

struct TORCH_API ProfilerStubs {
  virtual void record(int* device, ProfilerEventStub* event, int64_t* cpu_ns)
      const = 0;
  virtual float elapsed(
      const ProfilerEventStub* event,
      const ProfilerEventStub* event2) const = 0;
  virtual void mark(const char* name) const = 0;
  virtual void rangePush(const char* name) const = 0;
  virtual void rangePop() const = 0;
  virtual bool enabled() const {
    return false;
  }
  virtual void onEachDevice(std::function<void(int)> op) const = 0;
  virtual void synchronize() const = 0;
  virtual ~ProfilerStubs();
};

TORCH_API void registerCUDAMethods(ProfilerStubs* stubs);
TORCH_API const ProfilerStubs* cudaStubs();
TORCH_API void registerITTMethods(ProfilerStubs* stubs);
TORCH_API const ProfilerStubs* ittStubs();

} // namespace impl
} // namespace profiler
} // namespace torch
