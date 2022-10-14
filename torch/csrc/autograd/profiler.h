#pragma once

#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#include <torch/csrc/profiler/standalone/profiler_legacy.h>

// There are some components which use these symbols. Until we migrate them
// we have to mirror them in the old autograd namespace.
namespace torch {
namespace autograd {
namespace profiler {
using namespace ::torch::profiler_legacy;
using ::torch::profiler::impl::ActivityType;
using ::torch::profiler::impl::getProfilerConfig;
using ::torch::profiler::impl::ProfilerConfig;
using ::torch::profiler::impl::profilerEnabled;
using ::torch::profiler::impl::ProfilerState;
} // namespace profiler
} // namespace autograd
} // namespace torch
