#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <c10/util/ArrayRef.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/collection.h>

namespace torch::autograd::profiler {

// Annotate the libkineto activity backing each Result with typed metadata
// (input shapes, call stacks, allocation sizes, ...). When
// Kineto is disabled the metadata annotation is a no-op.
void addTensorboardFields(
    const std::shared_ptr<torch::profiler::impl::Result>& result,
    c10::ArrayRef<std::string> module_hierarchy,
    c10::ArrayRef<std::string> stack);

void addGenericMetadata(
    std::shared_ptr<torch::profiler::impl::Result>& result,
    const torch::profiler::impl::ProfilerConfig* config);

// Lightweight metadata pass for trace_only mode: annotates Kineto activities
// with the same metadata as materializeOpEvents but without creating
// KinetoEvent wrappers or building eventTree.
void addTraceMetadata(
    std::vector<std::shared_ptr<torch::profiler::impl::Result>>& events,
    const torch::profiler::impl::ProfilerConfig& config,
    int64_t trace_end_ns);

} // namespace torch::autograd::profiler
