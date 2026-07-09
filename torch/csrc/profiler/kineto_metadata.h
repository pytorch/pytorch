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
// (input shapes, call stacks, allocation sizes, ...). These entry points name
// no libkineto type and are safe to call unconditionally: all libkineto
// typed-metadata usage (the GenericMetadataFields catalog, MetadataField) is
// confined to kineto_metadata.cpp and compiled only under USE_KINETO. When
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
// KinetoEvent wrappers or building eventTree. The unfinished-event end-time
// fixup runs regardless of Kineto; only the annotation is Kineto-gated.
void addTraceMetadata(
    std::vector<std::shared_ptr<torch::profiler::impl::Result>>& events,
    const torch::profiler::impl::ProfilerConfig& config,
    int64_t trace_end_ns);

} // namespace torch::autograd::profiler
