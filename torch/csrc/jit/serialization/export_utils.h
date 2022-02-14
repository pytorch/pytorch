#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_debug_info.h>
#include <torch/csrc/jit/frontend/source_range.h>

namespace torch {
namespace jit {
SourceRangeRecords getBackendSourceRanges(const Module& m);

void updateSourceRangeTags(
    const SourceRangeRecords& ranges,
    SourceRangeTagMap& m_source_range_tags,
    int64_t* m_current_source_range_tag);

bool isLoweredModule(const Module& m);

} // namespace jit
} // namespace torch
