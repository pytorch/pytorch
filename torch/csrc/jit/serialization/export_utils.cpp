#include <torch/csrc/jit/serialization/export_utils.h>

namespace torch {
namespace jit {

bool isLoweredModule(const Module& m) {
  c10::QualifiedName type_name;
  if (m.type()->name()) {
    type_name = m.type()->name().value();
  }
  bool isLoweredModule = false;
  for (const auto& atom : type_name.atoms()) {
    if (atom == "LoweredModule") {
      isLoweredModule = true;
      break;
    }
  }
  return isLoweredModule;
}

SourceRangeRecords getBackendSourceRanges(const Module& m) {
  SourceRangeRecords sr_records;
  if (isLoweredModule(m)) {
    constexpr size_t kSourceRange = 1;
    auto backend_debug_info =
        m.attr("__backend_debug_info").toCustomClass<PyTorchBackendDebugInfo>();
    const auto& map = backend_debug_info->getDebugInfoMap();
    if (map) {
      const auto& map_val = map.value();
      // This map is map of debug handle-to-DebugInfoTuple
      // DebugInfoTuple= <source range, op name, inlined_cs_ptr>
      for (const auto& it : map_val) {
        auto& source_range =
            std::get<kDebugInfoTupleSourceRangeIndex>(it.second);
        sr_records.emplace_back(
            std::numeric_limits<size_t>::max(), source_range);
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        auto cs_ptr = std::get<kDebugInfoTupleInlinedCSIndex>(it.second);
        if (cs_ptr) {
          for (const auto& e : cs_ptr->vec()) {
            // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
            const auto sr = std::get<kSourceRange>(e);
            sr_records.emplace_back(std::numeric_limits<size_t>::max(), sr);
          }
        }
      }
    }
  }
  for (const auto& c : m.children()) {
    const auto& child_sr_records = getBackendSourceRanges(c);
    sr_records.reserve(sr_records.size() + child_sr_records.size());
    std::move(
        child_sr_records.begin(),
        child_sr_records.end(),
        std::back_inserter(sr_records));
  }
  return sr_records;
}

void updateSourceRangeTags(
    const SourceRangeRecords& ranges,
    SourceRangeTagMap& m_source_range_tags,
    int64_t* m_current_source_range_tag) {
  for (const auto& range : ranges) {
    if (m_source_range_tags.find(range.range) == m_source_range_tags.end()) {
      m_source_range_tags[range.range] = *m_current_source_range_tag;
      (*m_current_source_range_tag)++;
    }
  }
}

} // namespace jit
} // namespace torch
