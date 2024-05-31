#pragma once

#include <c10/core/Allocator.h>
#include <torch/csrc/jit/frontend/source_range.h>

#include <ATen/core/ivalue.h>

#include <unordered_map>
#include <vector>

namespace c10 {
struct IValue;
}

namespace torch {
namespace jit {

class Pickler;
class SourceRangeSerializer;
static constexpr size_t kByteOffsetIndex = 0;
static constexpr size_t kSourceRangeIndex = 1;
static constexpr size_t kSourceRangeTagIndex = 2;
constexpr c10::string_view kFormatWithStringTable = "FORMAT_WITH_STRING_TABLE";

class SourceRangePickler {
 public:
  SourceRangePickler();

  std::vector<char> pickle(
      const SourceRangeRecords& ranges,
      const SourceRangeTagMap& source_range_tags);

 private:
  std::shared_ptr<SourceRangeSerializer> srs;
};

class SourceRangeDeserializer {
 public:
  SourceRangeDeserializer() = default;
  explicit SourceRangeDeserializer(const c10::IValue& text_table) {
    for (const auto& x : text_table.toTuple()->elements()) {
      text_table_.emplace_back(std::make_shared<std::string>(x.toStringRef()));
    }
  }
  SourceRange deserialize(const c10::IValue& iv);

 private:
  std::shared_ptr<Source> deserialize_source(const c10::IValue& iv);
  std::unordered_map<
      c10::intrusive_ptr<c10::ivalue::Tuple>,
      std::shared_ptr<Source>>
      cached_sources;
  std::vector<std::shared_ptr<std::string>> text_table_;
};

class SourceRangeUnpickler {
 public:
  virtual std::optional<SourceRange> findSourceRangeThatGenerated(
      const SourceRange& range) = 0;

  virtual ~SourceRangeUnpickler() = default;
};

TORCH_API void setShouldUseFormatWithStringTable(
    bool should_use_format_with_string_table);

} // namespace jit
} // namespace torch
