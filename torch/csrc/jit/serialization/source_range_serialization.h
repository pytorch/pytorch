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
  SourceRange deserialize(const c10::IValue& iv);
 private:
  std::shared_ptr<Source> deserialize_source(const c10::IValue& iv);
  std::unordered_map<
      c10::intrusive_ptr<c10::ivalue::Tuple>,
      std::shared_ptr<Source>>
      cached_sources;
};

class SourceRangeUnpickler {
 public:
  virtual c10::optional<SourceRange> findSourceRangeThatGenerated(
      const SourceRange& range) = 0;

  virtual ~SourceRangeUnpickler() = default;
};

} // namespace jit
} // namespace torch
