#pragma once

#include <c10/core/Allocator.h>
#include <torch/csrc/jit/frontend/source_range.h>

#include <unordered_map>
#include <vector>

namespace c10 {
struct IValue;
}

namespace torch {
namespace jit {

class Pickler;
class SourceRangeSerializer;
class SourceRangeDeserializer;

class SourceRangePickler {
 public:
  SourceRangePickler();

  std::vector<char> pickle(const SourceRangeRecords& ranges);

 private:
  std::shared_ptr<SourceRangeSerializer> srs;
};

class SourceRangeUnpickler {
 public:
  virtual c10::optional<SourceRange> findSourceRangeThatGenerated(
      const SourceRange& range) = 0;

  virtual ~SourceRangeUnpickler() = default;
};

} // namespace jit
} // namespace torch
