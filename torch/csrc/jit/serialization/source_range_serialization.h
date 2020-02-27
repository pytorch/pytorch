
copy: fbcode/caffe2/torch/csrc/jit/serialization/source_range_serialization.h
copyrev: eb2e6dc173564b050658d8c0b3348ac67c8ff730

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

  virtual ~SourceRangeUnpickler() {}
};

} // namespace jit
} // namespace torch
