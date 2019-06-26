#pragma once

#include <c10/core/Allocator.h>
#include <torch/csrc/jit/source_range.h>

#include <unordered_map>
#include <vector>

namespace c10 {
class IValue;
}

namespace torch {
namespace jit {

class Pickler;
class SourceRangeSerializer;
class SourceRangeDeserializer;

class SourceRangePickler {
 public:
  SourceRangePickler();

  void pickle(const SourceRangeRecords& ranges);

  const std::vector<char>& get_data();

 private:
  std::shared_ptr<Pickler> p;
  std::shared_ptr<SourceRangeSerializer> srs;
};

class SourceRangeUnpickler {
 public:
  SourceRangeUnpickler(at::DataPtr&& data, size_t size);

  c10::optional<SourceRange> findSourceRangeThatGenerated(
      const SourceRange& range);

 private:
  at::DataPtr data;
  size_t size;

  void unpickle();

  std::shared_ptr<SourceRangeDeserializer> deserializer;
  std::shared_ptr<SourceRangeRecords> unpickled_records;
};

} // namespace jit
} // namespace torch