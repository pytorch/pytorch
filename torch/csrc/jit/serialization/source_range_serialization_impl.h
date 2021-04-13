#pragma once

#include <torch/csrc/jit/serialization/source_range_serialization.h>

namespace torch {
namespace jit {

// Do this clownyness with virtual functions because of the split
// between ATen core and torch

class ConcreteSourceRangeUnpickler : public SourceRangeUnpickler {
 public:
  ConcreteSourceRangeUnpickler(at::DataPtr&& data, size_t size);

  c10::optional<SourceRange> findSourceRangeThatGenerated(
      const SourceRange& range) override;

 private:
  at::DataPtr data;
  size_t size;

  void unpickle();

  std::shared_ptr<SourceRangeDeserializer> deserializer;
  std::shared_ptr<SourceRangeRecords> unpickled_records;
};

} // namespace jit
} // namespace torch
