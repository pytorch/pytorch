#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/source_range.h>

namespace torch {
namespace jit {

class SourceRangeSerializer {
 public:
  // Serialize SourceRange as Tuple[SourceType, int, int]
  // where SourceType = Tuple[str, Optional[str], int, List[int]],
  // the serialized form of Source
  c10::IValue serialize(const SourceRange& sr) {
    std::vector<c10::IValue> elements = {
        serialize_source(sr.source()), (int64_t)sr.start(), (int64_t)sr.end()};
    return c10::ivalue::Tuple::create(std::move(elements));
  }

 private:
  // Serialize Source as Tuple[str, Optional[str], int, List[int]]
  // This caches serialized sources, since many SourceRanges can
  // refer to the same one.
  c10::IValue serialize_source(const std::shared_ptr<Source>& s) {
    if (serialized_sources.count(s)) {
      return serialized_sources.at(s);
    }
    std::vector<c10::IValue> elements{
        s->text(), s->filename(), (int64_t)s->starting_line_no()};
    auto serialized = c10::ivalue::Tuple::create(std::move(elements));
    serialized_sources[s] = serialized;
    return serialized;
  }

  std::unordered_map<std::shared_ptr<Source>, c10::IValue> serialized_sources;
};

} // namespace jit
} // namespace torch