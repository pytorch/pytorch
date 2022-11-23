#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/source_range_serialization_impl.h>

#include <c10/util/Exception.h>
#include <c10/util/Flags.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <algorithm>

namespace torch {
namespace jit {

// "Whether to emit compact debug_pkl when saving a model to .pt file."
// "Compact file is smaller but cannot be loaded by old torch binaries."
// TODO(qihan) remove when all binaries are using string table.
thread_local bool should_use_format_with_string_table_ = true;

class SourceRangeSerializer {
 public:
  // Serialize SourceRange as Tuple[SourceType, int, int]
  // where SourceType = Tuple[int, int, int, List[int]],
  // The first 2 ints are positions into the vector returned by textSaved
  // after all the Ranges are processed. textSaved() returns a vector of str
  // the serialized form of Source
  c10::IValue serialize(const SourceRange& sr);

  const std::vector<c10::IValue>& texts_saved() {
    return texts_;
  }

  SourceRangeSerializer() {
    texts_.emplace_back("");
    text_to_idx_[texts_.back().toStringRef()] = 0;
  }

 private:
  // Serialize Source as Tuple[str, Optional[str], int, List[int]]
  // This caches serialized sources, since many SourceRanges can
  // refer to the same one.
  c10::IValue serialize_source(const std::shared_ptr<Source>& s);
  std::unordered_map<std::shared_ptr<Source>, c10::IValue> serialized_sources;

  int64_t store_text_and_get_index(const std::string& text_view);

  std::vector<c10::IValue> texts_;
  std::unordered_map<c10::string_view, int64_t> text_to_idx_;
};

SourceRange SourceRangeDeserializer::deserialize(const c10::IValue& iv) {
  const auto& tup_elems = iv.toTupleRef().elements();
  TORCH_INTERNAL_ASSERT(tup_elems.size() == 3);
  std::shared_ptr<Source> source_ = deserialize_source(tup_elems[0]);
  int64_t start_ = tup_elems[1].toInt();
  int64_t end_ = tup_elems[2].toInt();
  return SourceRange(source_, start_, end_);
}

std::shared_ptr<Source> SourceRangeDeserializer::deserialize_source(
    const c10::IValue& iv) {
  auto tup = iv.toTuple();
  auto it = cached_sources.find(tup);
  if (it != cached_sources.end()) {
    return it->second;
  }
  std::shared_ptr<Source> source;
  const auto& tup_elems = tup->elements();
  TORCH_INTERNAL_ASSERT(tup_elems.size() == 3);
  if (!text_table_.empty()) {
    const auto& textIndex = tup_elems[0].toIntList();
    int64_t fnameIndex = tup_elems[1].toInt();
    int64_t starting_line_no_ = tup_elems[2].toInt();
    c10::optional<std::string> filename = c10::nullopt;

    filename = *text_table_[fnameIndex];

    std::vector<c10::string_view> pieces;
    std::vector<std::shared_ptr<std::string>> strs;

    for (int64_t i : textIndex) {
      pieces.emplace_back(*text_table_[i]);
      strs.emplace_back(text_table_[i]);
    }

    StringCordView str_cord(std::move(pieces), std::move(strs));

    source = std::make_shared<Source>(str_cord, filename, starting_line_no_);
  } else {
    std::string text_ = tup_elems[0].toStringRef();
    c10::optional<std::string> filename_ =
        tup_elems[1].toOptional<std::string>();
    int64_t starting_line_no_ = tup_elems[2].toInt();
    source = std::make_shared<Source>(
        std::move(text_), std::move(filename_), starting_line_no_);
  }
  cached_sources[tup] = source;
  return source;
}

c10::IValue SourceRangeSerializer::serialize(const SourceRange& sr) {
  return c10::ivalue::Tuple::create(
      serialize_source(sr.source()), (int64_t)sr.start(), (int64_t)sr.end());
}

int64_t SourceRangeSerializer::store_text_and_get_index(
    const std::string& text_view) {
  auto text_iter = text_to_idx_.find(text_view);
  if (text_iter == text_to_idx_.end()) {
    int64_t text_pos = static_cast<int64_t>(texts_.size());
    texts_.emplace_back(text_view);
    text_to_idx_[texts_.back().toStringView()] = text_pos;
    return text_pos;
  } else {
    return text_iter->second;
  }
}

c10::IValue SourceRangeSerializer::serialize_source(
    const std::shared_ptr<Source>& s) {
  if (serialized_sources.count(s)) {
    return serialized_sources.at(s);
  }
  c10::intrusive_ptr<c10::ivalue::Tuple> serialized;
  c10::List<int64_t> lines;
  if (should_use_format_with_string_table_) {
    if (s == nullptr) {
      serialized = c10::ivalue::Tuple::create({lines, 0, 0});
    } else {
      for (size_t lineno = 0; lineno < s->num_lines(); lineno++) {
        std::string line_content = s->get_line(lineno).str();
        int64_t text_pos = store_text_and_get_index(line_content);
        lines.push_back(text_pos);
      }

      int64_t fname_pos = 0;
      if (s->filename().has_value()) {
        fname_pos = store_text_and_get_index(*s->filename());
      }
      serialized = c10::ivalue::Tuple::create(
          {lines, fname_pos, (int64_t)s->starting_line_no()});
    }
  } else {
    if (s == nullptr) {
      serialized = c10::ivalue::Tuple::create({"", "", 0});
    } else {
      serialized = c10::ivalue::Tuple::create(
          {s->text_str().str(), s->filename(), (int64_t)s->starting_line_no()});
    }
  }
  serialized_sources[s] = serialized;
  return serialized;
}

SourceRangePickler::SourceRangePickler() : srs(new SourceRangeSerializer()) {}

std::vector<char> SourceRangePickler::pickle(
    const SourceRangeRecords& ranges,
    const SourceRangeTagMap& source_range_tags) {
  std::vector<c10::IValue> ivalues;
  for (const auto& range : ranges) {
    int64_t source_range_tag{-1};
    const auto& it = source_range_tags.find(range.range);
    if (it != source_range_tags.end()) {
      source_range_tag = it->second;
    }

    ivalues.emplace_back(c10::ivalue::Tuple::create(
        {(int64_t)range.bytes,
         srs->serialize(range.range),
         static_cast<int64_t>(source_range_tag)}));
  }

  std::vector<at::Tensor> table;
  auto textTable = c10::ivalue::Tuple::create(srs->texts_saved());
  auto ivalue = c10::ivalue::Tuple::create(std::move(ivalues));
  std::vector<char> result;
  if (should_use_format_with_string_table_) {
    result = jit::pickle(
        c10::ivalue::Tuple::create({kFormatWithStringTable, textTable, ivalue}),
        &table);
  } else {
    result = jit::pickle(ivalue, &table);
  }
  TORCH_CHECK(table.size() == 0, "Expected 0 tensors to be written");
  return result;
}

ConcreteSourceRangeUnpickler::ConcreteSourceRangeUnpickler(
    at::DataPtr&& data,
    size_t size)
    : data(std::move(data)),
      size(size),
      deserializer(nullptr),
      unpickled_records(nullptr) {}

void ConcreteSourceRangeUnpickler::unpickle() {
  std::lock_guard<std::mutex> guard(mutex);
  if (unpickled_records) {
    return;
  }

  auto ivaluesTuple = jit::unpickle(
                          reinterpret_cast<const char*>(data.get()),
                          size,
                          nullptr,
                          {},
                          c10::parseType)
                          .toTuple();

  const auto& ivalues = ivaluesTuple->elements();
  unpickled_records = std::make_shared<SourceRangeRecords>();
  IValue lines;
  if (ivalues[0].isString() &&
      kFormatWithStringTable == ivalues[0].toStringRef()) {
    deserializer.reset(new SourceRangeDeserializer(ivalues[1]));
    lines = ivalues[2];
  } else {
    deserializer.reset(new SourceRangeDeserializer());
    lines = ivaluesTuple;
  }
  for (auto& val : lines.toTuple()->elements()) {
    const auto& tup_elems = val.toTupleRef().elements();
    int64_t offset = tup_elems[kByteOffsetIndex].toInt();
    auto source_range = deserializer->deserialize(tup_elems[kSourceRangeIndex]);
    unpickled_records->emplace_back(offset, std::move(source_range));
  }
}

c10::optional<SourceRange> ConcreteSourceRangeUnpickler::
    findSourceRangeThatGenerated(const SourceRange& range) {
  unpickle();

  auto query = TaggedRange(range.start(), SourceRange{});
  auto entry = std::upper_bound(
      unpickled_records->begin(),
      unpickled_records->end(),
      query,
      [](const TaggedRange& a, const TaggedRange& b) -> bool {
        return a.bytes < b.bytes;
      });

  // NB: must decrement iterator since upper_bound finds the element
  // *greater than* the query.
  if (entry != unpickled_records->begin()) {
    return (entry - 1)->range;
  }

  return c10::nullopt;
}

TORCH_API void setShouldUseFormatWithStringTable(
    bool should_use_format_with_string_table) {
  should_use_format_with_string_table_ = should_use_format_with_string_table;
}

} // namespace jit
} // namespace torch
