#include <torch/csrc/jit/mobile/debug_info.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <c10/util/string_view.h>

namespace torch {
namespace jit {

MobileDebugTable::MobileDebugTable(
    std::unique_ptr<caffe2::serialize::PyTorchStreamReader>& reader) {
  const std::vector<std::string>& record_names = reader->getAllRecords();
  const c10::string_view suffix("debug_pkl");
  for (const auto& record_name : record_names) {
    if (c10::string_view(
          record_name.data(), record_name.size()).ends_with(suffix)) {
      at::DataPtr debug_data;
      size_t debug_size;
      std::tie(debug_data, debug_size) = reader->getRecord(record_name);
      auto ivalues =
        jit::unpickle(
            reinterpret_cast<const char*>(debug_data.get()), debug_size)
        .toTuple()
        ->elements();
      std::unique_ptr<SourceRangeDeserializer> deserializer =
        std::make_unique<SourceRangeDeserializer>();
      for (auto& val : ivalues) {
        auto tup_elems = val.toTuple()->elements();
        TORCH_CHECK(tup_elems.size() == 3,
            "Source debug tuple must have three elements:"
            "byte_offset, source_tag, source_range");
        int64_t debug_handle = tup_elems[1].toInt();
        auto source_range = deserializer->deserialize(tup_elems[2]);
        source_range_map_.emplace(debug_handle, std::move(source_range));
      }
    }
  }
}

std::string MobileDebugTable::getSourceDebugString(const int64_t debug_handle) {
  const auto it = source_range_map_.find(debug_handle);
  if (it == source_range_map_.end()) {
    return "";
  }
  return source_range_map_[debug_handle].str();
}

} // namespace jit
} // namespace torch
