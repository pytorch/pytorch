#pragma once
#include <torch/csrc/jit/frontend/source_range.h>
#include <caffe2/serialize/inline_container.h>
#include <c10/util/flat_hash_map.h>

namespace torch {
namespace jit {
class MobileDebugTable {
 public:
  MobileDebugTable(
      std::unique_ptr<caffe2::serialize::PyTorchStreamReader>& reader);
  std::string getSourceDebugString(const int64_t debug_handle);
 private:
  ska::flat_hash_map<int64_t, SourceRange> source_range_map_;
};

} // namespace jit
} // namespace torch
