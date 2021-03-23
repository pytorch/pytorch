#pragma once
#include <c10/util/flat_hash_map.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/frontend/source_range.h>

namespace torch {
namespace jit {
/*
 * MobileDebugTable:
 * Deserializes debug_pkl records from PT model's zip archive and
 * stores them in a map of debug handles to source range.
 * Debug handles are unique per model and runtime, be in lite interpreter
 * or delegate, raises exception using debug handles.
 * getSourceDebugString method is responsible for translating debug
 * handles to correspond debug information.
 * At the moment this only contains information about model source.
 * But later diffs will include entire stack corresponding to debug handle.
 */
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
