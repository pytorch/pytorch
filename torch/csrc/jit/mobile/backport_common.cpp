#include <ATen/core/ivalue.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/backport_factory.h>

namespace torch {
namespace jit {

using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;

bool update_bytecode_version(
    std::vector<at::IValue>& bytecode_values,
    const int64_t to_version) {
  if (!bytecode_values.empty() && bytecode_values[0].isInt()) {
    bytecode_values[0] = at::IValue(to_version);
    return true;
  }
  return false;
}

void copy_non_bytecode(
    PyTorchStreamReader& reader,
    PyTorchStreamWriter& writer) {
  auto records = reader.getAllRecords();
  for (const auto& record : records) {
    // Don't copy archive `version` and archive `bytecode`
    // Archvie `version` will be written when PyTorchStreamWriter is going to
    // finalize and run writeEndOfFile()
    if (record != kArchiveNameVersion &&
        record.find(kArchiveNameBytecode) == std::string::npos) {
      auto data_ptr = reader.getRecord(record);
      auto data = std::get<0>(data_ptr).get();
      auto size = std::get<1>(data_ptr);
      writer.writeRecord(record, data, size);
    }
  }
}

bool check_bytecode_version(
    const std::vector<c10::IValue>& bytecode_values,
    const int64_t expect_bytecode_version) {
  if (bytecode_values.empty()) {
    TORCH_WARN("Empty bytecode archive.");
    return false;
  } else if (bytecode_values[0] != expect_bytecode_version) {
    TORCH_WARN(
        "Expect bytecode version ",
        expect_bytecode_version,
        ", but it gets ",
        bytecode_values[0]);
    return false;
  }
  return true;
}

} // namespace jit
} // namespace torch
