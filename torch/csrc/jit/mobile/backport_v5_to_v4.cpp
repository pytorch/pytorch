#include <ATen/core/ivalue.h>
//#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/backport_factory.h>
#include <torch/csrc/jit/mobile/model_compatibility.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/pickler.h>

namespace torch {
namespace jit {

using c10::IValue;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using caffe2::serialize::ReadAdapterInterface;
using torch::jit::TensorIndexMap;

namespace {

void writeArchive(
    PyTorchStreamWriter& writer,
    const std::string& archive_name,
    const IValue& value) {
  std::vector<char> data;

  // Vector to capture the run-time class types during pickling the IValues
  std::vector<c10::ClassTypePtr> memoizedClassTypes;
  Pickler data_pickle(
      [&](const char* buf, size_t size) {
        data.insert(data.end(), buf, buf + size);
      },
      nullptr,
      nullptr,
      &memoizedClassTypes);
  data_pickle.protocol();
  data_pickle.pushIValue(value);
  data_pickle.stop();
  size_t i = 0;
  std::string prefix = archive_name + "/";

  for (const auto& td : data_pickle.tensorData()) {
    WriteableTensorData writable_td = getWriteableTensorData(td);
    std::string fname = prefix + c10::to_string(i++);
    writer.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
  }
  std::string fname = archive_name + ".pkl";
  writer.writeRecord(fname, data.data(), data.size());
}

} // namespace

// The family of methods below load a serialized Mobile Module
bool backport_v5_to_v4(
    PyTorchStreamReader& reader,
    PyTorchStreamWriter& writer) {
  // 1) read from archive `bytecode` archive
  std::vector<IValue> bytecode_values = get_bytecode_values(reader);
  if (!check_bytecode_version(bytecode_values, kBytecodeVersionV5)) {
    TORCH_WARN("Incorrect bytecode version for input model.");
    return false;
  }

  // 2) Copy everything except bytecode related to new output
  copy_non_bytecode(reader, writer);

  // 3) write `bytecode` archive
  // Update the bytecode version in bytecode.pkl
  update_bytecode_version(bytecode_values, kBytecodeVersionV4);
  // Construct the list of ivalues to a big tuple
  auto bytecode_tuple = c10::ivalue::Tuple::create(std::move(bytecode_values));
  // write `bytecode` archive
  writeArchive(writer, "bytecode", bytecode_tuple);
  return true;
}

} // namespace jit
} // namespace torch
