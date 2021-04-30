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
TensorIndexMap get_tensors_archive_table(const IValue& value) {
  std::vector<char> data;
  TensorIndexMap tensors_archive_table;
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

  const auto tensor_candidates = data_pickle.tensorData();
  for (size_t tensor_index = 0; tensor_index < tensor_candidates.size();
       tensor_index++) {
    tensors_archive_table[tensor_candidates[tensor_index]] =
        std::make_pair(kArchiveNameConstants, tensor_index);
  }
  return tensors_archive_table;
}

void writeArchive(
    PyTorchStreamWriter& writer,
    const std::string& archive_name,
    const IValue& value,
    const TensorIndexMap& tensors_archive_table,
    bool use_tensors_archive_table) {
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
  if (use_tensors_archive_table && !tensors_archive_table.empty()) {
    data_pickle.updateTensorsArchiveTable(tensors_archive_table);
  }
  data_pickle.protocol();
  data_pickle.pushIValue(value);
  data_pickle.stop();
  size_t i = 0;
  std::string prefix = archive_name + "/";

  // Export deduplicate tensors only if use_tensors_archive_table is set to
  // true and archive name is `bytecode`
  bool can_use_tensors_archive_table =
      (use_tensors_archive_table && archive_name == kArchiveNameBytecode);

  for (const auto& td : data_pickle.tensorData()) {
    WriteableTensorData writable_td = getWriteableTensorData(td);
    std::string fname = prefix + c10::to_string(i++);
    if (can_use_tensors_archive_table) {
      const auto found = tensors_archive_table.find(td);
      if (found == tensors_archive_table.end()) {
        writer.writeRecord(
            fname, writable_td.data(), writable_td.sizeInBytes());
      }
    } else {
      writer.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
    }
  }
  std::string fname = archive_name + ".pkl";
  writer.writeRecord(fname, data.data(), data.size());
}

} // namespace

// The family of methods below load a serialized Mobile Module
bool backport_v5_to_v4(
    PyTorchStreamReader& reader,
    PyTorchStreamWriter& writer,
    std::vector<c10::IValue>& bytecode_values) {
  // 1) read from archive `constants` and construct the TensorIndexMap from the
  // tensors in `constants`.

  // Read archive `constants`
  std::vector<IValue> ivalues_from_constants_archive =
      readArchive("constants", reader).toTuple()->elements();

  auto ivalues_tuple_from_constants_archive =
      c10::ivalue::Tuple::create(ivalues_from_constants_archive);

  // construct tensors_archive_table map
  // key: tensor, value: pair(archive_name, index)
  TensorIndexMap tensors_archive_table =
      get_tensors_archive_table(ivalues_tuple_from_constants_archive);

  //  auto records = reader.getAllRecords();

  // 2) Copy everything except bytecode related to new output
  copy_non_bytecode(reader, writer);

  // 3) write `bytecode` archive
  // Update the bytecode version in bytecode.pkl
  update_bytecode_version(bytecode_values, kBytecodeVersionV4);
  // Construct the list of ivalues to a big tuple
  auto bytecode_tuple = c10::ivalue::Tuple::create(std::move(bytecode_values));
  // write `bytecode` archive
  writeArchive(
      writer, "bytecode", bytecode_tuple, tensors_archive_table, false);
  return true;
}

} // namespace jit
} // namespace torch
