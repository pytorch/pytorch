#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/backport_manager.h>
#include <torch/csrc/jit/mobile/model_compatibility.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <cstddef>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using caffe2::serialize::ReadAdapterInterface;

// Utility function, can be reused by backport_vn_to_vn-1(). If any utility
// function can be reused by other backport function, move it here.
namespace {
bool update_bytecode_version(
    std::vector<at::IValue>& bytecode_values,
    const int64_t to_version) {
  if (!bytecode_values.empty() && bytecode_values[0].isInt()) {
    bytecode_values[0] = c10::IValue(to_version);
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

} // namespace

// To add next backport
// function, for example, backport_vn_to_vn-1, create an anonymous namespace
// with a backport_vn_to_vn-1 function + other necessary customized function. If
// a function can be reused by other backport functions, move it to the utility
// function group. It will be easier to split out backport_manager.cpp to
// smaller files when it grows too long.

// The functions needed for backport model from v5 to v4.
namespace {

void writeArchiveV4(
    PyTorchStreamWriter& writer,
    const std::string& archive_name,
    const c10::IValue& value) {
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
  writeArchiveV4(writer, "bytecode", bytecode_tuple);
  return true;
}
} // namespace

// A generic contract for backport logic to the previous bytecode version.
// Args:
// * PyTorchStreamReader has access to the input model from N bytecode version.
// * PyTorchStreamWriter has access to the output model backported to the
// previous N-1 bytecode version. Returns true if successful, false otherwise.
using BytecodeBackportFunction = std::function<bool(
    caffe2::serialize::PyTorchStreamReader&,
    caffe2::serialize::PyTorchStreamWriter&)>;

BackportManager::BackportManager() {
  registerBytecodeBackportFunction(kBytecodeVersionV5, backport_v5_to_v4);
}

std::unordered_map<
    int64_t,
    std::function<bool(
        caffe2::serialize::PyTorchStreamReader&,
        caffe2::serialize::PyTorchStreamWriter&)>>&
BackportManager::bytecodeBackportFunctions() const {
  static std::unordered_map<
      int64_t,
      std::function<bool(
          caffe2::serialize::PyTorchStreamReader&,
          caffe2::serialize::PyTorchStreamWriter&)>>
      backport_functions;
  return backport_functions;
}

bool BackportManager::hasBytecodeBackportFunction(
    const int64_t from_version) const {
  return bytecodeBackportFunctions().count(from_version);
}

void BackportManager::registerBytecodeBackportFunction(
    const int64_t from_version,
    const BytecodeBackportFunction& backport_function) {
  TORCH_CHECK(
      !hasBytecodeBackportFunction(from_version),
      "Backporting from version ",
      from_version,
      " is already registered.");
  bytecodeBackportFunctions()[from_version] = backport_function;
}

// The main function to run backport from version n to version i.
// All models (file or buffer) will be converted stream first, and
// istream_adapter has access to it. During the backport process,
// the intermediate result will be stored with stream.
bool BackportManager::backport(
    std::shared_ptr<IStreamAdapter> istream_adapter,
    PyTorchStreamWriter& final_writer,
    int64_t from_version,
    int64_t to_version) const {
  if (from_version <= to_version) {
    TORCH_WARN(
        "backport donesn't support backporting model to new version. It's trying to backport from version ",
        from_version,
        " to version ",
        to_version);
    return false;
  }
  int64_t bytecode_version = from_version;
  std::ostringstream out;
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };

  std::shared_ptr<IStreamAdapter> intermediate_istream_adapter =
      istream_adapter;
  std::ostringstream oss;
  bool backport_success = true;

  while (bytecode_version > to_version) {
    // Read from intermediate writer result if ostream is not empty, otherwise
    // it means that it's the first time to backport and read from the source.
    if (!out.str().empty()) {
      std::istringstream iss(out.str());
      intermediate_istream_adapter =
          std::make_shared<caffe2::serialize::IStreamAdapter>(&iss);
    }
    out.clear();

    PyTorchStreamReader intermediate_reader(intermediate_istream_adapter);
    PyTorchStreamWriter intermediate_writer(writer_func);

    if (!hasBytecodeBackportFunction(bytecode_version)) {
      return false;
    }

    // When it's the last backport process, write to the final destination
    // otherwise, export to the intermediate ostream.
    if (bytecode_version - 1 == to_version) {
      backport_success &= bytecodeBackportFunctions()[bytecode_version--](
          intermediate_reader, final_writer);
    } else {
      backport_success &= bytecodeBackportFunctions()[bytecode_version--](
          intermediate_reader, intermediate_writer);
    }
  }
  return backport_success;
}

} // namespace jit
} // namespace torch
