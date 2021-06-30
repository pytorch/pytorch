#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/backport_manager.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/model_compatibility.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <cstddef>
#include <sstream>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using caffe2::serialize::ReadAdapterInterface;

// Current support bytecode version
namespace {
constexpr int64_t kBytecodeVersionV4 = 0x4L;
constexpr int64_t kBytecodeVersionV5 = 0x5L;
constexpr int64_t kBytecodeVersionV6 = 0x6L;
} // namespace

// Utility function that can be reused by backport_vn_to_vn-1(). If any utility
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

// Copy files from source to destination except the files and dirs
void selective_copy(
    PyTorchStreamReader& reader,
    PyTorchStreamWriter& writer,
    const std::unordered_set<std::string>& excluded_files,
    const std::unordered_set<std::string>& excluded_dirs) {
  auto records = reader.getAllRecords();
  for (const auto& record : records) {
    // Don't copy archive in excluded_files, usually archive `version` and
    // `bytecode`. Archvie `version` will be written when PyTorchStreamWriter is
    // going to finalize and run writeEndOfFile()

    // records is the list of all files names in the zip file, and each record
    // is one file with path to parent folder, the example records is:
    // data.pkl
    // code/__torch__/___torch_mangle_5.py
    // code/__torch__/___torch_mangle_5.py.debug_pkl
    // constants/140245072983168.storage
    // constants.pkl
    // bytecode.pkl
    // version
    bool skip = excluded_files.count(record) > 0;

    // Skip dirs, find the last '/' and compare it with record
    for (const auto& excluded_dir : excluded_dirs) {
      std::size_t found = record.find_last_of("/\\");
      auto path = record.substr(0, found);
      if (excluded_dir == path) {
        skip = true;
        break;
      }
    }
    if (!skip) {
      auto data_ptr = reader.getRecord(record);
      auto data = std::get<0>(data_ptr).get();
      auto size = std::get<1>(data_ptr);
      writer.writeRecord(record, data, size);
    }
  }
}

// Copy all content from reader to stringstream
void get_model_stream(PyTorchStreamReader& reader, std::stringstream& out) {
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  PyTorchStreamWriter writer(writer_func);
  selective_copy(
      reader,
      writer,
      std::unordered_set<std::string>({"version"}),
      std::unordered_set<std::string>());
}

} // namespace

/*
 To add next backport function, for example, backport_vn_to_vn-1, create an
 anonymous namespace with a backport_vn_to_vn-1 function + other necessary
 customized function. If a function can be reused by other backport functions,
 move it to the utility function group. It will be easier to split out
 backport_manager.cpp to smaller files when it grows too long.

 How to add backport_v{i}_to_v{i-1} ?
 There are two options:
 1) [Format change only, recommended] Constrcut a reader with the
 input_model_stream, modify the file, and use PyTorchWriter to write it to
 output_model_stream. See backport_v5_to_v4.

 2) [Both format and content change] ]Use torch.jit.load() to load the stream,
 and save it to output_model_stream.

 The first option is preferred, because it will be purely format change, and
 the model doesn't need to go through inline again and model content will
 remain the same.

 A note for manipulate stringstream, it's recommend to declare a new
 stringstream, tmp_stream, and swap it with the argument output_model_stream
 once it's ready, output_model_stream.swap(tmp_stream). Do not use
 output_model_stream.clear(). It only clears out error state flag
 (https://www.cplusplus.com/reference/ios/ios/clear/), while the content is the
 same. It's cleaner to just declare a new one and swap.

*/

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

std::stringstream backport_v5_to_v4(std::stringstream& input_model_stream) {
  // 1) read from archive `bytecode` archive
  PyTorchStreamReader reader(&input_model_stream);
  std::vector<IValue> bytecode_values = get_bytecode_ivalues(reader);
  std::vector<IValue> constants_values =
      readArchive(kArchiveNameConstants, reader).toTuple()->elements();

  // 2) Copy everything to new output, except some specific files and dirs
  // (usually version, bytecode.pkl and bytecode folder are skipped)
  std::unordered_set<std::string> excluded_files{
      "constants.pkl",
      "bytecode.pkl",
      "version",
  };

  std::unordered_set<std::string> excluded_dirs{
      "constants",
      "bytecode",
  };

  std::stringstream ouput_model_stream;
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    ouput_model_stream.write(static_cast<const char*>(buf), nbytes);
    return !ouput_model_stream ? 0 : nbytes;
  };

  PyTorchStreamWriter writer(writer_func);

  selective_copy(reader, writer, excluded_files, excluded_dirs);

  // 3) write `bytecode` archive
  // Update the bytecode version in bytecode.pkl
  update_bytecode_version(bytecode_values, kBytecodeVersionV4);
  // Construct the list of ivalues to a big tuple
  auto bytecode_tuple = c10::ivalue::Tuple::create(std::move(bytecode_values));
  // write `bytecode` archive
  writeArchiveV4(writer, kArchiveNameBytecode, bytecode_tuple);
  // write `constants` archive
  auto constants_tuple =
      c10::ivalue::Tuple::create(std::move(constants_values));
  writeArchiveV4(writer, kArchiveNameConstants, constants_tuple);
  return ouput_model_stream;
}

void writeArchiveV5(
    PyTorchStreamWriter& writer,
    const IValue& value,
    const std::string& archive_name,
    const std::string& archive_dir,
    const std::string& tensor_dir,
    bool use_storage_context,
    SerializationStorageContext& storage_context) {
  std::vector<char> data;
  // Vector to capture the run-time class types during pickling the IValues
  std::vector<c10::ClassTypePtr> memoizedClassTypes;
  std::vector<std::string> tensor_names;
  Pickler data_pickle(
      [&](const char* buf, size_t size) {
        data.insert(data.end(), buf, buf + size);
      },
      nullptr,
      nullptr,
      &memoizedClassTypes,
      [&](const at::Tensor& tensor) {
        // returns a string to use in picker.cpp as storage obj key
        if (use_storage_context) {
          std::string string_id =
              std::to_string(reinterpret_cast<std::intptr_t>(
                  tensor.storage().unsafeGetStorageImpl()));
          tensor_names.push_back(string_id + ".storage");
          storage_context.getOrAddStorage(tensor.storage());
        } else {
          tensor_names.push_back(std::to_string(tensor_names.size()));
        }
        return tensor_names.back();
      });
  data_pickle.protocol();
  data_pickle.pushIValue(value);
  data_pickle.stop();
  // write out tensor data
  size_t i = 0;
  std::string prefix = archive_name + "/";

  TORCH_INTERNAL_ASSERT(tensor_names.size() == data_pickle.tensorData().size());
  const std::vector<std::string>& pre_serialized_files =
      writer.getAllWrittenRecords();

  for (const auto& td : data_pickle.tensorData()) {
    WriteableTensorData writable_td = getWriteableTensorData(td);
    std::string fname = tensor_dir + tensor_names[i++];
    if (use_storage_context &&
        std::find(
            pre_serialized_files.begin(), pre_serialized_files.end(), fname) !=
            pre_serialized_files.end()) {
      // storage has been serialzed already, skip
      continue;
    }
    writer.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
  }

  std::string fname = archive_dir + archive_name + ".pkl";
  writer.writeRecord(fname, data.data(), data.size());
}

std::stringstream backport_v6_to_v5(std::stringstream& input_model_stream) {
  std::shared_ptr<IStreamAdapter> rai =
      std::make_shared<IStreamAdapter>(&input_model_stream);
  auto reader = std::make_shared<PyTorchStreamReader>(rai);
  std::vector<IValue> constants_values =
      readArchive(kArchiveNameConstants, *reader.get()).toTuple()->elements();

  // If there are debug info files in the original model file, it should also
  // show up in the backported model
  bool hasBytecodeDebug = reader->hasRecord("mobile_debug_handles.pkl");

  // extra_files are kept
  auto records = reader->getAllRecords();
  ExtraFilesMap extra_files;
  for (const auto& record : records) {
    std::size_t found = record.find_last_of("/\\");
    auto path = record.substr(0, found);
    if ("extra" == path) {
      extra_files.emplace(record.substr(found + 1), "");
    }
  }
  // Loading the TS module is required for this backport, because bytecode needs
  // to be re-emitted (refer to the comments below)
  Module torch_script = torch::jit::load(rai, c10::nullopt, extra_files);

  // The RAII guard to change the flag, emitBytecodeDefaultInputs, to true, so
  // that TS stores the default argument values in the constant table, and emits
  // the instructions (LOADC, for example), to push the values to the stack. It
  // restores the behavior of V5 and before. For V6, the default arg values are
  // resolved at runtime init stage for better operator compatibility.
  std::stringstream intermediate_model_stream;
  {
    BytecodeEmitDefaultInputsGuard argNumGuard(true);
    torch_script._save_for_mobile(
        intermediate_model_stream, extra_files, hasBytecodeDebug);
  }

  // Update the bytecode version (from 6 to 5)

  PyTorchStreamReader reader_bytecode(&intermediate_model_stream);
  std::vector<IValue> bytecode_values = get_bytecode_ivalues(reader_bytecode);
  std::unordered_set<std::string> excluded_files{
      "constants.pkl",
      "bytecode.pkl",
      "version",
  };

  std::unordered_set<std::string> excluded_dirs{
      "constants",
      "bytecode",
  };

  std::stringstream ouput_model_stream;
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    ouput_model_stream.write(static_cast<const char*>(buf), nbytes);
    return !ouput_model_stream ? 0 : nbytes;
  };

  PyTorchStreamWriter writer_bytecode(writer_func);

  selective_copy(
      reader_bytecode, writer_bytecode, excluded_files, excluded_dirs);

  update_bytecode_version(bytecode_values, kBytecodeVersionV5);
  auto bytecode_tuple = c10::ivalue::Tuple::create(std::move(bytecode_values));
  SerializationStorageContext storage_context;
  writeArchiveV5(
      writer_bytecode,
      c10::ivalue::Tuple::create(constants_values),
      /*archive_name=*/"constants",
      /*archive_dir=*/"",
      /*tensor_dir=*/"constants/",
      /*use_storage_context=*/true,
      storage_context);
  writeArchiveV5(
      writer_bytecode,
      bytecode_tuple,
      /*archive_name=*/"bytecode",
      /*archive_dir=*/"",
      /*tensor_dir=*/"constants/",
      /*use_storage_context=*/true,
      storage_context);

  return ouput_model_stream;
}
} // namespace

// A generic contract for backport logic to the previous bytecode version.
// Args:
// * PyTorchStreamReader has access to the input model from N bytecode version.
// * PyTorchStreamWriter has access to the output model backported to the
// previous N-1 bytecode version. Returns true if successful, false otherwise.
using BytecodeBackportFunction =
    std::function<std::stringstream(std::stringstream&)>;

BackportManager::BackportManager() {
  registerBytecodeBackportFunction(kBytecodeVersionV5, backport_v5_to_v4);
  registerBytecodeBackportFunction(kBytecodeVersionV6, backport_v6_to_v5);
}

std::unordered_map<
    int64_t,
    std::function<std::stringstream(std::stringstream&)>>&
BackportManager::bytecodeBackportFunctions() const {
  static std::unordered_map<
      int64_t,
      std::function<std::stringstream(std::stringstream&)>>
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
  PyTorchStreamReader start_reader(istream_adapter);

  if (from_version <= to_version) {
    TORCH_WARN(
        "backport donesn't support backporting model to new version. It's trying to backport from version ",
        from_version,
        " to version ",
        to_version);
    return false;
  }
  int64_t bytecode_version = from_version;
  bool backport_success = true;

  // 1) Given an istream_adapter (an adapter with access to the input model, the
  // model can be from istream, file and etc), copy all model content to
  // stringstream
  std::stringstream oss;
  get_model_stream(start_reader, oss);
  std::stringstream input_model_stream(oss.str());
  std::stringstream output_model_stream;

  // 2) backport model, backport_v{i}_to_v{i-1} function's argurment is
  // (input_model_stream and output_model_stream)
  while (bytecode_version > to_version) {
    // Swap input and output if it's not the first time and output_model_stream
    // has value.
    if (!output_model_stream.str().empty()) {
      input_model_stream.swap(output_model_stream);
      // reset output_model_stream
      output_model_stream.str("");
    }

    if (!hasBytecodeBackportFunction(bytecode_version)) {
      return false;
    }

    auto input_model_stream_version =
        _get_model_bytecode_version(input_model_stream);

    if (input_model_stream_version != bytecode_version) {
      TORCH_WARN(
          "The bytecode version of input model stream is supposed to be ",
          bytecode_version,
          ", but it gets ",
          input_model_stream_version);
      return false;
    }

    // Keep backporting till request version
    std::stringstream backport_model_stream =
        bytecodeBackportFunctions()[bytecode_version--](input_model_stream);

    output_model_stream.swap(backport_model_stream);
    auto output_model_stream_version =
        _get_model_bytecode_version(output_model_stream);

    if (output_model_stream_version != bytecode_version) {
      TORCH_WARN(
          "The bytecode version of output model stream is supposed to be ",
          bytecode_version,
          ", but it gets ",
          output_model_stream_version);
      return false;
    }
  }

  // 3) Write the final output_model_stream to final_writer, final_writer has
  // access to the final model destination (file, ostream and etc)
  if (output_model_stream.str().empty()) {
    TORCH_WARN("No output model from backport.");
    return false;
  }
  PyTorchStreamReader last_model_reader(&output_model_stream);
  selective_copy(
      last_model_reader,
      final_writer,
      std::unordered_set<std::string>({"version"}),
      std::unordered_set<std::string>());

  return backport_success;
}

} // namespace jit
} // namespace torch
