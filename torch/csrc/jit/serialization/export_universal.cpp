#include <torch/csrc/jit/serialization/export_universal.h>

#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h> 

namespace torch {
namespace jit {

void ScriptModuleSerializerUniversal::serialize(
    Module &module, 
    const std::string &ts_id) {
  // Serialize the model object
  writeArchive("data", module._ivalue(), ts_id);
  // Then we serialize all code info.
  convertTypes(module.type());
  // The tensor constants from the code are written to a separate archive
  // so loading the code does not depend on loading the data
  std::vector<IValue> ivalue_constants(constant_table_.begin(), constant_table_.end());
  writeArchive("constants", c10::ivalue::Tuple::create(ivalue_constants), ts_id);
}

void ScriptModuleSerializerUniversal::writeArchive(
    const std::string &archive_name,
    const IValue &value,
    const std::string &ts_id) {
  std::vector<char> data;
  // Vector to capture the run-time class types during pickling the IValues
  std::vector<c10::ClassTypePtr> memoizedClassTypes;
  // from the exporter
  std::vector<std::string> tensor_names;
  Pickler data_pickle(
      [&](const char *buf, size_t size) {
        data.insert(data.end(), buf, buf + size);
      },
      nullptr,
      [&](const c10::ClassTypePtr &t) {
        return type_name_uniquer_.getUniqueName(t);
      },
      &memoizedClassTypes,
      [&]() {
        // returns a string to use in picker.cpp as storage obj key
        tensor_names.push_back(getNextStorageID() + ".storage");
        return tensor_names.back();
      });
  data_pickle.protocol();
  data_pickle.pushIValue(value);
  data_pickle.stop();

  size_t i = 0;
  assert(tensor_names.size() == data_pickle.tensorData().size());
  for (const auto &td : data_pickle.tensorData()) {
    WriteableTensorData writable_td = getWriteableTensorData(td);
    std::string fname = ".data/" + tensor_names[i++];
    writer_.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
  }

  std::string fname = ".data/ts_code/" + ts_id + "/" + archive_name + ".pkl";
  writer_.writeRecord(fname, data.data(), data.size());

  // serialize all the captured run-time class types
  for (const c10::ClassTypePtr &wroteType : memoizedClassTypes) {
    convertNamedType(wroteType);
  }
}

void ScriptModuleSerializerUniversal::convertTypes(
    const at::NamedTypePtr &root_type) {
  class_deps_.add(root_type);
  for (size_t i = 0; i < class_deps_.size(); ++i) {
    // note: convertNameType may extend class_deps_, so re-checking .size() is necessary
    convertNamedType(class_deps_[i]);
  }
}

void ScriptModuleSerializerUniversal::writeFiles() {
  // Mapping of filename => src. We need this because multiple classes may go
  // in the same file (e.g. foo.bar.Baz and foo.bar.Qux)
  for (auto &item : file_streams_) {
    const std::string filename =
        qualifierToArchivePath(item.key(), ".data/ts_code/code/");

    std::string src = item.value().str();    
    // Only compress these records if they're not tiny.
    // The cpu cost of generating zip datastructs and compressing isn't
    // well-spent for very small records.
    static constexpr size_t kMinToCompress = 200;

    writer_.writeRecord(filename, src.c_str(), src.size(),
                        src.size() > kMinToCompress /*compress*/);

    // Write out the debug information
    std::string debugFilename = filename + ".debug_pkl";
    SourceRangePickler source_range_pickler;
    auto range_data = source_range_pickler.pickle(item.value().ranges());
    writer_.writeRecord(debugFilename, range_data.data(), range_data.size(),
                        range_data.size() > kMinToCompress /*compress*/);
  }
}

void ScriptModuleSerializerUniversal::convertNamedType(
    const c10::NamedTypePtr &class_type) {
  if (converted_types_.count(class_type)) {
    return;
  }
  converted_types_.insert(class_type);
  auto qualname = type_name_uniquer_.getUniqueName(class_type);
  std::string qualifier = qualname.prefix();
  PythonPrint *pp = file_streams_.find(qualifier);

  auto type_printer = [&](
      const c10::ConstTypePtr &t) -> c10::optional<std::string> {
    auto namedType = t->cast<c10::NamedType>();
    if (namedType && namedType->name()) {
      return type_name_uniquer_.getUniqueName(namedType).qualifiedName();
    }
    return c10::nullopt;
  };
  if (!pp) {
    pp = &file_streams_.insert(
        std::move(qualifier),
        PythonPrint(constant_table_, class_deps_, type_printer,
                    /*enforce_importable=*/true));
  }
  pp->printNamedType(class_type);
}

std::string ScriptModuleSerializerUniversal::getNextStorageID() {
  // call exporter's get_storage_id function
  py::object result = package_exporter.attr("get_storage_id")();
  return std::string(py::str(result));
}

}  // namespace jit
}  // namespace torch
