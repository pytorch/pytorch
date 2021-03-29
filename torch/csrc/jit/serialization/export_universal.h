#pragma once

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace jit {

struct TORCH_API ScriptModuleSerializerUniversal {
  explicit ScriptModuleSerializerUniversal(caffe2::serialize::PyTorchStreamWriter& export_writer, py::object exporter)
      : writer_(export_writer), package_exporter(exporter) {}

  void serialize(Module& module, const std::string& ts_id);
  void writeFiles();

 private:
  void writeArchive(
      const std::string& archive_name, 
      const IValue& value,
      const std::string& pickle_dir_ext);

  void convertTypes(const at::NamedTypePtr& root_type);
  void convertNamedType(const c10::NamedTypePtr& class_type);
  std::string getNextStorageID();
 
  caffe2::serialize::PyTorchStreamWriter& writer_;
  std::vector<at::IValue> constant_table_;
  std::unordered_set<c10::NamedTypePtr> converted_types_;
  PrintDepsTable class_deps_;
  TypeNameUniquer type_name_uniquer_;
  py::object package_exporter; 

  // qualifier, e.g. '__torch__.Bar' -> PythonPrint for the file that will be
  // created
  OrderedDict<std::string, PythonPrint> file_streams_;
};

} // namespace jit
} // namespace torch
