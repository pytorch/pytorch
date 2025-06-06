#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/export/pybind.h>
#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::_export {

void initExportBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto exportModule = rootModule.def_submodule("_export");
  auto pt2ArchiveModule = exportModule.def_submodule("pt2_archive_constants");

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ExportedProgram>(exportModule, "CppExportedProgram");

  exportModule.def(
      "deserialize_exported_program", [](const std::string& serialized) {
        return nlohmann::json::parse(serialized).get<ExportedProgram>();
      });

  exportModule.def("serialize_exported_program", [](const ExportedProgram& ep) {
    return nlohmann::json(ep).dump();
  });

  for (const auto& entry : torch::_export::archive_spec::kAllConstants) {
    pt2ArchiveModule.attr(entry.first) = entry.second;
  }
}
} // namespace torch::_export
