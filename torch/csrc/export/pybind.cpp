#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::_export {

void initExportBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = rootModule.def_submodule("_export");

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ExportedProgram>(m, "CppExportedProgram");

  m.def("deserialize_exported_program", [](const std::string& serialized) {
    return nlohmann::json::parse(serialized).get<ExportedProgram>();
  });

  m.def("serialize_exported_program", [](const ExportedProgram& ep) {
    return nlohmann::json(ep).dump();
  });
}
} // namespace torch::_export
