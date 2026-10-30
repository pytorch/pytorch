#include <torch/csrc/export/example_upgraders.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/export/pybind.h>
#include <torch/csrc/export/upgrader.h>
#include <torch/csrc/utils/generated_serialization_bindings.h>
#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::_export {

void initExportBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto exportModule = rootModule.def_submodule("_export");
  auto pt2ArchiveModule = exportModule.def_submodule("pt2_archive_constants");

  registerSerializationBindings(exportModule);

  exportModule.def(
      "deserialize_exported_program", [](const std::string& serialized) {
        // Query the current Python schema version as target (cached)
        // TODO: expose schema_version in generated_serialization_types.h and
        // access it here directly.
        static int target_version = []() {
          py::module_ schema_module =
              py::module_::import("torch._export.serde.schema");
          py::tuple v = schema_module.attr("SCHEMA_VERSION");
          return v[0].cast<int>();
        }();

        ExportedProgram result;
        {
          py::gil_scoped_release release;
          auto parsed = nlohmann::json::parse(serialized);
          auto upgraded = upgrade(parsed, target_version);
          result = upgraded.get<ExportedProgram>();
        }
        return result;
      });

  exportModule.def("serialize_exported_program", [](const ExportedProgram& ep) {
    return nlohmann::json(ep).dump();
  });

  exportModule.def(
      "upgrade", [](const std::string& serialized_json, int target_version) {
        auto parsed = nlohmann::json::parse(serialized_json);
        auto upgraded = upgrade(parsed, target_version);
        return upgraded.dump();
      });

  exportModule.def(
      "register_example_upgraders", []() { registerExampleUpgraders(); });

  exportModule.def(
      "deregister_example_upgraders", []() { deregisterExampleUpgraders(); });

  exportModule.def(
      "deserialize_payload_config", [](const std::string& json_str) {
        PayloadConfig result;
        {
          py::gil_scoped_release release;
          auto parsed = nlohmann::json::parse(json_str);
          result = parsed.get<PayloadConfig>();
        }
        return result;
      });

  for (const auto& entry : torch::_export::archive_spec::kAllConstants) {
    pt2ArchiveModule.attr(entry.first) = entry.second;
  }
}
} // namespace torch::_export
