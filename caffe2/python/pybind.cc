#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/core/db.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/operator_schema.h"

namespace caffe2 {

namespace py = pybind11;
using namespace db;

template <typename Registry>
std::function<const char*(const string&)> DefinitionGetter(
    const Registry* registry) {
  return [registry](const string& name) { return registry->HelpMessage(name); };
}

PYBIND11_PLUGIN(caffe2_pybind11) {
  py::module m("caffe2_pybind11", "module for working with DB interface");
  py::class_<Transaction>(m, "Transaction")
      .def("put", &Transaction::Put)
      .def("commit", &Transaction::Commit);
  py::class_<Cursor>(m, "Cursor")
      .def("supports_seak", &Cursor::SupportsSeek)
      .def("seek_to_first", &Cursor::SeekToFirst)
      .def("next", &Cursor::Next)
      .def("key", &Cursor::key)
      .def("value", &Cursor::value)
      .def("valid", &Cursor::Valid);
  py::enum_<Mode>(m, "Mode")
      .value("read", Mode::READ)
      .value("write", Mode::WRITE)
      .value("new", Mode::NEW)
      .export_values();
  py::class_<DB /*, std::unique_ptr<DB>*/>(m, "DB")
      .def("new_transaction", &DB::NewTransaction)
      .def("new_cursor", &DB::NewCursor)
      .def("close", &DB::Close);
  m.def("create_db", &CreateDB);

  py::class_<OpSchema>(m, "OpSchema")
      .def_property_readonly("file", &OpSchema::file)
      .def_property_readonly("line", &OpSchema::line)
      .def_property_readonly(
          "doc", &OpSchema::doc, py::return_value_policy::reference)
      .def_property_readonly("arg_desc", &OpSchema::arg_desc)
      .def_property_readonly("input_desc", &OpSchema::input_desc)
      .def_property_readonly("output_desc", &OpSchema::output_desc)
      .def_static(
          "get", &OpSchemaRegistry::Schema, py::return_value_policy::reference)
      .def_static(
          "get_cpu_impl",
          DefinitionGetter(CPUOperatorRegistry()),
          py::return_value_policy::reference)
      .def_static(
          "get_cuda_impl",
          DefinitionGetter(CUDAOperatorRegistry()),
          py::return_value_policy::reference)
      .def_static(
          "get_gradient_impl",
          DefinitionGetter(GradientRegistry()),
          py::return_value_policy::reference);

  return m.ptr();
}

} // caffe2
