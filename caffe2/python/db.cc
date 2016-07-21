#include <pybind11/pybind11.h>

#include "caffe2/core/db.h"

namespace caffe2 {
namespace db {

namespace py = pybind11;

PYBIND11_PLUGIN(caffe2_pybind11) {
  py::module db("caffe2_pybind11", "module for working with DB interface");
  py::class_<Transaction>(db, "Transaction")
      .def("put", &Transaction::Put)
      .def("commit", &Transaction::Commit);
  py::class_<Cursor>(db, "Cursor")
      .def("supports_seak", &Cursor::SupportsSeek)
      .def("seek_to_first", &Cursor::SeekToFirst)
      .def("next", &Cursor::Next)
      .def("key", &Cursor::key)
      .def("value", &Cursor::value)
      .def("valid", &Cursor::Valid);
  py::enum_<Mode>(db, "Mode")
      .value("read", Mode::READ)
      .value("write", Mode::WRITE)
      .value("new", Mode::NEW)
      .export_values();
  py::class_<DB /*, std::unique_ptr<DB>*/>(db, "DB")
      .def("new_transaction", &DB::NewTransaction)
      .def("new_cursor", &DB::NewCursor)
      .def("close", &DB::Close);
  db.def("create_db", &CreateDB);

  return db.ptr();
}

} // db
} // caffe2
