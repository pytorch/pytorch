#pragma once

#include <string>
#include <utility>

#include <c10/macros/Macros.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/custom_class.h>

namespace torch::jit {
struct OpaqueObject : public CustomClassHolder {
  OpaqueObject(py::object payload) : payload_(std::move(payload)) {}

  void setPayload(py::object payload) {
    payload_ = std::move(payload);
  }

  py::object getPayload() {
    return payload_;
  }

  py::object payload_;
};

static auto register_opaque_obj_class =
    torch::class_<OpaqueObject>("aten", "OpaqueObject")
        .def(
            "__eq__",
            [](const c10::intrusive_ptr<OpaqueObject>& self,
               const c10::intrusive_ptr<OpaqueObject>& other) {
              auto self_payload = self->getPayload();
              auto other_payload = other->getPayload();

              if (!self_payload.ptr() || !other_payload.ptr()) {
                return false;
              }

              py::gil_scoped_acquire gil;
              auto res = PyObject_RichCompareBool(
                  self_payload.ptr(), other_payload.ptr(), Py_EQ);
              if (res == -1) {
                throw py::error_already_set();
              }
              return res > 0;
            })
        .def_pickle(
            [](const c10::intrusive_ptr<OpaqueObject>& self) { // __getstate__
              // Since we cannot directly return the py::object due to
              // CustomClassHolder's signature limitations, we will have to
              // serialize it directly here. We also can't return py::bytes so
              // need to encode it into a string.
              py::module_ pickle = py::module_::import("pickle");
              py::module_ base64 = py::module_::import("base64");
              py::bytes pickled_payload =
                  pickle.attr("dumps")(self->getPayload());
              py::bytes encoded_payload =
                  base64.attr("b64encode")(pickled_payload);
              return std::string(encoded_payload);
            },
            [](const std::string& state) { // __setstate__
              py::module_ pickle = py::module_::import("pickle");
              py::module_ base64 = py::module_::import("base64");
              py::bytes state_bytes(state);
              py::bytes decoded_payload = base64.attr("b64decode")(state_bytes);
              py::object restored_payload =
                  pickle.attr("loads")(decoded_payload);
              return c10::make_intrusive<OpaqueObject>(restored_payload);
            })
        .def(
            "__obj_flatten__",
            [](const c10::intrusive_ptr<OpaqueObject>& self) {
              throw std::runtime_error(
                  "Unable to implement __obj_flatten__ for opaque objects.");
            });

} // namespace torch::jit
