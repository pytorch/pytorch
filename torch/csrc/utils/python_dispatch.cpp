#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>

#include <pybind11/operators.h>

#include <iostream>

namespace py = pybind11;

namespace torch {
namespace impl {
namespace dispatch {

c10::optional<c10::DispatchKey> parseDispatchKey(const std::string& k) {
  static std::unordered_map<std::string, c10::DispatchKey> key_map = {
    {"cpu", c10::DispatchKey::CPUTensorId},
    {"cuda", c10::DispatchKey::CUDATensorId},
    {"xla", c10::DispatchKey::XLATensorId},
    {"autograd", c10::DispatchKey::VariableTensorId},
    {"", c10::DispatchKey::Undefined},
  };
  auto it = key_map.find(k);
  TORCH_CHECK(it != key_map.end(), "could not parse ", k);
  if (it->second == c10::DispatchKey::Undefined) {
    return c10::nullopt;
  } else {
    return c10::make_optional(it->second);
  }
}


template <typename Func>
inline c10::CppFunction dispatch_str(const char* key, Func&& raw_f) {
  auto mb_key = parseDispatchKey(key);
  if (mb_key) {
    return c10::dispatch(*mb_key, std::move(raw_f));
  } else {
    c10::CppFunction f(std::forward<Func>(raw_f));
    return f;
  }
}


void initDispatchBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // TODO: figure out how to do chaining
  py::class_<c10::Module>(m, "_DispatchModule")
    .def("def_", [](py::object self, const char* schema) {
      self.cast<c10::Module&>().def(schema);
      return self;
    })
    // We can't conveniently turn Python functions into valid functions
    // in the dispatcher.  So instead we provide a bunch of precanned
    // functions for testing purposes.  You're NOT intended to actually
    // call these functions; they're just here so we can actually register
    // something
    //
    // Mangling scheme: args_rets.  One character per.
    //  t = Tensor
    .def("def_t_t", [](py::object self, const char* name, const char* dispatch) {
      self.cast<c10::Module&>().def(name, dispatch_str(dispatch, [](const at::Tensor& a) {
        return a;
      }));
      return self;
    }, "", py::arg("name"), py::arg("dispatch") = "")
    .def("impl_t_t", [](py::object self, const char* name, const char* dispatch) {
      self.cast<c10::Module&>().impl(name, dispatch_str(dispatch, [](const at::Tensor& a) {
        return a;
      }));
      return self;
    }, "", py::arg("name"), py::arg("dispatch") = "")
    .def("impl_tt_t", [](py::object self, const char* name, const char* dispatch) {
      self.cast<c10::Module&>().impl(name, dispatch_str(dispatch, [](const at::Tensor& a, const at::Tensor& b) {
        return a;
      }));
      return self;
    }, "", py::arg("name"), py::arg("dispatch") = "")
  ;

  // NB: no support for namespace because cannot guarantee its lifetime
  m.def("_dispatch_import", []() {
    return std::make_unique<c10::Module>(torch::import());
  });

  m.def("_dispatch_dump", [](const char* name) -> std::string {
    auto op = c10::Dispatcher::singleton().findSchema(torch::jit::parseName(name));
    if (!op) {
      return "";
    } else {
      return op->dumpState();
    }
  });

  m.def("_dispatch_check_invariants", [](const char* name) {
    auto op = c10::Dispatcher::singleton().findSchema(torch::jit::parseName(name));
    if (!op) {
    } else {
      return op->checkInvariants();
    }
  });

  m.def("_dispatch_check_all_invariants", []() {
    c10::Dispatcher::singleton().checkInvariants();
  });
}

}}} // namespace torch::impl::dispatch
