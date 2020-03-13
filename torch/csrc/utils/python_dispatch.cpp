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



void initDispatchBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // TODO: figure out how to do chaining
  py::class_<c10::Module>(m, "_DispatchModule")
    .def("def_", [](c10::Module& m, const char* schema) {
      m.def(schema);
    })
    // We can't conveniently turn Python functions into valid functions
    // in the dispatcher.  So instead we provide a bunch of precanned
    // functions for testing purposes.  You're NOT intended to actually
    // call these functions; they're just here so we can actually register
    // something
    //
    // Mangling scheme: args_rets.  One character per.
    //  t = Tensor
    .def("impl_t_t", [](c10::Module& m, const char* name) {
      m.impl(name, [](const at::Tensor& a) {
        return a;
      });
    })
    .def("impl_tt_t", [](c10::Module& m, const char* name) {
      m.impl(name, [](const at::Tensor& a, const at::Tensor& b) {
        return a;
      });
    })
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
}

}}} // namespace torch::impl::dispatch
