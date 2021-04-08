#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <pybind11/operators.h>

#include <iostream>

namespace py = pybind11;

namespace torch {
namespace impl {
namespace dispatch {

torch::Library::Kind parseKind(const std::string& k) {
  static std::unordered_map<std::string, torch::Library::Kind> kind_map = {
    {"DEF", torch::Library::DEF},
    {"IMPL", torch::Library::IMPL},
    {"FRAGMENT", torch::Library::FRAGMENT},
  };
  auto it = kind_map.find(k);
  TORCH_CHECK(it != kind_map.end(), "could not parse ", k);
  return it->second;
}

c10::optional<c10::DispatchKey> parseDispatchKey(const std::string& k) {
  static std::unordered_map<std::string, c10::DispatchKey> key_map = {
    {"CPU", c10::DispatchKey::CPU},
    {"CUDA", c10::DispatchKey::CUDA},
    {"XLA", c10::DispatchKey::XLA},
    {"QuantizedCPU", c10::DispatchKey::QuantizedCPU},
    {"CompositeImplicitAutograd", c10::DispatchKey::CompositeImplicitAutograd},
    {"Autograd", c10::DispatchKey::Autograd},
    {"CompositeExplicitAutograd", c10::DispatchKey::CompositeExplicitAutograd},
    {"AutogradCPU", c10::DispatchKey::AutogradCPU},
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

c10::AliasAnalysisKind parseAliasAnalysisKind(const std::string& k) {
  static std::unordered_map<std::string, c10::AliasAnalysisKind> key_map = {
    {"CONSERVATIVE", c10::AliasAnalysisKind::CONSERVATIVE},
    {"FROM_SCHEMA", c10::AliasAnalysisKind::FROM_SCHEMA},
    {"PURE_FUNCTION", c10::AliasAnalysisKind::PURE_FUNCTION},
    {"", c10::AliasAnalysisKind::FROM_SCHEMA},  // default
  };
  auto it = key_map.find(k);
  TORCH_CHECK(it != key_map.end(), "could not parse ", k);
  return it->second;
}


template <typename Func>
inline torch::CppFunction dispatch_str(const char* key, Func&& raw_f) {
  auto mb_key = parseDispatchKey(key);
  if (mb_key) {
    return torch::dispatch(*mb_key, std::forward<Func>(raw_f));
  } else {
    torch::CppFunction f(std::forward<Func>(raw_f));
    return f;
  }
}

void initDispatchBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<c10::OperatorHandle>(m, "_DispatchOperatorHandle")
    .def("schema", &c10::OperatorHandle::schema);

  // TODO: figure out how to do chaining
  py::class_<torch::Library>(m, "_DispatchModule")
    .def("def_", [](py::object self, const char* schema, const char* alias) {
      self.cast<torch::Library&>().def(torch::schema(schema, parseAliasAnalysisKind(alias)));
      return self;
    }, "", py::arg("schema"), py::arg("alias") = "")
    // Simulated "legacy" def where alias analysis kind is not set.
    // Ordinarily this can only be exercised from RegisterOperators() API
    // but I am not going to bind that here
    .def("def_legacy", [](py::object self, const char* schema) {
      self.cast<torch::Library&>().def(torch::jit::parseSchema(schema));
      return self;
    }, "", py::arg("schema"))
    // We can't conveniently turn Python functions into valid functions
    // in the dispatcher.  So instead we provide a bunch of precanned
    // functions for testing purposes.  You're NOT intended to actually
    // call these functions; they're just here so we can actually register
    // something
    //
    // Mangling scheme: args_rets.  One character per.
    //  t = Tensor
    .def("def_name_t_t", [](py::object self, const char* name, const char* dispatch, const char* debug) {
      self.cast<torch::Library&>().def(
        name,
        dispatch_str(dispatch, [](const at::Tensor& a) {
          return a;
        }).debug(debug)
      );
      return self;
    }, "", py::arg("name"),
           py::arg("dispatch") = "",
           py::arg("debug") = "default_def_name_t_t")
    .def("def_schema_t_t", [](py::object self, const char* schema, const char* dispatch, const char* alias, const char* debug) {
      self.cast<torch::Library&>().def(
        torch::schema(schema, parseAliasAnalysisKind(alias)),
        dispatch_str(dispatch, [](const at::Tensor& a) {
          return a;
        }).debug(debug)
      );
      return self;
    }, "", py::arg("name"),
           py::arg("dispatch") = "",
           py::arg("alias") = "",
           py::arg("debug") = "default_def_schema_t_t")
    // TODO: maybe consider deduplicating the definitions here, it's getting
    // pretty long
    .def("impl_t_t", [](py::object self, const char* name, const char* dispatch, const char* debug) {
      self.cast<torch::Library&>().impl(
        name,
        dispatch_str(dispatch, [](const at::Tensor& a) {
          return a;
        }).debug(debug)
      );
      return self;
    }, "", py::arg("name"),
           py::arg("dispatch") = "",
           py::arg("debug") = "impl_t_t")
    .def("impl_tt_t", [](py::object self, const char* name, const char* dispatch, const char* debug) {
      self.cast<torch::Library&>().impl(
        name,
        dispatch_str(dispatch, [](const at::Tensor& a, const at::Tensor& b) {
          return a;
        }).debug(debug)
      );
      return self;
    }, "", py::arg("name"), py::arg("dispatch") = "", py::arg("debug") = "")
    .def("fallback_fallthrough", [](py::object self, const char* dispatch) {
      self.cast<torch::Library&>().fallback(
        dispatch_str(dispatch, CppFunction::makeFallthrough())
      );
      return self;
    }, "", py::arg("dispatch") = "")
  ;

  m.def("_dispatch_library", [](const char* kind, std::string name, const char* dispatch) {
    return std::make_unique<torch::Library>(parseKind(kind), std::move(name), parseDispatchKey(dispatch), "/dev/null", 0);
  });

  m.def("_dispatch_dump", [](const char* name) -> std::string {
    auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
    if (!op) {
      return "";
    } else {
      return op->dumpState();
    }
  });

  m.def("_dispatch_dump_table", [](const char* name) -> std::string {
    auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
    if (!op) {
      return "";
    } else {
      return op->dumpComputedTable();
    }
  });

  m.def("_dispatch_check_invariants", [](const char* name) {
    auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
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
