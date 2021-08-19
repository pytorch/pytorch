#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

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
    {"", c10::DispatchKey::Undefined},
    {"CPU", c10::DispatchKey::CPU},
    {"CUDA", c10::DispatchKey::CUDA},
    {"HIP", c10::DispatchKey::HIP},
    {"FPGA", c10::DispatchKey::FPGA},
    {"MSNPU", c10::DispatchKey::MSNPU},
    {"XLA", c10::DispatchKey::XLA},
    {"MLC", c10::DispatchKey::MLC},
    {"Vulkan", c10::DispatchKey::Vulkan},
    {"Metal", c10::DispatchKey::Metal},
    {"XPU", c10::DispatchKey::XPU},
    {"HPU", c10::DispatchKey::HPU},
    {"VE", c10::DispatchKey::VE},
    {"Lazy", c10::DispatchKey::Lazy},
    {"Meta", c10::DispatchKey::Meta},
    {"QuantizedCPU", c10::DispatchKey::QuantizedCPU},
    {"QuantizedCUDA", c10::DispatchKey::QuantizedCUDA},
    {"QuantizedXPU", c10::DispatchKey::QuantizedXPU},
    {"CustomRNGKeyId", c10::DispatchKey::CustomRNGKeyId},
    {"MkldnnCPU", c10::DispatchKey::MkldnnCPU},
    {"SparseCPU", c10::DispatchKey::SparseCPU},
    {"SparseCUDA", c10::DispatchKey::SparseCUDA},
    {"SparseHIP", c10::DispatchKey::SparseHIP},
    {"SparseXPU", c10::DispatchKey::SparseXPU},
    {"SparseVE", c10::DispatchKey::SparseVE},
    {"SparseCsrCPU", c10::DispatchKey::SparseCsrCPU},
    {"SparseCsrCUDA", c10::DispatchKey::SparseCsrCUDA},
    {"NestedTensor", c10::DispatchKey::NestedTensor},
    {"PrivateUse1", c10::DispatchKey::PrivateUse1},
    {"PrivateUse2", c10::DispatchKey::PrivateUse2},
    {"PrivateUse3", c10::DispatchKey::PrivateUse3},
    {"BackendSelect", c10::DispatchKey::BackendSelect},
    {"Python", c10::DispatchKey::Python},
    {"FuncTorchPython", c10::DispatchKey::FuncTorchPython},
    {"Named", c10::DispatchKey::Named},
    {"Conjugate", c10::DispatchKey::Conjugate},
    {"Negative", c10::DispatchKey::Negative},
    {"FuncTorchDynamicLayerBackMode", c10::DispatchKey::FuncTorchDynamicLayerBackMode},
    {"ADInplaceOrView", c10::DispatchKey::ADInplaceOrView},
    {"AutogradOther", c10::DispatchKey::AutogradOther},
    {"AutogradCPU", c10::DispatchKey::AutogradCPU},
    {"AutogradCUDA", c10::DispatchKey::AutogradCUDA},
    {"AutogradXLA", c10::DispatchKey::AutogradXLA},
    {"AutogradLazy", c10::DispatchKey::AutogradLazy},
    {"AutogradXPU", c10::DispatchKey::AutogradXPU},
    {"AutogradMLC", c10::DispatchKey::AutogradMLC},
    {"AutogradHPU", c10::DispatchKey::AutogradHPU},
    {"AutogradNestedTensor", c10::DispatchKey::AutogradNestedTensor},
    {"AutogradPrivateUse1", c10::DispatchKey::AutogradPrivateUse1},
    {"AutogradPrivateUse2", c10::DispatchKey::AutogradPrivateUse2},
    {"AutogradPrivateUse3", c10::DispatchKey::AutogradPrivateUse3},
    {"Tracer", c10::DispatchKey::Tracer},
    {"AutocastCPU", c10::DispatchKey::AutocastCPU},
    {"AutocastCUDA", c10::DispatchKey::AutocastCUDA},
    {"FuncTorchBatched", c10::DispatchKey::FuncTorchBatched},
    {"FuncTorchVmapMode", c10::DispatchKey::FuncTorchVmapMode},
    {"Batched", c10::DispatchKey::Batched},
    {"VmapMode", c10::DispatchKey::VmapMode},
    {"FuncTorchGradWrapper", c10::DispatchKey::FuncTorchGradWrapper},
    {"FuncTorchDynamicLayerFrontMode", c10::DispatchKey::FuncTorchDynamicLayerFrontMode},
    {"TESTING_ONLY_GenericWrapper", c10::DispatchKey::TESTING_ONLY_GenericWrapper},
    {"TESTING_ONLY_GenericMode", c10::DispatchKey::TESTING_ONLY_GenericMode},
    {"Autograd", c10::DispatchKey::Autograd},
    {"CompositeImplicitAutograd", c10::DispatchKey::CompositeImplicitAutograd},
    {"CompositeExplicitAutograd", c10::DispatchKey::CompositeExplicitAutograd},
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

  m.def("_dispatch_find_dangling_impls", []() -> std::vector<std::string> {
    auto danglingImpls =  c10::Dispatcher::singleton().findDanglingImpls();

    std::vector<std::string> states;
    states.reserve(danglingImpls.size());
    for (auto& danglingImpl : danglingImpls) {
      states.push_back(danglingImpl.dumpState());
    }

    return states;
  });

  m.def("_dispatch_print_registrations_for_dispatch_key", [](const char* dispatch_key) {
    auto k = parseDispatchKey(dispatch_key);
    if (!k) {
        std::cout << "Invalid DispatchKey: " << dispatch_key << std::endl;
    } else {
        c10::Dispatcher::singleton().printRegistrationsForDispatchKey(*k);
    }
  });
}

}}} // namespace torch::impl::dispatch
