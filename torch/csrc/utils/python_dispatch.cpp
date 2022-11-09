#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/utils/python_dispatch.h>

#include <ATen/ATen.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include <c10/core/SafePyObject.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <c10/util/flat_hash_map.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <iostream>

namespace py = pybind11;

namespace torch {
namespace impl {
namespace dispatch {

// NB: I'd like to index this on OperatorHandle, but I can't, as I can't
// guarantee that the main interpreter has finish doing all registrations before
// the other interpreters start banging on it
static ska::flat_hash_map<
    c10::OperatorName,
    ska::flat_hash_map<c10::DispatchKey, std::shared_ptr<c10::SafePyObject>>>
    python_registrations_;

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
c10::AliasAnalysisKind parseAliasAnalysisKind(const std::string& k) {
  static std::unordered_map<std::string, c10::AliasAnalysisKind> key_map = {
      {"CONSERVATIVE", c10::AliasAnalysisKind::CONSERVATIVE},
      {"FROM_SCHEMA", c10::AliasAnalysisKind::FROM_SCHEMA},
      {"PURE_FUNCTION", c10::AliasAnalysisKind::PURE_FUNCTION},
      {"", c10::AliasAnalysisKind::FROM_SCHEMA}, // default
  };
  auto it = key_map.find(k);
  TORCH_CHECK(it != key_map.end(), "could not parse ", k);
  return it->second;
}

template <typename Func>
inline torch::CppFunction dispatch_str(const char* key, Func&& raw_f) {
  auto mb_key = std::string(key) == ""
      ? c10::nullopt
      : c10::make_optional(c10::parseDispatchKey(key));
  if (mb_key) {
    return torch::dispatch(*mb_key, std::forward<Func>(raw_f));
  } else {
    torch::CppFunction f(std::forward<Func>(raw_f));
    return f;
  }
}

struct EnableHermeticPyObject {
  EnableHermeticPyObject()
      : old_(c10::impl::HermeticPyObjectTLS::get_state()),
        old_excluded_python_(
            c10::impl::tls_is_dispatch_key_excluded(at::DispatchKey::Python)),
        old_python_(
            c10::impl::tls_is_dispatch_key_included(at::DispatchKey::Python)),
        old_python_snapshot_(c10::impl::tls_is_dispatch_key_included(
            at::DispatchKey::PythonTLSSnapshot)) {
    c10::impl::HermeticPyObjectTLS::set_state(true);
    c10::impl::tls_set_dispatch_key_excluded(at::DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(at::DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        at::DispatchKey::PythonTLSSnapshot, false);
  }
  ~EnableHermeticPyObject() {
    c10::impl::HermeticPyObjectTLS::set_state(old_);
    c10::impl::tls_set_dispatch_key_excluded(
        at::DispatchKey::Python, old_excluded_python_);
    c10::impl::tls_set_dispatch_key_included(
        at::DispatchKey::Python, old_python_);
    c10::impl::tls_set_dispatch_key_included(
        at::DispatchKey::PythonTLSSnapshot, old_python_snapshot_);
  }
  bool old_;
  bool old_excluded_python_;
  bool old_python_;
  bool old_python_snapshot_;
};

class PythonKernelHolder : public c10::OperatorKernel {
  c10::SafePyObject func_;
  c10::DispatchKey dispatch_key_;

 public:
  PythonKernelHolder(py::object func, c10::DispatchKey dispatch_key)
      : func_(func.release().ptr(), getPyInterpreter()),
        dispatch_key_(dispatch_key) {}

  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack) {
    // Figure out if we can handle it hermetically, or if we have
    // to double dispatch

    // If Torch Dispatch Mode is active, use its PyInterpreter for dispatch
    const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
    if (mode_stack_len > 0) {
      const auto& cur_torch_dispatch_mode_state =
          c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
      cur_torch_dispatch_mode_state->pyinterpreter()
          ->python_op_registration_trampoline(op, dispatch_key_, stack);
      return;
    }

    const auto& schema = op.schema();
    const auto num_arguments = schema.arguments().size();

    // Otherwise, find a PyInterpreter on a Tensor IF if has Python key (which
    // means it's a nontrivial tensor subclass)
    for (const auto& ivalue : torch::jit::last(*stack, num_arguments)) {
      if (ivalue.isTensor()) {
        auto* interpreter = ivalue.unsafeToTensorImpl()->pyobj_interpreter();
        if (interpreter &&
            ivalue.unsafeToTensorImpl()->key_set().has(
                at::DispatchKey::Python)) {
          (*interpreter)
              ->python_op_registration_trampoline(op, dispatch_key_, stack);
          return;
        }
      } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
        // NB: use toListRef as it doesn't induce refcount bumps
        // (toTensorListRef is not a thing)
        for (const auto& nv : ivalue.toListRef()) {
          if (nv.isNone()) {
            continue;
          }
          auto* interpreter = nv.unsafeToTensorImpl()->pyobj_interpreter();
          if (interpreter &&
              nv.unsafeToTensorImpl()->key_set().has(at::DispatchKey::Python)) {
            (*interpreter)
                ->python_op_registration_trampoline(op, dispatch_key_, stack);
            return;
          }
        }
      }
    }

    // Nothing requires the operator to be homed to a specific interpreter, so
    // run it on the current interpreter

    auto arguments = torch::jit::pop(*stack, op.schema().arguments().size());
    py::gil_scoped_acquire g;
    EnableHermeticPyObject g2;
    auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
    auto obj = py::reinterpret_steal<py::object>(PyObject_Call(
        func_.ptr(getPyInterpreter()),
        args_kwargs.first.ptr(),
        args_kwargs.second.ptr()));
    if (!obj) {
      throw python_error();
    }
    pushPyOutToStack(op, stack, obj, "PythonKernelHolder");
  }
};

torch::_RegisterOrVerify register_or_verify() {
  if (isMainPyInterpreter()) {
    return torch::_RegisterOrVerify::REGISTER;
  } else {
    return torch::_RegisterOrVerify::VERIFY;
  }
}

void initDispatchBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<c10::OperatorHandle>(m, "_DispatchOperatorHandle")
      .def("schema", &c10::OperatorHandle::schema);

  // TODO: figure out how to do chaining
  py::class_<torch::Library>(m, "_DispatchModule")
      // Some of these APIs are only for testing and do not work in multipy
      // environment
      .def(
          "def_",
          [](py::object self, const char* schema, const char* alias) {
            TORCH_INTERNAL_ASSERT(isMainPyInterpreter());
            self.cast<torch::Library&>().def(
                torch::schema(schema, parseAliasAnalysisKind(alias)));
            return self;
          },
          "",
          py::arg("schema"),
          py::arg("alias") = "")
      // Simulated "legacy" def where alias analysis kind is not set.
      // Ordinarily this can only be exercised from RegisterOperators() API
      // but I am not going to bind that here
      .def(
          "def_legacy",
          [](py::object self, const char* schema) {
            TORCH_INTERNAL_ASSERT(isMainPyInterpreter());
            self.cast<torch::Library&>().def(torch::jit::parseSchema(schema));
            return self;
          },
          "",
          py::arg("schema"))
      // We can't conveniently turn Python functions into valid functions
      // in the dispatcher.  So instead we provide a bunch of precanned
      // functions for testing purposes.  You're NOT intended to actually
      // call these functions; they're just here so we can actually register
      // something
      //
      // Mangling scheme: args_rets.  One character per.
      //  t = Tensor
      .def(
          "def_name_t_t",
          [](py::object self,
             const char* name,
             const char* dispatch,
             const char* debug) {
            TORCH_INTERNAL_ASSERT(isMainPyInterpreter());
            self.cast<torch::Library&>().def(
                name, dispatch_str(dispatch, [](const at::Tensor& a) {
                        return a;
                      }).debug(debug));
            return self;
          },
          "",
          py::arg("name"),
          py::arg("dispatch") = "",
          py::arg("debug") = "default_def_name_t_t")
      .def(
          "def_schema_t_t",
          [](py::object self,
             const char* schema,
             const char* dispatch,
             const char* alias,
             const char* debug) {
            TORCH_INTERNAL_ASSERT(isMainPyInterpreter());
            self.cast<torch::Library&>().def(
                torch::schema(schema, parseAliasAnalysisKind(alias)),
                dispatch_str(dispatch, [](const at::Tensor& a) {
                  return a;
                }).debug(debug));
            return self;
          },
          "",
          py::arg("name"),
          py::arg("dispatch") = "",
          py::arg("alias") = "",
          py::arg("debug") = "default_def_schema_t_t")
      // TODO: maybe consider deduplicating the definitions here, it's getting
      // pretty long
      .def(
          "impl_t_t",
          [](py::object self,
             const char* name,
             const char* dispatch,
             const char* debug) {
            TORCH_INTERNAL_ASSERT(isMainPyInterpreter());
            self.cast<torch::Library&>().impl(
                name, dispatch_str(dispatch, [](const at::Tensor& a) {
                        return a;
                      }).debug(debug));
            return self;
          },
          "",
          py::arg("name"),
          py::arg("dispatch") = "",
          py::arg("debug") = "impl_t_t")
      .def(
          "impl",
          [](py::object self,
             const char* name,
             // TODO: empty string no longer works
             c10::DispatchKey dispatch,
             py::object func) {
            HANDLE_TH_ERRORS
            auto& lib = self.cast<torch::Library&>();
            lib.impl(
                name,
                torch::dispatch(
                    dispatch,
                    CppFunction::makeFromBoxedFunctor(
                        std::make_unique<PythonKernelHolder>(func, dispatch))),
                register_or_verify());
            python_registrations_[lib._resolve(name)].insert_or_assign(
                dispatch,
                std::make_shared<c10::SafePyObject>(
                    func.release().ptr(), getPyInterpreter()));
            END_HANDLE_TH_ERRORS_PYBIND
          },
          "",
          py::arg("name"),
          py::arg("dispatch"),
          py::arg("func"))
      .def(
          "define",
          [](py::object self, const char* schema, const char* alias_analysis) {
            auto parsed_schema =
                torch::schema(schema, parseAliasAnalysisKind(alias_analysis));
            self.cast<torch::Library&>().def(
                std::move(parsed_schema), {}, register_or_verify());
            // TODO: this is dumb, had to make a second copy
            return torch::schema(schema, parseAliasAnalysisKind(alias_analysis))
                .name();
          },
          "",
          py::arg("schema"),
          py::arg("alias_analysis") = "")
      .def(
          "fallback_fallthrough",
          [](py::object self, const char* dispatch) {
            TORCH_INTERNAL_ASSERT(isMainPyInterpreter());
            self.cast<torch::Library&>().fallback(
                dispatch_str(dispatch, CppFunction::makeFallthrough()));
            return self;
          },
          "",
          py::arg("dispatch") = "");

  m.def(
      "_dispatch_library",
      [](const char* kind,
         std::string name,
         const char* dispatch,
         const char* file,
         uint32_t linenum) {
        HANDLE_TH_ERRORS
        return std::make_unique<torch::Library>(
            parseKind(kind),
            std::move(name),
            std::string(dispatch) == ""
                ? c10::nullopt
                : c10::make_optional(c10::parseDispatchKey(dispatch)),
            "/dev/null", // temporary workaround
            linenum);
        END_HANDLE_TH_ERRORS_PYBIND
      },
      "",
      py::arg("kind"),
      py::arg("name"),
      py::arg("dispatch"),
      py::arg("file") = "/dev/null",
      py::arg("linenum") = 0);

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

  m.def("_dispatch_has_kernel", [](const char* name) -> bool {
    auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
    return static_cast<bool>(op);
  });

  m.def(
      // Returns whether or not a direct kernel registration exists
      // for this <op_name, dispatch_key> pair.
      "_dispatch_has_kernel_for_dispatch_key",
      [](const char* name, c10::DispatchKey dispatch) -> bool {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        TORCH_CHECK(op, "operator ", name, " does not exist");
        return op->hasKernelForDispatchKey(dispatch);
      });

  m.def(
      "_dispatch_has_kernel_for_any_dispatch_key",
      [](const char* name, c10::DispatchKeySet ks) -> bool {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        TORCH_CHECK(op, "operator ", name, " does not exist");
        return op->hasKernelForAnyDispatchKey(ks);
      });

  m.def(
      // Returns whether or not there is an entry in the runtime computed
      // dispatch table, for this <op_name, dispatch_key> pair. For example, if
      // "op" has a `CompositeImplicitAutograd` kernel, Then
      // _dispatch_has_computed_kernel_for_dispatch_key(op, backend) will return
      // true for all backends that are part of the alias set for
      // CompositeImplicitAutograd.
      "_dispatch_has_computed_kernel_for_dispatch_key",
      [](const char* name, const char* dispatch) -> bool {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        TORCH_CHECK(op, "operator ", name, " does not exist");
        return op->hasComputedKernelForDispatchKey(
            c10::parseDispatchKey(dispatch));
      });

  m.def("_dispatch_find_dangling_impls", []() -> std::vector<std::string> {
    auto danglingImpls = c10::Dispatcher::singleton().findDanglingImpls();

    std::vector<std::string> states;
    states.reserve(danglingImpls.size());
    for (auto& danglingImpl : danglingImpls) {
      states.push_back(danglingImpl.dumpState());
    }

    return states;
  });

  m.def("_dispatch_get_all_op_names", []() -> std::vector<std::string> {
    auto op_names = c10::Dispatcher::singleton().getAllOpNames();

    std::vector<std::string> names;
    names.reserve(op_names.size());
    for (auto& op : op_names) {
      std::stringstream ss;
      ss << op.name;
      if (!op.overload_name.empty()) {
        ss << "." << op.overload_name;
      }
      names.push_back(ss.str());
    }

    return names;
  });

  m.def(
      "_dispatch_tls_set_dispatch_key_excluded",
      [](c10::DispatchKey dispatch_key, bool desired_state) {
        c10::impl::tls_set_dispatch_key_excluded(dispatch_key, desired_state);
      });
  m.def(
      "_dispatch_tls_is_dispatch_key_excluded",
      [](c10::DispatchKey dispatch_key) {
        return c10::impl::tls_is_dispatch_key_excluded(dispatch_key);
      });

  m.def("_dispatch_isTensorSubclassLike", [](const at::Tensor& tensor) {
    return at::isTensorSubclassLike(tensor);
  });

  m.def("_dispatch_key_name", [](c10::DispatchKey k) {
    return c10::toString(k);
  });
  m.def("_dispatch_key_parse", [](c10::DispatchKey k) { return k; });
  m.def("_dispatch_num_backends", []() { return c10::num_backends; });

#define DEF_ONE(n) .value(#n, c10::DispatchKey::n)

  py::enum_<c10::DispatchKey>(m, "DispatchKey") DEF_ONE(Undefined)
      DEF_ONE(CompositeExplicitAutogradNonFunctional)
          DEF_ONE(CompositeExplicitAutograd)
              DEF_ONE(CompositeImplicitAutogradNestedTensor)
                  DEF_ONE(CompositeImplicitAutograd) DEF_ONE(AutogradOther)
                      DEF_ONE(Autograd) DEF_ONE(BackendSelect)
                          DEF_ONE(ADInplaceOrView) DEF_ONE(PythonTLSSnapshot)
                              DEF_ONE(Python)
                              DEF_ONE(FuncTorchDynamicLayerFrontMode)

#define DEF_SINGLE(n, prefix) .value(#prefix #n, c10::DispatchKey::prefix##n)
#define DEF_MULTIPLE(fullname, prefix)              \
  DEF_SINGLE(, fullname)                            \
  DEF_SINGLE(, StartOf##fullname##Backends)         \
  C10_FORALL_BACKEND_COMPONENTS(DEF_SINGLE, prefix) \
  DEF_SINGLE(, EndOf##fullname##Backends)

                                  C10_FORALL_FUNCTIONALITY_KEYS(DEF_MULTIPLE)

#undef DEF_MULTIPLE
#undef DEF_SINGLE
                                      ;

  py::class_<c10::DispatchKeySet>(m, "DispatchKeySet")
      .def(py::init<c10::DispatchKey>())
      .def("__or__", &c10::DispatchKeySet::operator|)
      .def("__sub__", &c10::DispatchKeySet::operator-)
      .def("__and__", &c10::DispatchKeySet::operator&)
      .def("highestPriorityTypeId", &c10::DispatchKeySet::highestPriorityTypeId)
      .def("has", &c10::DispatchKeySet::has)
      .def("__repr__", [](c10::DispatchKeySet d) { return c10::toString(d); });

  m.attr("_dispatch_autogradother_backends") =
      py::cast(c10::autogradother_backends);

  m.def("_dispatch_has_backend_fallback", [](c10::DispatchKey t) {
    return c10::Dispatcher::singleton().hasBackendFallbackForDispatchKey(t);
  });

  m.def("_dispatch_keyset_full_after", [](c10::DispatchKey t) {
    return c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, t);
  });

  m.def("_dispatch_keyset_to_string", [](c10::DispatchKeySet keyset) {
    return c10::toString(keyset);
  });

  m.def("_dispatch_get_backend_keyset_from_autograd", [](c10::DispatchKey k) {
    return c10::getBackendKeySetFromAutograd(k);
  });

  m.def("_dispatch_keys", [](const at::Tensor& tensor) {
    auto* impl = tensor.unsafeGetTensorImpl();
    return impl->key_set();
  });
  m.def("_dispatch_tls_local_include_set", []() {
    return c10::impl::tls_local_dispatch_key_set().included_;
  });
  m.def("_dispatch_tls_local_exclude_set", []() {
    return c10::impl::tls_local_dispatch_key_set().excluded_;
  });
  m.def(
      "_dispatch_is_included_in_alias",
      [](c10::DispatchKey a, c10::DispatchKey b) {
        return c10::isIncludedInAlias(a, b);
      });
  py::class_<c10::impl::ExcludeDispatchKeyGuard>(m, "ExcludeDispatchKeyGuard")
      .def(py::init<c10::DispatchKeySet>());

  py::class_<at::AutoDispatchBelowAutograd>(m, "_AutoDispatchBelowAutograd")
      .def(py::init<>());

  // Prints out the name of every operator that has a kernel registered to the
  // Dispatcher under [dispatch_key]. If no arguments are specified, it'll print
  // out the name of every operator that the Dispatcher knows of. This can be
  // useful to answer questions like "list all operators that do not have a CPU
  // kernel".
  m.def(
      "_dispatch_print_registrations_for_dispatch_key",
      [](const char* dispatch_key = "") {
        auto k = std::string(dispatch_key) == ""
            ? c10::nullopt
            : c10::make_optional(c10::parseDispatchKey(dispatch_key));
        auto op_names =
            c10::Dispatcher::singleton().getRegistrationsForDispatchKey(k);
        for (auto& op : op_names) {
          std::cout << op << std::endl;
        }
      },
      py::arg("dispatch_key") = static_cast<const char*>(""));

  m.def(
      "_dispatch_get_registrations_for_dispatch_key",
      [](const char* dispatch_key = "") {
        auto k = std::string(dispatch_key) == ""
            ? c10::nullopt
            : c10::make_optional(c10::parseDispatchKey(dispatch_key));
        auto op_names =
            c10::Dispatcher::singleton().getRegistrationsForDispatchKey(k);
        std::vector<std::string> names;
        names.reserve(op_names.size());
        for (auto& op : op_names) {
          names.push_back(
              op.name + (op.overload_name == "" ? "" : "." + op.overload_name));
        }
        return names;
      },
      py::arg("dispatch_key") = static_cast<const char*>(""));

  m.def(
      "_dispatch_is_main_interpreter", []() { return isMainPyInterpreter(); });

  m.def("_are_functorch_transforms_active", []() {
    auto include_set = c10::impl::tls_local_dispatch_key_set().included_;
    return (
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode));
  });
}

// TODO: dedupe with the kernel
void python_op_registration_trampoline_impl(
    const c10::OperatorHandle& op,
    c10::DispatchKey key,
    torch::jit::Stack* stack) {
  auto arguments = torch::jit::pop(*stack, op.schema().arguments().size());
  py::gil_scoped_acquire g;
  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
  const auto& func = python_registrations_[op.operator_name()][key];
  TORCH_INTERNAL_ASSERT(func != nullptr);
  auto* pyobj = func->ptr(getPyInterpreter());
  TORCH_INTERNAL_ASSERT(pyobj != nullptr);
  auto obj = py::reinterpret_steal<py::object>(
      PyObject_Call(pyobj, args_kwargs.first.ptr(), args_kwargs.second.ptr()));
  if (!obj) {
    throw python_error();
  }
  pushPyOutToStack(op, stack, obj, "PythonKernelHolder");
}

} // namespace dispatch
} // namespace impl
} // namespace torch
