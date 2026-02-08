#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/utils/python_dispatch.h>

#include <ATen/DTensorState.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/NestedIntSymNodeImpl.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <ATen/functorch/BatchedTensorImpl.h>
#include <torch/library.h>

#include <c10/core/SafePyObject.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/tensor_new.h>

#include <c10/util/flat_hash_map.h>
#include <torch/csrc/inductor/aoti_eager/kernel_holder.h>
#include <torch/csrc/utils/python_raii.h>

#include <iostream>
#include <utility>

namespace py = pybind11;

namespace torch::impl::dispatch {

// Global storage for leaked Python filenames to ensure they remain valid
// for the lifetime of Library objects
static std::vector<std::string> leaked_python_filenames_;

// NB: I'd like to index this on OperatorHandle, but I can't, as I can't
// guarantee that the main interpreter has finish doing all registrations before
// the other interpreters start banging on it
static ska::flat_hash_map<
    c10::OperatorName,
    ska::flat_hash_map<c10::DispatchKey, std::shared_ptr<c10::SafePyObject>>>
    python_registrations_;

static torch::Library::Kind parseKind(const std::string& k) {
  static std::unordered_map<std::string, torch::Library::Kind> kind_map = {
      {"DEF", torch::Library::DEF},
      {"IMPL", torch::Library::IMPL},
      {"FRAGMENT", torch::Library::FRAGMENT},
  };
  auto it = kind_map.find(k);
  TORCH_CHECK(it != kind_map.end(), "could not parse ", k);
  return it->second;
}
static c10::AliasAnalysisKind parseAliasAnalysisKind(const std::string& k) {
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
inline static torch::CppFunction dispatch_str(const char* key, Func&& raw_f) {
  if (key[0] != '\0') {
    return torch::dispatch(
        c10::parseDispatchKey(key), std::forward<Func>(raw_f));
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
  EnableHermeticPyObject(const EnableHermeticPyObject&) = delete;
  EnableHermeticPyObject(EnableHermeticPyObject&&) = delete;
  EnableHermeticPyObject& operator=(const EnableHermeticPyObject&) = delete;
  EnableHermeticPyObject& operator=(EnableHermeticPyObject&&) = delete;
  bool old_;
  bool old_excluded_python_;
  bool old_python_;
  bool old_python_snapshot_;
};

class PythonKernelHolder : public c10::OperatorKernel {
  c10::SafePyObject func_;
  c10::DispatchKey dispatch_key_;
  // If "with_keyset", then we expect a keyset as the first arg.
  bool with_keyset_;
  // If "with_op", then we expect the op as first arg (or second if keyset)
  bool with_op_;

 public:
  PythonKernelHolder(
      py::object func,
      c10::DispatchKey dispatch_key,
      bool with_keyset = false,
      bool with_op = false)
      : func_(func.release().ptr(), getPyInterpreter()),
        dispatch_key_(dispatch_key),
        with_keyset_(with_keyset),
        with_op_(with_op) {}

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
          ->python_op_registration_trampoline(
              op, dispatch_key_, keyset, stack, with_keyset_, with_op_);
      return;
    }

    const auto& schema = op.schema();
    const auto num_arguments = schema.arguments().size();

    // Otherwise, find a PyInterpreter on a Tensor IF if has Python key (which
    // means it's a nontrivial tensor subclass)
    for (const auto& ivalue : torch::jit::last(*stack, num_arguments)) {
      if (ivalue.isTensor()) {
        auto* interpreter =
            ivalue.unsafeToTensorImpl()->pyobj_slot()->pyobj_interpreter();
        if (interpreter &&
            ivalue.unsafeToTensorImpl()->key_set().has(
                at::DispatchKey::Python)) {
          (*interpreter)
              ->python_op_registration_trampoline(
                  op, dispatch_key_, keyset, stack, with_keyset_, with_op_);
          return;
        }
      } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
        // NB: use toListRef as it doesn't induce refcount bumps
        // (toTensorListRef is not a thing)
        for (const auto& nv : ivalue.toListRef()) {
          if (nv.isNone()) {
            continue;
          }
          auto* interpreter =
              nv.unsafeToTensorImpl()->pyobj_slot()->pyobj_interpreter();
          if (interpreter &&
              nv.unsafeToTensorImpl()->key_set().has(at::DispatchKey::Python)) {
            (*interpreter)
                ->python_op_registration_trampoline(
                    op, dispatch_key_, keyset, stack, with_keyset_, with_op_);
            return;
          }
        }
      }
    }

    // Nothing requires the operator to be homed to a specific interpreter, so
    // run it on the current interpreter

    auto arguments = torch::jit::pop(*stack, op.schema().arguments().size());
    py::gil_scoped_acquire g;
    auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
    auto func =
        py::reinterpret_borrow<py::object>(func_.ptr(getPyInterpreter()));
    auto obj = with_op_ ? with_keyset_
            ? func(
                  keyset,
                  torch::detail::getTorchApiFunction(op),
                  *args_kwargs.first,
                  **args_kwargs.second)
            : func(
                  torch::detail::getTorchApiFunction(op),
                  *args_kwargs.first,
                  **args_kwargs.second)
        : with_keyset_ ? func(keyset, *args_kwargs.first, **args_kwargs.second)
                        : func(*args_kwargs.first, **args_kwargs.second);
    if (!obj) {
      throw python_error();
    }
    pushPyOutToStack(op, stack, obj, "PythonKernelHolder");
  }
};

// @todo sahanp: Afait only register is used in the codebase. This can be
// removed / simplified
static torch::_RegisterOrVerify register_or_verify() {
  return torch::_RegisterOrVerify::REGISTER;
}

static py::object ophandle_call_boxed(
    const c10::OperatorHandle& handle,
    const py::args& args,
    const py::kwargs& kwargs) {
  auto stack = torch::jit::createStackForSchema(
      handle.schema(),
      args,
      kwargs,
      /*self=*/std::nullopt);
  {
    pybind11::gil_scoped_release no_gil_guard;
    handle.callBoxed(stack);
  }
  return torch::jit::createPyObjectForStack(std::move(stack));
}

// A small RAII guard that lets you explicitly *remove* a key from the TLS
// exclude set.
class SetExcludeDispatchKeyGuard {
 public:
  SetExcludeDispatchKeyGuard(at::DispatchKey k, bool set_excluded)
      : k(k), old(c10::impl::tls_is_dispatch_key_excluded(k)) {
    c10::impl::tls_set_dispatch_key_excluded(k, set_excluded);
  }
  ~SetExcludeDispatchKeyGuard() {
    c10::impl::tls_set_dispatch_key_excluded(k, old);
  }
  SetExcludeDispatchKeyGuard(const SetExcludeDispatchKeyGuard&) = delete;
  SetExcludeDispatchKeyGuard operator=(const SetExcludeDispatchKeyGuard&) =
      delete;
  SetExcludeDispatchKeyGuard(SetExcludeDispatchKeyGuard&&) = delete;
  SetExcludeDispatchKeyGuard operator=(SetExcludeDispatchKeyGuard&&) = delete;

 private:
  at::DispatchKey k;
  bool old;
};

void initDispatchBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<c10::OperatorHandle>(m, "_DispatchOperatorHandle")
      .def("schema", &c10::OperatorHandle::schema)
      .def("debug", &c10::OperatorHandle::debug)
      .def(
          "redispatch_boxed",
          [](const py::object& self,
             c10::DispatchKeySet keyset,
             py::args args,
             const py::kwargs& kwargs) {
            auto& handle = self.cast<c10::OperatorHandle&>();
            auto stack = torch::jit::createStackForSchema(
                handle.schema(),
                std::move(args),
                kwargs,
                /*self=*/std::nullopt);
            {
              pybind11::gil_scoped_release no_gil_guard;
              handle.redispatchBoxed(keyset, &stack);
            }
            return torch::jit::createPyObjectForStack(std::move(stack));
          });

  m.def("_dispatch_call_boxed", &ophandle_call_boxed);

  // TODO: figure out how to do chaining
  py::class_<torch::Library>(m, "_DispatchModule")
      .def(
          "reset",
          [](const py::object& self) {
            self.cast<torch::Library&>().reset();
            return;
          },
          "")
      // Some of these APIs are only for testing and do not work in
      // multipy environment  // codespell:ignore multipy
      .def(
          "def_",
          [](py::object self, const char* schema, const char* alias) {
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
          "impl_with_aoti_compile",
          [](const py::object& self,
             const char* ns,
             const char* op_name_with_overload,
             c10::DispatchKey dispatch) {
            HANDLE_TH_ERRORS
            std::string reg_op_name =
                std::string(ns).append("::").append(op_name_with_overload);

            auto& lib = self.cast<torch::Library&>();
            lib.impl(
                reg_op_name.c_str(),
                torch::dispatch(
                    dispatch,
                    CppFunction::makeFromBoxedFunctor(
                        std::make_unique<
                            torch::inductor::AOTIPythonKernelHolder>(
                            dispatch, ns, op_name_with_overload))),
                register_or_verify());
            END_HANDLE_TH_ERRORS_PYBIND
          },
          "",
          py::arg("ns"),
          py::arg("op_name_with_overload"),
          py::arg("dispatch"))
      .def(
          "impl",
          [](const py::object& self,
             const char* name,
             // TODO: empty string no longer works
             c10::DispatchKey dispatch,
             py::object func,
             bool with_keyset) {
            HANDLE_TH_ERRORS
            auto& lib = self.cast<torch::Library&>();
            if (func.is(py::module::import("torch.library")
                            .attr("fallthrough_kernel"))) {
              lib.impl(
                  name,
                  torch::dispatch(dispatch, CppFunction::makeFallthrough()),
                  register_or_verify());
            } else {
              lib.impl(
                  name,
                  torch::dispatch(
                      dispatch,
                      CppFunction::makeFromBoxedFunctor(
                          std::make_unique<PythonKernelHolder>(
                              func, dispatch, with_keyset))),
                  register_or_verify());
              python_registrations_[lib._resolve(name)].insert_or_assign(
                  dispatch,
                  std::make_shared<c10::SafePyObject>(
                      func.release().ptr(), getPyInterpreter()));
            }
            END_HANDLE_TH_ERRORS_PYBIND
          },
          "",
          py::arg("name"),
          py::arg("dispatch"),
          py::arg("func"),
          py::arg("with_keyset") = false)
      .def(
          "define",
          [](const py::object& self,
             const char* schema,
             const char* alias_analysis,
             const std::vector<at::Tag>& tags) {
            auto parsed_schema =
                torch::schema(schema, parseAliasAnalysisKind(alias_analysis));
            self.cast<torch::Library&>().def(
                std::move(parsed_schema), tags, register_or_verify());
            // TODO: this is dumb, had to make a second copy
            return torch::schema(schema, parseAliasAnalysisKind(alias_analysis))
                .name();
          },
          "",
          py::arg("schema"),
          py::arg("alias_analysis") = "",
          py::arg("tags") = std::vector<at::Tag>())
      .def(
          "fallback_fallthrough",
          [](py::object self, const char* dispatch) {
            self.cast<torch::Library&>().fallback(
                dispatch_str(dispatch, CppFunction::makeFallthrough()));
            return self;
          },
          "",
          py::arg("dispatch") = "")
      .def(
          "fallback",
          [](const py::object& self,
             c10::DispatchKey dispatch,
             const py::object& func,
             bool with_keyset) {
            HANDLE_TH_ERRORS
            auto& lib = self.cast<torch::Library&>();
            if (func.is(py::module::import("torch.library")
                            .attr("fallthrough_kernel"))) {
              lib.fallback(
                  torch::dispatch(dispatch, CppFunction::makeFallthrough()));
            } else {
              lib.fallback(torch::dispatch(
                  dispatch,
                  CppFunction::makeFromBoxedFunctor(
                      std::make_unique<PythonKernelHolder>(
                          func, dispatch, with_keyset, /*with_op*/ true))));
            }
            END_HANDLE_TH_ERRORS_PYBIND
          },
          "",
          py::arg("dispatch"),
          py::arg("func"),
          py::arg("with_keyset") = false)
      .def(
          "register_ad_inplace_or_view_fallback",
          [](const py::object& self, const char* name) {
            HANDLE_TH_ERRORS
            auto& lib = self.cast<torch::Library&>();
            lib.impl(
                name,
                c10::DispatchKey::ADInplaceOrView,
                torch::autograd::autogradNotImplementedInplaceOrViewFallback());
            END_HANDLE_TH_ERRORS_PYBIND
          },
          "",
          py::arg("name"));

  m.def(
      "_dispatch_library",
      [](const char* kind,
         std::string name,
         const char* dispatch,
         const char* file,
         uint32_t linenum) {
        HANDLE_TH_ERRORS
        // Store the file string in global storage to ensure it remains valid
        // for the lifetime of the Library object
        leaked_python_filenames_.emplace_back(file);
        const char* leaked_file = leaked_python_filenames_.back().c_str();

        return std::make_unique<torch::Library>(
            parseKind(kind),
            std::move(name),
            std::string(dispatch).empty()
                ? std::nullopt
                : std::make_optional(c10::parseDispatchKey(dispatch)),
            leaked_file,
            linenum);
        END_HANDLE_TH_ERRORS_PYBIND
      },
      "",
      py::arg("kind"),
      py::arg("name"),
      py::arg("dispatch"),
      py::arg("file") = "/dev/null",
      py::arg("linenum") = 0);

  m.def(
      "_dispatch_clear_leaked_python_filenames",
      []() { leaked_python_filenames_.clear(); },
      "Clear the global storage of leaked Python filenames. "
      "WARNING: Only call this if you're sure no Library objects are still using the filenames.");

  m.def(
      "_dispatch_find_schema_or_throw",
      [](const char* name, const char* overload_name) -> c10::OperatorHandle {
        return c10::Dispatcher::singleton().findSchemaOrThrow(
            name, overload_name);
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
      // Returns whether or not the kernel for this dispatach key is a
      // fallthrough kernel
      "_dispatch_kernel_for_dispatch_key_is_fallthrough",
      [](const char* name, c10::DispatchKey dispatch) -> bool {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        return op->isKernelFallthroughKernel(dispatch);
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

  // Bind SafeKernelFunction class
  py::class_<c10::SafeKernelFunction>(m, "_SafeKernelFunction")
      .def(
          "call_boxed",
          [](const c10::SafeKernelFunction& self,
             c10::DispatchKeySet keyset,
             py::args args,
             const py::kwargs& kwargs) {
            const auto& op = self.opHandle();
            auto stack = torch::jit::createStackForSchema(
                op.schema(),
                std::move(args),
                kwargs,
                /*self=*/std::nullopt);
            self.callBoxed(op, keyset, &stack);
            return torch::jit::createPyObjectForStack(std::move(stack));
          })
      .def(
          "__repr__",
          [](const c10::SafeKernelFunction& self) {
            return "SafeKernelFunction(debug='" + self.debug() + "')";
          })
      .def_property_readonly(
          "op_handle", [](const c10::SafeKernelFunction& self) -> py::object {
            return py::cast(self.opHandle());
          });

  m.def(
      "_dispatch_get_computed_kernel_for_dispatch_key",
      [](const char* name,
         c10::DispatchKey dispatch) -> c10::SafeKernelFunction {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        TORCH_CHECK(op, "operator ", name, " does not exist");
        return op->getComputedKernelForDispatchKey(dispatch);
      });

  m.def("_dispatch_find_dangling_impls", []() -> std::vector<std::string> {
    auto danglingImpls = c10::Dispatcher::singleton().findDanglingImpls();

    std::vector<std::string> states;
    states.reserve(danglingImpls.size());
    for (auto& danglingImpl : danglingImpls) {
      states.emplace_back(danglingImpl.dumpState());
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
        ss << '.' << op.overload_name;
      }
      names.emplace_back(std::move(ss).str());
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
  m.def(
      "_dispatch_tls_set_dispatch_key_included",
      [](c10::DispatchKey dispatch_key, bool desired_state) {
        c10::impl::tls_set_dispatch_key_included(dispatch_key, desired_state);
      });
  m.def(
      "_dispatch_tls_is_dispatch_key_included",
      [](c10::DispatchKey dispatch_key) {
        return c10::impl::tls_is_dispatch_key_included(dispatch_key);
      });

  m.def("_dispatch_isTensorSubclassLike", [](const at::Tensor& tensor) {
    return at::isTensorSubclassLike(tensor);
  });

  m.def("_dispatch_key_name", [](c10::DispatchKey k) {
    return c10::toString(k);
  });
  m.def("_dispatch_key_parse", [](c10::DispatchKey k) { return k; });
  m.def("_to_functionality_key", [](c10::DispatchKey k) {
    return c10::toFunctionalityKey(k);
  });
  // E.g. given `DispatchKey::AutogradFunctionality`, returns a keyset of:
  //  AutogradCPU
  //  AutogradCUDA
  //  ...
  //  AutogradPrivateUse3
  m.def("_functionality_to_backend_keys", [](c10::DispatchKey key) {
    std::vector<c10::DispatchKey> keys;
    if (c10::isPerBackendFunctionalityKey(key)) {
      auto ks = c10::DispatchKeySet(key) |
          c10::DispatchKeySet(c10::DispatchKeySet::RAW, c10::full_backend_mask);
      for (auto k : ks) {
        keys.push_back(k);
      }
    } else {
      keys.push_back(key);
    }
    return keys;
  });
  m.def("_dispatch_num_backends", []() { return c10::num_backends; });

#define DEF_ONE(n) .value(#n, c10::DispatchKey::n)

  py::enum_<c10::DispatchKey>(m, "DispatchKey")
      // clang-format off
      DEF_ONE(Undefined)
      DEF_ONE(CompositeExplicitAutogradNonFunctional)
      DEF_ONE(CompositeExplicitAutograd)
      DEF_ONE(CompositeImplicitAutogradNestedTensor)
      DEF_ONE(CompositeImplicitAutograd)
      // NestedTensor is not a backend key
      DEF_ONE(AutogradNestedTensor)
      DEF_ONE(AutogradOther)
      DEF_ONE(Autograd)
      DEF_ONE(Conjugate)
      DEF_ONE(ZeroTensor)
      DEF_ONE(Negative)
      DEF_ONE(BackendSelect)
      DEF_ONE(ADInplaceOrView)
      DEF_ONE(PythonTLSSnapshot)
      DEF_ONE(Python)
      DEF_ONE(FuncTorchDynamicLayerFrontMode)
      DEF_ONE(FuncTorchDynamicLayerBackMode)
      DEF_ONE(FuncTorchBatchedDecomposition)
      DEF_ONE(FuncTorchBatched)
      DEF_ONE(FuncTorchVmapMode)
      DEF_ONE(FuncTorchGradWrapper)
      DEF_ONE(PythonDispatcher)
      DEF_ONE(PreDispatch)
      DEF_ONE(Functionalize)
      DEF_ONE(AutocastCPU)
      DEF_ONE(AutocastMPS)
      DEF_ONE(AutocastXPU)
      DEF_ONE(AutocastHPU)
      DEF_ONE(AutocastIPU)
      DEF_ONE(AutocastCUDA)
      DEF_ONE(AutocastPrivateUse1)
  // clang-format on

#define DEF_SINGLE(n, prefix) .value(#prefix #n, c10::DispatchKey::prefix##n)
#define DEF_MULTIPLE(fullname, prefix)              \
  DEF_SINGLE(, fullname)                            \
  DEF_SINGLE(, StartOf##fullname##Backends)         \
  C10_FORALL_BACKEND_COMPONENTS(DEF_SINGLE, prefix) \
  DEF_SINGLE(, EndOf##fullname##Backends)

      // clang-format off
  C10_FORALL_FUNCTIONALITY_KEYS(DEF_MULTIPLE)
  // clang-format on

#undef DEF_MULTIPLE
#undef DEF_SINGLE
          ;

  py::class_<c10::DispatchKeySet>(m, "DispatchKeySet")
      .def(py::init<c10::DispatchKey>())
      .def("__or__", &c10::DispatchKeySet::operator|)
      .def("__sub__", &c10::DispatchKeySet::operator-)
      .def("__and__", &c10::DispatchKeySet::operator&)
      .def("raw_repr", &c10::DispatchKeySet::raw_repr)
      .def("highestPriorityTypeId", &c10::DispatchKeySet::highestPriorityTypeId)
      .def(
          "remove",
          [](c10::DispatchKeySet self, c10::DispatchKey k) {
            return self.remove(k);
          })
      .def(
          "add",
          [](c10::DispatchKeySet self, c10::DispatchKey k) {
            return self.add(k);
          })
      .def("has", &c10::DispatchKeySet::has)
      .def("__repr__", [](c10::DispatchKeySet d) { return c10::toString(d); })
      .def(
          "__eq__",
          [](c10::DispatchKeySet self, c10::DispatchKeySet other) {
            return self.raw_repr() == other.raw_repr();
          })
      .def(py::pickle(
          [](const c10::DispatchKeySet&
                 obj) { // __getstate__ : creates tuple of state
            return py::make_tuple(obj.raw_repr());
          },
          [](const py::tuple& t) { // __setstate__ : restores state from tuple
            TORCH_CHECK(
                t.size() == 1, "__setstate__ expected tuple with one element");
            return c10::DispatchKeySet::from_raw_repr(t[0].cast<uint64_t>());
          }))
      .def_static("from_raw_repr", &c10::DispatchKeySet::from_raw_repr);

  m.attr("_dispatch_autogradother_backends") =
      py::cast(c10::autogradother_backends);

  m.attr("_additional_keys_to_prop_for_wrapper_tensors") =
      py::cast(at::functorch::kKeysToPropagateToWrapper);

  m.attr("_after_autograd_keyset") = py::cast(c10::after_autograd_keyset);
  m.attr("_after_ADInplaceOrView_keyset") =
      py::cast(c10::after_ADInplaceOrView_keyset);

  m.def("_dispatch_has_backend_fallback", [](c10::DispatchKey t) {
    return c10::Dispatcher::singleton().hasBackendFallbackForDispatchKey(t);
  });

  m.def("_dispatch_keyset_full_after", [](c10::DispatchKey t) {
    return c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, t);
  });

  m.def("_dispatch_keyset_full", []() {
    return c10::DispatchKeySet(c10::DispatchKeySet::FULL);
  });

  m.def("_dispatch_is_alias_key", c10::isAliasDispatchKey);

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
  m.def("_functionalization_reapply_views_tls", []() {
    return at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
  });
  m.def(
      "_dispatch_is_included_in_alias",
      [](c10::DispatchKey a, c10::DispatchKey b) {
        return c10::isIncludedInAlias(a, b);
      });

  // DEPRECATED, please don't use this. Instead use
  // torch._C._ExcludeDispatchKeyGuard
  py_context_manager_DEPRECATED<
      c10::impl::ExcludeDispatchKeyGuard,
      c10::DispatchKeySet>(m, "ExcludeDispatchKeyGuard");

  py_context_manager<
      c10::impl::ForceDispatchKeyGuard,
      c10::DispatchKeySet,
      c10::DispatchKeySet>(m, "_ForceDispatchKeyGuard");
  py_context_manager<c10::impl::ForceDispatchKeyGuard>(
      m, "_PreserveDispatchKeyGuard");
  py_context_manager<c10::impl::IncludeDispatchKeyGuard, c10::DispatchKey>(
      m, "_IncludeDispatchKeyGuard");
  py_context_manager<c10::impl::ExcludeDispatchKeyGuard, c10::DispatchKeySet>(
      m, "_ExcludeDispatchKeyGuard");
  py_context_manager<SetExcludeDispatchKeyGuard, c10::DispatchKey, bool>(
      m, "_SetExcludeDispatchKeyGuard");

  py_context_manager_DEPRECATED<at::AutoDispatchBelowAutograd>(
      m, "_AutoDispatchBelowAutograd");
  py_context_manager<at::AutoDispatchBelowADInplaceOrView>(
      m, "_AutoDispatchBelowADInplaceOrView");

  // Prints out the name of every operator that has a kernel registered to the
  // Dispatcher under [dispatch_key]. If no arguments are specified, it'll print
  // out the name of every operator that the Dispatcher knows of. This can be
  // useful to answer questions like "list all operators that do not have a CPU
  // kernel".
  m.def(
      "_dispatch_print_registrations_for_dispatch_key",
      [](const char* dispatch_key = "") {
        auto k = std::string(dispatch_key).empty()
            ? std::nullopt
            : std::make_optional(c10::parseDispatchKey(dispatch_key));
        auto op_names =
            c10::Dispatcher::singleton().getRegistrationsForDispatchKey(k);
        for (auto& op : op_names) {
          std::cout << op << '\n';
        }
      },
      py::arg("dispatch_key") = static_cast<const char*>(""));

  m.def(
      "_parse_dispatch_key",
      [](const char* dispatch_key) -> std::optional<c10::DispatchKey> {
        try {
          return c10::parseDispatchKey(dispatch_key);
        } catch (const c10::Error&) {
          return std::nullopt;
        }
      });

  m.def(
      "_dispatch_get_registrations_for_dispatch_key",
      [](const char* dispatch_key = "") {
        auto k = std::string(dispatch_key).empty()
            ? std::nullopt
            : std::make_optional(c10::parseDispatchKey(dispatch_key));
        auto op_names =
            c10::Dispatcher::singleton().getRegistrationsForDispatchKey(k);
        std::vector<std::string> names;
        names.reserve(op_names.size());
        for (auto& op : op_names) {
          names.emplace_back(
              op.name +
              (op.overload_name.empty() ? "" : "." + op.overload_name));
        }
        return names;
      },
      py::arg("dispatch_key") = static_cast<const char*>(""));
  m.def(
      "_dispatch_set_report_error_callback",
      [](c10::OperatorHandle& handle, py::object callback) {
        auto obj = callback.release().ptr();
        auto callback_obj =
            std::make_unique<c10::SafePyObject>(obj, getPyInterpreter());
        handle.setReportErrorCallback_(std::move(callback_obj));
      });

  m.def("_dispatch_pystub", [](const char* name, const char* overload) {
    return c10::Dispatcher::singleton().getPyStub(
        c10::OperatorName(name, overload));
  });

  m.def("_replace_", [](const at::Tensor& a, const at::Tensor& b) {
    return at::functionalization::impl::replace_(a, b);
  });
  m.def("_propagate_xla_data", [](const at::Tensor& a, const at::Tensor& b) {
    at::functionalization::impl::propagate_xla_data(a, b);
  });
  m.def("_commit_update", [](const at::Tensor& a) {
    return at::functionalization::impl::commit_update(a);
  });
  m.def("_unsafe_reset_storage", [](const at::Tensor& a) {
    return at::functionalization::impl::unsafe_reset_storage(a);
  });

  m.def("_dispatch_key_for_device", [](const std::string& device_type) {
    auto device = c10::Device(device_type);
    TORCH_CHECK(
        !device.has_index(),
        "Expected device_type string to not have a device index; got ",
        device_type);
    return c10::toString(
        c10::computeDispatchKey(std::nullopt, std::nullopt, device));
  });

  m.def("_are_functorch_transforms_active", []() {
    auto include_set = c10::impl::tls_local_dispatch_key_set().included_;
    return (
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode));
  });

  m.def("_autocast_supported_devices", []() {
    std::vector<std::string> result;
    for (const auto device_type : at::autocast::_AUTOCAST_SUPPORTED_DEVICES) {
      result.emplace_back(
          c10::DeviceTypeName(device_type, /*lower_case*/ true));
    }
    return result;
  });

  m.def("_get_nested_int", [](int64_t data, int64_t coeff) {
    return c10::SymInt(c10::SymNode(
        c10::make_intrusive<c10::NestedIntSymNodeImpl>(data, coeff)));
  });

  m.def("_get_constant_bool_symnode", [](int64_t data) {
    return c10::SymNode(
        c10::make_intrusive<c10::ConstantSymNodeImpl<bool>>(data));
  });

  m.def("_non_sym_sizes", [](const at::Tensor& a) {
    return a.sizes(); // NB: NOT sym_size
  });

  m.def("_set_throw_on_mutable_data_ptr", [](const at::Tensor& t) {
    if (!t.unsafeGetTensorImpl()->has_storage()) {
      // If the Tensor doesn't have a storage, then accessing .data_ptr()
      // will already raise an error.
      return;
    }
    // Otherwise, set (on the StorageImpl) that accessing (mutable) data_ptr
    // will throw.
    t.unsafeGetTensorImpl()
        ->storage()
        .unsafeGetStorageImpl()
        ->set_throw_on_mutable_data_ptr();
  });

  // Invariant: you must ONLY call this with FakeTensors.
  m.def("_set_warn_deprecated_on_mutable_data_ptr", [](const at::Tensor& t) {
    if (!t.unsafeGetTensorImpl()->has_storage()) {
      // If the Tensor doesn't have a storage, then accessing .data_ptr()
      // will already raise an error.
      return;
    }
    t.unsafeGetTensorImpl()
        ->storage()
        .unsafeGetStorageImpl()
        ->set_warn_deprecated_on_mutable_data_ptr();
  });

  m.def("_only_lift_cpu_tensors", &torch::utils::only_lift_cpu_tensors);
  m.def("_set_only_lift_cpu_tensors", &torch::utils::set_only_lift_cpu_tensors);

  m.def(
      "_get_dtensor_allow_implicit_replication",
      &at::get_dtensor_allow_implicit_replication);
  m.def(
      "_set_dtensor_allow_implicit_replication",
      &at::set_dtensor_allow_implicit_replication);

  using c10::impl::TorchDispatchModeKey;
  py::enum_<TorchDispatchModeKey>(m, "_TorchDispatchModeKey")
      .value("FUNCTIONAL", TorchDispatchModeKey::FUNCTIONAL)
      .value("PROXY", TorchDispatchModeKey::PROXY)
      .value("FAKE", TorchDispatchModeKey::FAKE);
}

// TODO: dedupe with the kernel
void python_op_registration_trampoline_impl(
    const c10::OperatorHandle& op,
    c10::DispatchKey key,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack,
    bool with_keyset,
    bool with_op) {
  auto arguments = torch::jit::pop(*stack, op.schema().arguments().size());
  py::gil_scoped_acquire g;
  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
  const auto& func = python_registrations_[op.operator_name()][key];
  TORCH_INTERNAL_ASSERT(func != nullptr);
  auto* pyobj = func->ptr(getPyInterpreter());
  TORCH_INTERNAL_ASSERT(pyobj != nullptr);
  auto callable = py::reinterpret_borrow<py::object>(pyobj);
  auto obj = with_op ? with_keyset ? callable(
                                         keyset,
                                         torch::detail::getTorchApiFunction(op),
                                         *args_kwargs.first,
                                         **args_kwargs.second)
                                   : callable(
                                         torch::detail::getTorchApiFunction(op),
                                         *args_kwargs.first,
                                         **args_kwargs.second)
      : with_keyset ? callable(keyset, *args_kwargs.first, **args_kwargs.second)
                    : callable(*args_kwargs.first, **args_kwargs.second);
  if (!obj) {
    throw python_error();
  }
  pushPyOutToStack(op, stack, obj, "PythonKernelHolder");
}

} // namespace torch::impl::dispatch
