#include <torch/csrc/functorch/python_aot_autograd.h>

#include <ATen/record_function.h>
#include <c10/core/AutogradState.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/irange.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#include <torch/csrc/utils/object_ptr.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace torch::functorch::impl {

// NOLINTBEGIN(hicpp-exception-baseclass)

namespace {

// Built once by AOTAutograd wrapper codegen. Runtime reads this spec to set TLS
// state, optionally copy/detach selected args, and restore state after the
// compiled function returns.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct CompiledFnCallSpec {
  PyObject_HEAD
  std::vector<int64_t> indices_of_inps_to_detach;
  std::vector<int64_t> epilogue_args_idx;
  std::vector<int64_t> indices_of_inps_to_increment_version;
  bool trace_joint;
  bool disable_amp;
  bool record_runtime_overhead;
};

class ViewReplayGuard {
 public:
  explicit ViewReplayGuard(bool trace_joint)
      : restore_(trace_joint),
        prev_enabled_(
            c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    if (trace_joint && !prev_enabled_) {
      c10::AutogradState::get_tls_state().set_view_replay_enabled(true);
    }
  }

  ~ViewReplayGuard() {
    if (restore_ &&
        c10::AutogradState::get_tls_state().get_view_replay_enabled() !=
            prev_enabled_) {
      c10::AutogradState::get_tls_state().set_view_replay_enabled(
          prev_enabled_);
    }
  }

 private:
  bool restore_;
  bool prev_enabled_;
};

void CompiledFnCallSpec_dealloc(CompiledFnCallSpec* self) {
  self->indices_of_inps_to_increment_version.~vector<int64_t>();
  self->epilogue_args_idx.~vector<int64_t>();
  self->indices_of_inps_to_detach.~vector<int64_t>();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyTypeObject CompiledFnCallSpecType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._CompiledFnCallSpec", /* tp_name */
    sizeof(CompiledFnCallSpec), /* tp_basicsize */
    0, /* tp_itemsize */
    reinterpret_cast<destructor>(CompiledFnCallSpec_dealloc), /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_as_async */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
};

[[noreturn]] void raise_python_error() {
  throw python_error(); // @allow-raw-throw
}

std::vector<int64_t> read_int_tuple(PyObject* tuple) {
  TORCH_INTERNAL_ASSERT(PyTuple_CheckExact(tuple));
  std::vector<int64_t> result;
  const auto size = PyTuple_GET_SIZE(tuple);
  result.reserve(size);
  for (const auto i : c10::irange(size)) {
    PyObject* item = PyTuple_GET_ITEM(tuple, i);
    const auto value = PyLong_AsLongLong(item);
    if (value == -1 && PyErr_Occurred()) {
      raise_python_error();
    }
    TORCH_INTERNAL_ASSERT(value >= 0);
    result.push_back(value);
  }
  return result;
}

THPObjectPtr make_int_tuple(const std::vector<int64_t>& values) {
  THPObjectPtr tuple(PyTuple_New(values.size()));
  if (!tuple) {
    raise_python_error();
  }
  for (const auto i : c10::irange(values.size())) {
    PyObject* item = PyLong_FromLongLong(values[i]);
    if (!item) {
      raise_python_error();
    }
    PyTuple_SET_ITEM(tuple.get(), static_cast<Py_ssize_t>(i), item);
  }
  return tuple;
}

PyObject* CompiledFnCallSpec_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!kwargs || PyDict_GET_SIZE(kwargs) == 0);

  PyObject* detach_indices = nullptr;
  PyObject* epilogue_args_idx = nullptr;
  PyObject* increment_version_indices = nullptr;
  int trace_joint = 0;
  int disable_amp = 0;
  int record_runtime_overhead = 0;
  if (!PyArg_ParseTuple(
          args,
          "OpppOO:_CompiledFnCallSpec",
          &detach_indices,
          &trace_joint,
          &disable_amp,
          &record_runtime_overhead,
          &epilogue_args_idx,
          &increment_version_indices)) {
    raise_python_error();
  }
  auto detach_indices_vec = read_int_tuple(detach_indices);
  auto epilogue_args_idx_vec = read_int_tuple(epilogue_args_idx);
  auto increment_version_indices_vec =
      read_int_tuple(increment_version_indices);
  TORCH_INTERNAL_ASSERT(trace_joint || PyTuple_GET_SIZE(detach_indices) == 0);

  auto* spec = reinterpret_cast<CompiledFnCallSpec*>(type->tp_alloc(type, 0));
  if (!spec) {
    raise_python_error();
  }
  new (&spec->indices_of_inps_to_detach)
      std::vector<int64_t>(std::move(detach_indices_vec));
  new (&spec->epilogue_args_idx)
      std::vector<int64_t>(std::move(epilogue_args_idx_vec));
  new (&spec->indices_of_inps_to_increment_version)
      std::vector<int64_t>(std::move(increment_version_indices_vec));
  spec->trace_joint = trace_joint != 0;
  spec->disable_amp = disable_amp != 0;
  spec->record_runtime_overhead = record_runtime_overhead != 0;
  return reinterpret_cast<PyObject*>(spec);
  END_HANDLE_TH_ERRORS
}

PyObject* CompiledFnCallSpec_reduce(
    CompiledFnCallSpec* self,
    PyObject* _unused) {
  HANDLE_TH_ERRORS
  // This is required for compile_to_python, which is a debug option to
  // emits the generated runtime wrapper plus its closed-over
  // globals as standalone source.
  //
  // This __reduce__ lets it reconstruct the closed-over _CompiledFnCallSpec as:
  //   torch._C._CompiledFnCallSpec(detach, trace_joint, disable_amp,
  //       record_runtime_overhead, epilogue_args_idx, increment_version_idxs)
  // Normal runtime uses the live spec object directly; this is not the hot
  // path.
  THPObjectPtr detach_indices(make_int_tuple(self->indices_of_inps_to_detach));
  THPObjectPtr epilogue_args_idx(make_int_tuple(self->epilogue_args_idx));
  THPObjectPtr increment_version_indices(
      make_int_tuple(self->indices_of_inps_to_increment_version));

  THPObjectPtr ctor_args(PyTuple_New(6));
  if (!ctor_args) {
    raise_python_error();
  }
  THPObjectPtr trace_joint(PyBool_FromLong(self->trace_joint));
  THPObjectPtr disable_amp(PyBool_FromLong(self->disable_amp));
  THPObjectPtr record_runtime_overhead(
      PyBool_FromLong(self->record_runtime_overhead));
  PyTuple_SET_ITEM(ctor_args.get(), 0, detach_indices.release());
  PyTuple_SET_ITEM(ctor_args.get(), 1, trace_joint.release());
  PyTuple_SET_ITEM(ctor_args.get(), 2, disable_amp.release());
  PyTuple_SET_ITEM(ctor_args.get(), 3, record_runtime_overhead.release());
  PyTuple_SET_ITEM(ctor_args.get(), 4, epilogue_args_idx.release());
  PyTuple_SET_ITEM(ctor_args.get(), 5, increment_version_indices.release());

  THPObjectPtr result(PyTuple_New(2));
  if (!result) {
    raise_python_error();
  }
  Py_INCREF(&CompiledFnCallSpecType);
  PyTuple_SET_ITEM(
      result.get(), 0, reinterpret_cast<PyObject*>(&CompiledFnCallSpecType));
  PyTuple_SET_ITEM(result.get(), 1, ctor_args.release());
  return result.release();
  END_HANDLE_TH_ERRORS
}

static PyMethodDef CompiledFnCallSpec_methods[] = {
    {"__reduce__",
     reinterpret_cast<PyCFunction>(CompiledFnCallSpec_reduce),
     METH_NOARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyObject* get_inputs_at_indices(
    PyObject* boxed_args,
    const std::vector<int64_t>& indices) {
  // return {idx: boxed_args[arg] for idx in indices}
  TORCH_INTERNAL_ASSERT(PyList_CheckExact(boxed_args));
  const auto num_args = PyList_GET_SIZE(boxed_args);
  THPObjectPtr orig_inputs(PyDict_New());
  if (!orig_inputs) {
    raise_python_error();
  }
  for (const auto arg_idx : indices) {
    TORCH_INTERNAL_ASSERT(arg_idx < num_args);
    PyObject* key = PyLong_FromLongLong(arg_idx);
    if (!key) {
      raise_python_error();
    }
    PyObject* value =
        PyList_GET_ITEM(boxed_args, static_cast<Py_ssize_t>(arg_idx));
    if (PyDict_SetItem(orig_inputs.get(), key, value) < 0) {
      Py_DECREF(key);
      raise_python_error();
    }
    Py_DECREF(key);
  }
  return orig_inputs.release();
}

void increment_versions_at_indices(
    PyObject* boxed_args,
    const std::vector<int64_t>& indices) {
  TORCH_INTERNAL_ASSERT(PyList_CheckExact(boxed_args));
  const auto num_args = PyList_GET_SIZE(boxed_args);
  for (const auto arg_idx : indices) {
    TORCH_INTERNAL_ASSERT(arg_idx < num_args);
    PyObject* item =
        PyList_GET_ITEM(boxed_args, static_cast<Py_ssize_t>(arg_idx));
    TORCH_INTERNAL_ASSERT(THPVariable_Check(item));
    auto tensor = THPVariable_Unpack(item);
    if (!tensor.is_inference()) {
      torch::autograd::increment_version(tensor);
    }
  }
}

THPObjectPtr maybe_detach_at_indices(
    PyObject* boxed_args,
    const std::vector<int64_t>& indices) {
  // Returns nullptr if nothing to detach, otherwise a new list.
  if (indices.empty()) {
    return THPObjectPtr();
  }

  TORCH_INTERNAL_ASSERT(PyList_CheckExact(boxed_args));
  const auto num_args = PyList_GET_SIZE(boxed_args);
  THPObjectPtr call_args(PyList_New(num_args));
  if (!call_args) {
    raise_python_error();
  }

  for (const auto i : c10::irange(num_args)) {
    PyObject* item = PyList_GET_ITEM(boxed_args, i);
    Py_INCREF(item);
    PyList_SET_ITEM(call_args.get(), static_cast<Py_ssize_t>(i), item);
  }

  for (const auto arg_idx : indices) {
    TORCH_INTERNAL_ASSERT(arg_idx < num_args);
    const auto py_idx = static_cast<Py_ssize_t>(arg_idx);
    PyObject* item = PyList_GET_ITEM(call_args.get(), py_idx);
    if (!THPVariable_Check(item)) {
      continue;
    }
    PyObject* detached = THPVariable_Wrap(THPVariable_Unpack(item).detach());
    if (!detached) {
      raise_python_error();
    }
    Py_DECREF(item);
    PyList_SET_ITEM(call_args.get(), py_idx, detached);
  }

  return call_args;
}

PyObject* THPModule_aot_autograd_call_compiled_fn(
    PyObject* _unused,
    PyObject* const* args,
    Py_ssize_t nargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(nargs == 3);
  TORCH_INTERNAL_ASSERT(Py_IS_TYPE(args[0], &CompiledFnCallSpecType));
  const auto* spec = reinterpret_cast<CompiledFnCallSpec*>(args[0]);
  PyObject* compiled_fn = args[1];
  PyObject* boxed_args = args[2];

  // Begin profiler range if profiler is enabled.
  std::unique_ptr<at::RecordFunction> record_function;
  if (spec->record_runtime_overhead &&
      (torch::profiler::impl::ProfilerStateBase::getGlobal() != nullptr ||
       torch::profiler::impl::ProfilerStateBase::getTLS() != nullptr)) {
    record_function =
        std::make_unique<at::RecordFunction>(at::RecordScope::FUNCTION);
    record_function->before(
        std::string_view("AOTDispatcher Runtime Wrapper Prologue"));
  }

  // Inputs needed by alias and mutation epilogues.
  THPObjectPtr orig_inputs(
      get_inputs_at_indices(boxed_args, spec->epilogue_args_idx));

  increment_versions_at_indices(
      boxed_args, spec->indices_of_inps_to_increment_version);

  THPObjectPtr detached_args =
      maybe_detach_at_indices(boxed_args, spec->indices_of_inps_to_detach);
  PyObject* compiled_fn_args = detached_args ? detached_args.get() : boxed_args;

  c10::AutoGradMode grad_mode(spec->trace_joint);
  ViewReplayGuard view_replay_guard(spec->trace_joint);
  record_function.reset();
  std::optional<c10::impl::ExcludeDispatchKeyGuard> autocast_guard;
  if (spec->disable_amp) {
    autocast_guard.emplace(c10::autocast_dispatch_keyset);
  }

  PyObject* vectorcall_args[1] = {compiled_fn_args};
  THPObjectPtr all_outs(
      PyObject_Vectorcall(compiled_fn, vectorcall_args, 1, nullptr));
  if (!all_outs) {
    raise_python_error();
  }

  THPObjectPtr result(PyTuple_New(2));
  if (!result) {
    raise_python_error();
  }
  PyTuple_SET_ITEM(result.get(), 0, orig_inputs.release());
  PyTuple_SET_ITEM(result.get(), 1, all_outs.release());
  return result.release();
  END_HANDLE_TH_ERRORS
}

static PyMethodDef aot_autograd_methods[] = {
    {"_aot_autograd_call_compiled_fn",
     reinterpret_cast<PyCFunction>(
         reinterpret_cast<void (*)()>(THPModule_aot_autograd_call_compiled_fn)),
     METH_FASTCALL,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

} // namespace

bool InitializeAOTAutogradHelpers(PyObject* module) {
  CompiledFnCallSpecType.tp_new = CompiledFnCallSpec_new;
  CompiledFnCallSpecType.tp_methods = CompiledFnCallSpec_methods;

  if (PyModule_AddType(module, &CompiledFnCallSpecType) < 0) {
    return false;
  }
  if (PyModule_AddFunctions(module, aot_autograd_methods) < 0) {
    return false;
  }
  return true;
}

// NOLINTEND(hicpp-exception-baseclass)

} // namespace torch::functorch::impl
