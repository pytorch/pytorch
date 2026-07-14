#include <torch/csrc/autograd/python_aot_autograd.h>

#include <c10/util/irange.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_raii.h>

namespace torch::autograd {

// NOLINTBEGIN(hicpp-exception-baseclass)

namespace {

int64_t read_tuple_i64(PyObject* tuple, Py_ssize_t idx) {
  PyObject* item = PyTuple_GET_ITEM(tuple, idx);
  auto result = PyLong_AsLongLong(item);
  if (result == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  return result;
}

Range read_range(PyObject* range) {
  TORCH_INTERNAL_ASSERT(PyTuple_CheckExact(range));
  TORCH_INTERNAL_ASSERT(PyTuple_GET_SIZE(range) == 2);
  Range result{read_tuple_i64(range, 0), read_tuple_i64(range, 1)};
  TORCH_INTERNAL_ASSERT(result.start >= 0 && result.count >= 0);
  return result;
}

void AOTAutogradSavePlan_dealloc(AOTAutogradSavePlan* self) {
  for (auto* dims : self->dynamic_dims) {
    Py_XDECREF(dims);
  }
  self->saved_tensor_is_graph_input.~vector<uint8_t>();
  self->dynamic_dims.~vector<PyObject*>();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyTypeObject AOTAutogradSavePlanType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._AOTAutogradSavePlan", /* tp_name */
    sizeof(AOTAutogradSavePlan), /* tp_basicsize */
    0, /* tp_itemsize */
    reinterpret_cast<destructor>(AOTAutogradSavePlan_dealloc), /* tp_dealloc */
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

AOTAutogradSavePlan* unpack_save_plan(PyObject* obj) {
  TORCH_INTERNAL_ASSERT(Py_IS_TYPE(obj, &AOTAutogradSavePlanType));
  return reinterpret_cast<AOTAutogradSavePlan*>(obj);
}

PyObject* detach_if_intermediate_view(PyObject* item, bool is_graph_input) {
  // Saved views that are graph intermediates must be detached before storing on
  // ctx, otherwise ctx can keep the intermediate view and its grad_fn alive in
  // a reference cycle. Graph inputs are already held by the autograd
  // invocation, so retaining the original input view is safe and avoids the
  // detach cost. See pytorch/pytorch#94990 and graph_compile.py's
  // saved_tensor_is_graph_input note.
  if (is_graph_input) {
    Py_INCREF(item);
    return item;
  }

  const auto& tensor = THPVariable_Unpack(item);
  if (tensor.is_view()) {
    return THPVariable_Wrap(tensor.detach());
  }

  Py_INCREF(item);
  return item;
}

// This is for cross-graph-break dynamic shape propagation
void maybe_mark_dynamic_saved_tensor(
    PyObject* tensor,
    const AOTAutogradSavePlan* plan,
    int64_t saved_tensor_idx) {
  auto* dims = plan->dynamic_dims[saved_tensor_idx];
  if (!dims) {
    return;
  }

  static PyObject* attr = nullptr;
  if (!attr) {
    attr = PyUnicode_InternFromString("_dynamo_propagated_dynamic_indices");
    if (!attr) {
      throw python_error();
    }
  }

  PyObject* existing = nullptr;
  int found = PyObject_GetOptionalAttr(tensor, attr, &existing);
  if (found < 0) {
    throw python_error();
  }
  if (found == 0) {
    if (PyObject_SetAttr(tensor, attr, dims) < 0) {
      throw python_error();
    }
    return;
  }

  THPObjectPtr existing_ptr(existing);
  THPObjectPtr updated(PyNumber_InPlaceOr(existing_ptr.get(), dims));
  if (!updated) {
    throw python_error();
  }
  if (PyObject_SetAttr(tensor, attr, updated.get()) < 0) {
    throw python_error();
  }
}

void read_saved_tensor_is_graph_input(
    AOTAutogradSavePlan* plan,
    PyObject* flags,
    int64_t n) {
  TORCH_INTERNAL_ASSERT(PyTuple_GET_SIZE(flags) == n);
  plan->saved_tensor_is_graph_input.reserve(n);
  for (const auto i : c10::irange(n)) {
    PyObject* item = PyTuple_GET_ITEM(flags, static_cast<Py_ssize_t>(i));
    auto is_true = PyObject_IsTrue(item);
    if (is_true < 0) {
      throw python_error();
    }
    plan->saved_tensor_is_graph_input.push_back(is_true != 0);
  }
}

void read_dynamic_saved_tensor_specs(
    AOTAutogradSavePlan* plan,
    PyObject* specs,
    int64_t n) {
  TORCH_INTERNAL_ASSERT(PyTuple_CheckExact(specs));
  plan->dynamic_dims.resize(n, nullptr);

  for (const auto i : c10::irange(PyTuple_GET_SIZE(specs))) {
    PyObject* spec = PyTuple_GET_ITEM(specs, i);
    TORCH_INTERNAL_ASSERT(
        PyTuple_CheckExact(spec) && PyTuple_GET_SIZE(spec) == 2);
    auto saved_idx = read_tuple_i64(spec, 0);
    PyObject* dims_tuple = PyTuple_GET_ITEM(spec, 1);
    TORCH_INTERNAL_ASSERT(saved_idx >= 0 && saved_idx < n);
    TORCH_INTERNAL_ASSERT(plan->dynamic_dims[saved_idx] == nullptr);

    TORCH_INTERNAL_ASSERT(PyTuple_CheckExact(dims_tuple));
    THPObjectPtr dims_set(PySet_New(nullptr));
    if (!dims_set) {
      throw python_error();
    }
    for (const auto dim_idx : c10::irange(PyTuple_GET_SIZE(dims_tuple))) {
      THPObjectPtr dim(
          PyLong_FromLongLong(read_tuple_i64(dims_tuple, dim_idx)));
      if (!dim) {
        throw python_error();
      }
      if (PySet_Add(dims_set.get(), dim.get()) < 0) {
        throw python_error();
      }
    }
    THPObjectPtr dims(PyFrozenSet_New(dims_set.get()));
    if (!dims) {
      throw python_error();
    }
    plan->dynamic_dims[saved_idx] = dims.release();
  }
}

PyObject* AOTAutogradSavePlan_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!kwargs || PyDict_GET_SIZE(kwargs) == 0);

  PyObject* vc_obj = nullptr;
  PyObject* no_vc_obj = nullptr;
  PyObject* opaque_obj = nullptr;
  PyObject* symint_obj = nullptr;
  PyObject* flags = nullptr;
  PyObject* specs = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "OOOOOO:_AOTAutogradSavePlan",
          &vc_obj,
          &no_vc_obj,
          &opaque_obj,
          &symint_obj,
          &flags,
          &specs)) {
    throw python_error();
  }

  auto vc = read_range(vc_obj);
  auto no_vc = read_range(no_vc_obj);
  auto opaque = read_range(opaque_obj);
  auto symint = read_range(symint_obj);
  TORCH_INTERNAL_ASSERT(PyTuple_CheckExact(flags));
  const auto n = vc.count + no_vc.count;

  auto* plan = reinterpret_cast<AOTAutogradSavePlan*>(type->tp_alloc(type, 0));
  if (!plan) {
    throw python_error();
  }
  plan->tensors_saved_with_vc_check_range = vc;
  plan->tensors_saved_no_vc_check_range = no_vc;
  plan->opaque_object_outs_range = opaque;
  plan->symint_outs_range = symint;
  new (&plan->saved_tensor_is_graph_input) std::vector<uint8_t>();
  new (&plan->dynamic_dims) std::vector<PyObject*>();

  try {
    read_saved_tensor_is_graph_input(plan, flags, n);
    read_dynamic_saved_tensor_specs(plan, specs, n);
  } catch (...) {
    Py_DECREF(reinterpret_cast<PyObject*>(plan));
    throw;
  }

  return reinterpret_cast<PyObject*>(plan);
  END_HANDLE_TH_ERRORS
}

void save_tensors_saved_with_vc_check(
    THPFunction* fn,
    PyObject* fw_outs,
    const AOTAutogradSavePlan* plan) {
  const auto& vc = plan->tensors_saved_with_vc_check_range;
  const auto n = vc.count;

  THPObjectPtr to_save(PyTuple_New(static_cast<Py_ssize_t>(n)));
  if (!to_save) {
    throw python_error();
  }
  for (const auto i : c10::irange(n)) {
    auto idx = vc.start + i;
    auto is_graph_input = plan->saved_tensor_is_graph_input[i] != 0;
    PyObject* item = PyList_GET_ITEM(fw_outs, static_cast<Py_ssize_t>(idx));
    THPObjectPtr saved(detach_if_intermediate_view(item, is_graph_input));
    maybe_mark_dynamic_saved_tensor(saved.get(), plan, i);
    PyTuple_SET_ITEM(
        to_save.get(), static_cast<Py_ssize_t>(i), saved.release());
  }
  Py_CLEAR(fn->to_save);
  fn->to_save = to_save.release();
}

void save_tensors_saved_no_vc_check(
    PyObject* ctx,
    PyObject* fw_outs,
    const AOTAutogradSavePlan* plan) {
  const auto& vc = plan->tensors_saved_with_vc_check_range;
  const auto& no_vc = plan->tensors_saved_no_vc_check_range;
  const auto n_vc = vc.count;
  const auto n = no_vc.count;

  THPObjectPtr no_vc_list(PyList_New(static_cast<Py_ssize_t>(n)));
  if (!no_vc_list) {
    throw python_error();
  }
  for (const auto i : c10::irange(n)) {
    auto saved_tensor_idx = n_vc + i;
    auto idx = no_vc.start + i;
    auto is_graph_input =
        plan->saved_tensor_is_graph_input[saved_tensor_idx] != 0;
    PyObject* item = PyList_GET_ITEM(fw_outs, static_cast<Py_ssize_t>(idx));
    THPObjectPtr saved(detach_if_intermediate_view(item, is_graph_input));
    maybe_mark_dynamic_saved_tensor(saved.get(), plan, saved_tensor_idx);
    PyList_SET_ITEM(
        no_vc_list.get(), static_cast<Py_ssize_t>(i), saved.release());
  }
  if (PyObject_SetAttrString(ctx, "_tensors_no_vc_check", no_vc_list.get()) <
      0) {
    throw python_error();
  }
}

void save_symints(
    PyObject* ctx,
    PyObject* fw_outs,
    const AOTAutogradSavePlan* plan) {
  const auto& r = plan->symint_outs_range;
  THPObjectPtr symints(PyList_New(static_cast<Py_ssize_t>(r.count)));
  if (!symints) {
    throw python_error();
  }
  for (const auto i : c10::irange(r.count)) {
    auto idx = r.start + i;
    PyObject* item = PyList_GET_ITEM(fw_outs, static_cast<Py_ssize_t>(idx));
    Py_INCREF(item);
    PyList_SET_ITEM(symints.get(), static_cast<Py_ssize_t>(i), item);
  }
  if (PyObject_SetAttrString(ctx, "symints", symints.get()) < 0) {
    throw python_error();
  }
}

void save_opaque_objects(
    PyObject* ctx,
    PyObject* fw_outs,
    const AOTAutogradSavePlan* plan) {
  const auto& r = plan->opaque_object_outs_range;
  THPObjectPtr opaque_objects(PyList_New(static_cast<Py_ssize_t>(r.count)));
  if (!opaque_objects) {
    throw python_error();
  }
  for (const auto i : c10::irange(r.count)) {
    auto idx = r.start + i;
    PyObject* item = PyList_GET_ITEM(fw_outs, static_cast<Py_ssize_t>(idx));
    Py_INCREF(item);
    PyList_SET_ITEM(opaque_objects.get(), static_cast<Py_ssize_t>(i), item);
  }

  if (PyObject_SetAttrString(ctx, "opaque_objects", opaque_objects.get()) < 0) {
    throw python_error();
  }
}

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
PyTypeObject* getAOTAutogradSavePlanType() {
  AOTAutogradSavePlanType.tp_new = AOTAutogradSavePlan_new;
  return &AOTAutogradSavePlanType;
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
PyObject* THPModule_aot_autograd_save_from_forward(
    PyObject* _unused,
    PyObject* const* args,
    Py_ssize_t nargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(nargs == 3);
  PyObject* ctx = args[0];
  PyObject* fw_outs = args[1];
  PyObject* plan_obj = args[2];
  TORCH_INTERNAL_ASSERT(THPFunction_Check(ctx));
  TORCH_INTERNAL_ASSERT(PyList_CheckExact(fw_outs));
  auto* fn = reinterpret_cast<THPFunction*>(ctx);
  const auto* plan = unpack_save_plan(plan_obj);

  save_tensors_saved_with_vc_check(fn, fw_outs, plan);
  save_tensors_saved_no_vc_check(ctx, fw_outs, plan);
  save_symints(ctx, fw_outs, plan);
  save_opaque_objects(ctx, fw_outs, plan);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTEND(hicpp-exception-baseclass)

} // namespace torch::autograd
