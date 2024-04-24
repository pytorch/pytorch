#include <torch/csrc/autograd/python_function.h>

#include <ATen/ATen.h>
#include <ATen/SequenceNumber.h>
#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <structmember.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/FuncTorchTLS.h>
#include <ATen/functorch/DynamicLayer.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/dynamo/compiled_autograd.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_dtypes.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace torch;
using namespace torch::autograd;
using at::Tensor;

PyObject* THPFunctionClass = nullptr;
PyObject* THPGradientEdgeClass = nullptr;

#define THPFunction_assert(condition, ...) \
  if (!(condition)) {                      \
    THPUtils_setError(__VA_ARGS__);        \
    throw python_error();                  \
  }

// Anonymous namespace for helpful functions used in this file
namespace {

// TODO: We shouldn't need to call this function because the engine
// can already persist the errors for us. This still seems to be
// needed for the DistEngine however.
//
// python test/distributed/rpc/test_tensorpipe_agent.py -k
// test_backward_autograd_engine_error
//
// See Note [ Persisting PyErr state across autograd engine threads ]
void throw_python_error() {
  python_error err;
  err.persist();
  throw std::move(err);
}

static PyObject* unpack_saved_variables(
    THPFunction* self,
    const std::function<PyObject*(const Variable&)>& unpack_fn) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(!self->has_freed_buffers, ERR_BACKWARD_TWICE);
  auto& saved_variables = self->saved_variables;
  if (saved_variables.empty())
    return PyTuple_New(0);

  auto num_saved = saved_variables.size();
  THPObjectPtr saved(PyTuple_New(static_cast<Py_ssize_t>(num_saved)));
  if (!saved)
    return nullptr;
  auto saved_for = self->cdata.lock();
  // This is really a true assert, because we've already tested for the
  // self->has_freed_buffers case at the beginning of this function:
  // buffers are freed when PyNode dies; if the buffers are not freed,
  // PyNode must be live.  (Note that the buffers could be freed
  // even though the PyNode is live, but that doesn't matter here
  // because we will never hit this line of code if the buffers are freed--
  // and in any case saved_for will be non-NULL.)
  TORCH_INTERNAL_ASSERT(saved_for);
  for (const auto i : c10::irange(num_saved)) {
    auto unpacked_var = saved_variables[i].unpack(saved_for);
    THPObjectPtr value;
    if (!unpacked_var.defined()) {
      Py_INCREF(Py_None);
      value = Py_None;
    } else {
      value = unpack_fn(unpacked_var);
    }
    PyTuple_SET_ITEM(saved.get(), i, value.release());
  }
  return saved.release();
  END_HANDLE_TH_ERRORS
}

PyObject* to_py_size(const std::vector<c10::SymInt>& size) {
  c10::SymIntArrayRef sym_sizes(size);

  auto ret = THPObjectPtr(THPSizeType.tp_alloc(
      &THPSizeType, static_cast<Py_ssize_t>(sym_sizes.size())));
  if (!ret)
    throw python_error();

  for (auto i : c10::irange(sym_sizes.size())) {
    auto symint = sym_sizes[i];
    if (auto maybe_int = symint.maybe_as_int(); maybe_int.has_value()) {
      PyTuple_SET_ITEM(ret.get(), i, THPUtils_packInt64(*maybe_int));
    } else {
      auto py_symint = py::cast(symint).release().ptr();
      PyTuple_SET_ITEM(ret.get(), i, py_symint);
    }
  }
  return ret.release();
}

} // namespace

namespace torch::autograd {

// NOTE: this function is written in a way that assumes it's only called for
// backward; it's used by engine.cpp.  This is responsible for forwarding a call
// from C++'s Node::apply to a Python method "apply".
auto PyNode::apply(variable_list&& inputs) -> variable_list {
  pybind11::gil_scoped_acquire gil;
  at::OptionalDeviceGuard _device_guard;
  THPFunction* py_fn = (THPFunction*)obj;

  // Massage a C++ variable_list into a Python arguments tuple
  THPObjectPtr pyInputs(to_py_args(inputs, &_device_guard));

  THPObjectPtr apply_fn(PyObject_GetAttrString(obj, "apply"));
  if (!apply_fn)
    throw_python_error();
  THPObjectPtr r(PyObject_CallObject(apply_fn, pyInputs.get()));
  if (!r)
    throw_python_error();
  ensure_tuple(r);

  auto& is_variable_input = py_fn->is_variable_input;
  auto num_outputs = PyTuple_GET_SIZE(r.get());
  auto num_forward_inputs = static_cast<Py_ssize_t>(is_variable_input.size());
  // Returning too many results is ok, but only as long as they're all None.
  // Truncate the result tuple in that case.
  if (num_outputs > num_forward_inputs) {
    bool all_none = true;
    for (const auto i : c10::irange(num_forward_inputs, num_outputs)) {
      all_none &= PyTuple_GET_ITEM(r.get(), i) == Py_None;
    }
    if (all_none) {
      num_outputs = num_forward_inputs;
      r = PyTuple_GetSlice(r.get(), 0, num_forward_inputs);
      if (!r)
        throw_python_error();
    }
  }

  // Now the number of gradients should match
  if (num_outputs != num_forward_inputs) {
    std::string msg("function ");
    msg += name() + " returned an incorrect number of gradients (expected ";
    msg += std::to_string(num_forward_inputs) + ", got ";
    msg += std::to_string(num_outputs) + ")";
    throw std::runtime_error(msg);
  }

  // Massage the Python results tuple back into a C++ variable_list
  return to_variable_list(r.get(), is_variable_input);
}

auto PyNode::compiled_apply(
    variable_list&& inputs,
    std::optional<PyObject*> compiler) -> variable_list {
  pybind11::gil_scoped_acquire gil;
  at::OptionalDeviceGuard _device_guard;
  THPFunction* py_fn = (THPFunction*)obj;

  // Massage a C++ variable_list into a Python arguments tuple
  THPObjectPtr pyInputs(to_py_args(inputs, &_device_guard));

  const auto& is_variable_input = py_fn->is_variable_input;
  const auto& input_infos = py_fn->input_info;
  // input_info only contains info from variable inputs and should be a subset
  TORCH_INTERNAL_ASSERT(is_variable_input.size() >= input_infos.size());

  // The gradients returned in the backwards need to match the number of inputs
  // to the forward, and their metadata, so we pass the fwdInputs
  THPObjectPtr fwdInputMetadatas(
      PyTuple_New(static_cast<Py_ssize_t>(is_variable_input.size())));
  if (!fwdInputMetadatas)
    throw python_error();

  int offset = 0;
  for (const auto i : c10::irange(is_variable_input.size())) {
    if (!is_variable_input[i]) {
      // input at i is not a variable, skip index
      PyTuple_SET_ITEM(fwdInputMetadatas.get(), i, Py_None);
      offset++;
      continue;
    }

    const auto& input_info = input_infos[i - offset];

    PyObject* device(THPDevice_New(input_info.device));
    if (!device)
      throw_python_error();
    // Metadata is a tuple of 4 elements: (layout, device, dtype, size)
    PyObject* fwdInputMetadata = PyTuple_Pack(
        4,
        autograd::utils::wrap(input_info.layout),
        device,
        autograd::utils::wrap(input_info.scalar_type),
        to_py_size(input_info.size));
    if (!fwdInputMetadata)
      throw python_error();

    PyTuple_SET_ITEM(fwdInputMetadatas.get(), i, fwdInputMetadata);
  }
  THPObjectPtr saved_tensors(unpack_saved_variables(
      py_fn, [](const Variable& var) { return THPVariable_Wrap(var); }));
  TORCH_INTERNAL_ASSERT(
      _backward_idx.has_value(),
      "indices should already be set by compiled_args, called before apply_with_saved");
  TORCH_INTERNAL_ASSERT(!_backward_state_idx.has_value());
  THPObjectPtr r(PyObject_CallMethod(
      *compiler,
      "proxy_call_backward",
      "OOOi",
      pyInputs.get(),
      fwdInputMetadatas.get(),
      saved_tensors.get(),
      *_backward_idx));

  if (!r)
    throw_python_error();
  ensure_tuple(r);

  // Massage the Python results tuple back into a C++ variable_list
  return to_variable_list(r.get(), is_variable_input);
}

auto PyNode::is_traceable() -> bool {
  pybind11::gil_scoped_acquire gil;
  THPObjectPtr forward_class{PyObject_GetAttrString(obj, "_forward_cls")};
  if (!forward_class)
    throw_python_error();
  THPObjectPtr traceable_py_bool{
      PyObject_GetAttrString(forward_class, "is_traceable")};
  if (!traceable_py_bool)
    throw_python_error();
  return traceable_py_bool == Py_True;
}

auto PyNode::release_variables() -> void {
  // This function is called as part of the Node destructor!
  // Since this object might be kept alive by C++, it is possible
  // that the python interpreter is already dead here. In that case
  // we just leak the saved objects.
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    auto f = (THPFunction*)obj;
    f->saved_variables.clear();
    f->has_freed_buffers = 1;
  }
}

auto PyNode::name() const -> std::string {
  pybind11::gil_scoped_acquire gil;
  auto f = (THPFunction*)obj;
  auto name = std::string(Py_TYPE(f)->tp_name);
  return name;
}

auto PyNode::compiled_autograd_should_lift() const -> bool {
  pybind11::gil_scoped_acquire gil;
  static PyObject* attr_name =
      PyUnicode_InternFromString("_compiled_autograd_should_lift");
  THPObjectPtr should_lift(PyObject_GetAttr(obj, attr_name));
  return PyObject_IsTrue(should_lift.get()) == 1;
}

void PyNode::compiled_args(CompiledNodeArgs& args) {
  static PyObject* method_name =
      PyUnicode_InternFromString("_compiled_autograd_key");
  THPObjectPtr pykey(PyObject_CallMethodNoArgs(obj, method_name));
  if (!pykey)
    throw_python_error();
  TORCH_CHECK(
      PyTuple_CheckExact(pykey.get()),
      "_compiled_autograd_key should return tuple of ints");
  auto size = PyTuple_GET_SIZE(pykey.get());
  TORCH_INTERNAL_ASSERT(size > 0);
  // first value is unique id managed by AUTOGRAD_FUNCTION_COUNTER
  auto key = PyLong_AsSsize_t(PyTuple_GET_ITEM(pykey.get(), 0));
  if (C10_UNLIKELY(key < 0)) {
    TORCH_CHECK(PyErr_Occurred(), "key must be positive");
    throw_python_error();
  }
  args.collect_size(static_cast<size_t>(key));
  args.collect_size(static_cast<size_t>(size));

  auto f = (THPFunction*)obj;
  f->compiled_autograd_symints.clear();
  f->compiled_autograd_symints.reserve(size - 1);
  for (const auto i : c10::irange(1, size)) {
    auto val = PyLong_AsSsize_t(PyTuple_GET_ITEM(pykey.get(), i));
    if (C10_UNLIKELY(val == -1 && PyErr_Occurred()))
      throw_python_error();
    f->compiled_autograd_symints.emplace_back(val);
  }

  // AotAutograd symints are all dynamic
  auto prior =
      args.set_default_dyn_type(torch::dynamo::autograd::SizeInput::DYNAMIC);
  args.collect(f->compiled_autograd_symints);
  args.set_default_dyn_type(prior);

  args.collect(f->saved_variables);
  args.collect(f->materialize_grads);
  args.collect(f->is_variable_input);
  args.collect(f->needs_input_grad);
  args.collect(f->materialize_non_diff_grads);
  args.collect(f->output_info);
  args.collect(f->input_info);

  static PyObject* forward_cls_name =
      PyUnicode_InternFromString("_forward_cls");
  THPObjectPtr forward_cls(PyObject_GetAttr(obj, forward_cls_name));
  static PyObject* backward_name = PyUnicode_InternFromString("backward");
  PyObject* backward(PyObject_GetAttr(forward_cls.get(), backward_name));
  _backward_idx =
      args.add_backward(c10::SafePyObject(backward, getPyInterpreter()));

  PyObject* bw_state = f->compiled_autograd_backward_state;
  if (args.cond(bw_state != nullptr)) {
    Py_INCREF(bw_state);
    _backward_state_idx = args.add_backward_state(
        c10::SafePyObject(bw_state, getPyInterpreter()));
  }
}

variable_list PyNode::apply_with_saved(
    const variable_list& inputs,
    SwapSavedVariables& saved) {
  auto f = (THPFunction*)obj;
  TORCH_INTERNAL_ASSERT(!f->compiled_autograd_tracing);
  saved.before(f->compiled_autograd_symints);
  saved.before(f->saved_variables);
  saved.before(f->needs_input_grad);
  saved.before(f->materialize_non_diff_grads);
  saved.before(f->output_info);
  saved.before(f->input_info);
  f->compiled_autograd_tracing = true;
  variable_list result;
  if (!compiled_autograd_should_lift()) {
    if (_backward_state_idx.has_value()) {
      PyObject* r = PyObject_CallMethod(
          saved.get_py_compiler(),
          "bind_backward_state",
          "i",
          *_backward_state_idx);
      if (r == nullptr) {
        throw python_error();
      }
      THPObjectPtr prior(f->compiled_autograd_backward_state);
      f->compiled_autograd_backward_state = r;
      result = apply(variable_list(inputs));
      Py_CLEAR(f->compiled_autograd_backward_state);
      f->compiled_autograd_backward_state = prior.release();
    } else {
      result = apply(variable_list(inputs));
    }
  } else {
    result = compiled_apply(variable_list(inputs), saved.get_py_compiler());
  }
  f->compiled_autograd_tracing = false;
  saved.after(f->compiled_autograd_symints);
  saved.after(f->saved_variables);
  saved.after(f->needs_input_grad);
  saved.after(f->materialize_non_diff_grads);
  saved.after(f->output_info);
  saved.after(f->input_info);
  return result;
}

PyObject* PyNode::to_py_args(
    const variable_list& inputs,
    at::OptionalDeviceGuard* device_guard) {
  THPFunction* py_fn = (THPFunction*)obj;

  auto zeros_without_gil = [](const VariableInfo& variable,
                              at::OptionalDeviceGuard& dg) {
    pybind11::gil_scoped_release gil;
    return variable.zeros(dg);
  };

  auto num_inputs = inputs.size();
  PyObject* pyInputs = PyTuple_New(static_cast<Py_ssize_t>(num_inputs));
  if (!pyInputs)
    throw_python_error();
  auto& output_info = py_fn->output_info;
  for (const auto i : c10::irange(num_inputs)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyObject* input;
    if (inputs[i].defined() || !py_fn->materialize_grads ||
        (input_metadata(i).was_default_constructed() &&
         !py_fn->materialize_non_diff_grads)) {
      input = THPVariable_Wrap(inputs[i]);
    } else {
      input =
          THPVariable_Wrap(zeros_without_gil(output_info[i], *device_guard));
    }
    if (!input)
      throw_python_error();
    PyTuple_SET_ITEM(pyInputs, i, input);
  }

  return pyInputs;
}

variable_list PyNode::to_variable_list(
    const PyObject* outputs,
    const std::vector<bool>& is_variable_input) {
  auto num_outputs = PyTuple_GET_SIZE(outputs);
  variable_list results;
  results.reserve(num_outputs);
  for (int i = 0; i != num_outputs; ++i) {
    PyObject* output = PyTuple_GET_ITEM(outputs, i);
    bool was_variable = is_variable_input[i];
    if (!was_variable) {
      if (output != Py_None) {
        std::string msg("function ");
        msg += name() + " returned a gradient different than None at position ";
        msg += std::to_string(i + 1) +
            ", but the corresponding forward input was not a Variable";
        throw std::runtime_error(msg);
      }
      continue;
    }
    if (output == Py_None) {
      results.emplace_back();
    } else {
      if (!THPVariable_Check(output)) {
        std::string msg("expected Variable or None (got ");
        msg += THPUtils_typename(output);
        msg += ")";
        throw std::runtime_error(msg);
      }
      results.emplace_back(THPVariable_Unpack(output));
    }
  }

  return results;
}

} // namespace torch::autograd

// Traverse and clear are required for supporting Python's GC cycle handling.
static int THPFunction_traverse(THPFunction* self, visitproc visit, void* arg) {
  // NB: We should not traverse PyObbject stored on PyNode, since we only hold
  // as weak reference to the PyNode.
  Py_VISIT(self->to_save);
  Py_VISIT(self->non_differentiable);
  Py_VISIT(self->dirty_tensors);
  Py_VISIT(self->compiled_autograd_backward_state);
  Py_VISIT(self->saved_for_forward);
  return 0;
}

static int THPFunction_clear(THPFunction* self) {
  // Note that the cdata might not be expired yet in the case where this
  // object is part of a cycle and the GC happens to tp_clear this PyObject
  // before the other ones that trigger the de-allocation of the cdata

  Py_CLEAR(self->needs_input_grad);

  Py_CLEAR(self->to_save);
  Py_CLEAR(self->non_differentiable);
  Py_CLEAR(self->dirty_tensors);
  Py_CLEAR(self->compiled_autograd_backward_state);
  Py_CLEAR(self->saved_for_forward);

  self->output_info.clear();
  self->input_info.clear();
  self->saved_variables.clear();
  self->is_variable_input.clear();

  return 0;
}

static void THPFunction_dealloc(THPFunction* self) {
  // Why is this guaranteed to be true?  Suppose that self->cdata is non-null
  // (otherwise the condition is trivially true).  Then there is a PyNode
  // which contains an owning reference to this object.  But we are only
  // allowed to clear if all owning references are gone!  Contradiction.
  //
  // However, note that THPFunction_clear is typically called in the shared_ptr
  // destructor of PyNode; in that case, per
  // https://cplusplus.github.io/LWG/lwg-active.html#2751 it's not currently
  // specified in the standard that this is guaranteed.  If you see this
  // assert triggering in the wild, feel free to comment it out.  They're
  // likely to standardize that you ARE guaranteed to see the weak pointers
  // as expired in the destructor in the future, so we'll keep this for now.
  TORCH_INTERNAL_ASSERT(self->cdata.expired());

  PyObject_GC_UnTrack(self);
  THPFunction_clear(self);
  self->cdata.~weak_ptr<PyNode>();
  self->output_info.~vector();
  self->input_info.~vector();
  self->saved_variables.~vector();
  self->is_variable_input.~vector();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* THPFunction_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (!obj)
    return nullptr;
  // Python zero-initializes the object memory, so there's no need to initialize
  // most fields
  THPFunction* self = (THPFunction*)obj;
  // Setup the PyNode later; we can't keep it live here
  new (&self->cdata) std::weak_ptr<PyNode>();
  new (&self->output_info) std::vector<VariableInfo>();
  new (&self->input_info) std::vector<VariableInfo>();
  new (&self->saved_variables) std::vector<SavedVariable>();
  new (&self->is_variable_input) std::vector<bool>();
  self->materialize_grads = true;
  self->materialize_non_diff_grads = true;
  self->compiled_autograd_tracing = false;
  return obj;
}

////////////////////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////////////////////

// Bump the counters of all recorded dirty input tensors, adding each of them
// into dirty_inputs.  Also does some sanity checking.
static std::unordered_set<at::TensorImpl*> _mark_dirty(THPFunction* self) {
  // Increase versions of modified tensors
  std::unordered_set<at::TensorImpl*> dirty_inputs;
  if (!self->dirty_tensors)
    return dirty_inputs;

  THPFunction_assert(
      PyTuple_Check(self->dirty_tensors),
      "autograd "
      "internal error: dirty_tensors attribute is expected to be a tuple "
      "but is ",
      THPUtils_typename(self->dirty_tensors));
  Py_ssize_t num_dirty = PyTuple_GET_SIZE(self->dirty_tensors);
  dirty_inputs.reserve(num_dirty);
  for (const auto i : c10::irange(num_dirty)) {
    PyObject* obj = PyTuple_GET_ITEM(self->dirty_tensors, i);
    THPFunction_assert(
        THPVariable_Check(obj),
        "mark_dirty can "
        "only accept variables, but argument ",
        i,
        " is of type ",
        THPUtils_typename(obj));

    const auto& tensor = THPVariable_Unpack(obj);
    dirty_inputs.insert(tensor.unsafeGetTensorImpl());
    torch::autograd::impl::bump_version(tensor);
  }
  // We're not going to ever need this so let's remove references now
  Py_CLEAR(self->dirty_tensors);
  return dirty_inputs;
}

static std::unordered_set<at::TensorImpl*> _parse_non_differentiable(
    THPFunction* self);

// Given a Python tuple of raw output tensors (raw_output), set each of
// the corresponding entries in a different Python tuple (outputs) with
// these tensors wrapped with variables.  We save the gradient function (self)
// to the variable if the output requires grad.
//
// There is a considerable amount of complexity to handle if the operation
// that produced these output tensors is inplace.  A mapping of *input*
// tensors to variables (t2var) is used to test if this occurred, and
// the set of dirty tensors (dirty_inputs) is used to figure out what to
// do in this case.  After this method is run, t2var is extended with
// mappings for output tensors as well.
static void _wrap_outputs(
    const std::shared_ptr<PyNode>& cdata,
    THPFunction* self,
    const variable_list& input_vars,
    PyObject* raw_output,
    PyObject* outputs,
    bool is_executable,
    const std::unordered_set<at::TensorImpl*>& to_save_if_setup_context) {
  auto cdata_if_executable = is_executable ? cdata : nullptr;
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(raw_output);
  if (is_executable) {
    self->output_info.clear();
    self->output_info.reserve(num_outputs);
  }

  auto non_differentiable = _parse_non_differentiable(self);
  auto dirty_inputs = _mark_dirty(self);

  std::vector<c10::optional<Variable>> raw_output_vars;
  raw_output_vars.reserve(num_outputs);
  for (const auto i : c10::irange(num_outputs)) {
    PyObject* obj = PyTuple_GET_ITEM(raw_output, i);
    // Only process tensors as outputs for autograd purposes.
    if (THPVariable_Check(obj)) {
      raw_output_vars.emplace_back(THPVariable_Unpack(obj));
    } else {
      raw_output_vars.emplace_back();
    }
  }

  _jvp_fn_t jvp_user_function = [self](
                                    variable_list inputs,
                                    variable_list grad_inputs) {
    pybind11::gil_scoped_acquire gil;

    // Massage a C++ variable_list into a Python arguments tuple
    // Making sure to introduce the proper None for non-Tensor inputs
    auto num_inputs = self->is_variable_input.size();
    THPObjectPtr pyInputs(PyTuple_New(static_cast<Py_ssize_t>(num_inputs)));
    if (!pyInputs)
      throw_python_error();
    int64_t variable_idx = 0;
    for (const auto i : c10::irange(num_inputs)) {
      PyObject* input = nullptr;
      if (self->is_variable_input[i]) {
        if (grad_inputs[variable_idx].defined() || !self->materialize_grads ||
            !isDifferentiableType(inputs[variable_idx].scalar_type())) {
          input = THPVariable_Wrap(grad_inputs[variable_idx]);
        } else {
          input = THPVariable_Wrap(at::zeros_like(inputs[variable_idx]));
        }
        if (!input) {
          throw_python_error();
        }
        variable_idx++;
      } else {
        Py_INCREF(Py_None);
        input = Py_None;
      }
      PyTuple_SET_ITEM(pyInputs.get(), i, input);
    }

    THPObjectPtr apply_jvp_fn(
        PyObject_GetAttrString((PyObject*)self, "apply_jvp"));
    if (!apply_jvp_fn)
      throw_python_error();
    THPObjectPtr r(PyObject_CallObject(apply_jvp_fn, pyInputs.get()));
    if (!r)
      throw_python_error();
    ensure_tuple(r);

    // Massage the Python results tuple back into a C++ variable_list
    // Don't do any check on the number of results here as
    // it is handled by the caller
    const int num_outputs = PyTuple_GET_SIZE(r.get());
    variable_list results;
    results.reserve(num_outputs);
    for (const auto i : c10::irange(num_outputs)) {
      PyObject* output = PyTuple_GET_ITEM(r.get(), i);
      if (output == Py_None) {
        results.emplace_back();
      } else {
        TORCH_CHECK(
            THPVariable_Check(output),
            "expected Variable or None (got ",
            THPUtils_typename(output),
            ") for grad output ",
            i,
            ".")
        results.emplace_back(THPVariable_Unpack(output));
      }
    }

    return results;
  };

  auto view_as_self_fn = [](const at::Tensor& x) -> at::Tensor {
    pybind11::gil_scoped_acquire gil;
    THPObjectPtr py_x(THPVariable_Wrap(x));
    THPObjectPtr py_view_as_method(PyObject_GetAttrString(py_x, "view_as"));
    if (!py_view_as_method)
      throw python_error();
    THPObjectPtr args(PyTuple_Pack(1, py_x.get()));
    if (!args)
      throw python_error();
    THPObjectPtr result(PyObject_CallObject(py_view_as_method, args));
    if (!result)
      throw python_error();
    return THPVariable_Unpack(result);
  };

  // Wrap only the tensor outputs.
  auto wrapped_outputs = _wrap_outputs(
      input_vars,
      non_differentiable,
      dirty_inputs,
      raw_output_vars,
      cdata_if_executable,
      jvp_user_function,
      to_save_if_setup_context,
      view_as_self_fn);

  for (const auto i : c10::irange(num_outputs)) {
    PyObject* obj = PyTuple_GetItem(raw_output, i);
    // Keep the non-tensor outputs as is.
    if (!THPVariable_Check(obj)) {
      if (is_executable) {
        self->output_info.emplace_back();
      }
      Py_INCREF(obj);
      PyTuple_SetItem(outputs, i, obj);
    } else {
      if (is_executable) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        self->output_info.emplace_back(*wrapped_outputs[i]);
      }
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      PyTuple_SetItem(outputs, i, THPVariable_Wrap(*wrapped_outputs[i]));
    }
  }
}

static void _get_tensors_to_save(
    THPFunction* self,
    std::unordered_set<at::TensorImpl*>& to_save_if_setup_context,
    std::vector<c10::optional<at::Tensor>>& tensors_to_save,
    bool overridden_setup_context,
    bool is_executable) {
  if (self->saved_for_forward && overridden_setup_context) {
    // We look at saved_for_forward here purely for the purpose of populating
    // to_save_if_setup_context, the actual saving is not done here.
    THPFunction_assert(
        PyTuple_Check(self->saved_for_forward),
        "autograd internal "
        "error: saved_for_forward attribute is expected to be a tuple but is ",
        THPUtils_typename(self->saved_for_forward));
    Py_ssize_t num_saved_for_forward =
        PyTuple_GET_SIZE(self->saved_for_forward);
    for (const auto i : c10::irange(num_saved_for_forward)) {
      PyObject* obj = PyTuple_GET_ITEM(self->saved_for_forward, i);
      if (THPVariable_Check(obj)) {
        const auto& tensor = THPVariable_Unpack(obj);
        to_save_if_setup_context.insert(tensor.unsafeGetTensorImpl());
      }
    }
  }
  if (self->to_save) {
    THPFunction_assert(
        PyTuple_Check(self->to_save),
        "autograd internal "
        "error: to_save attribute is expected to be a tuple but is ",
        THPUtils_typename(self->to_save));

    Py_ssize_t num_saved = PyTuple_GET_SIZE(self->to_save);
    for (const auto i : c10::irange(num_saved)) {
      PyObject* obj = PyTuple_GET_ITEM(self->to_save, i);
      if (obj == Py_None) {
        tensors_to_save.emplace_back(c10::nullopt);
        continue;
      } else if (THPVariable_Check(obj)) {
        const auto& tensor = THPVariable_Unpack(obj);
        if (overridden_setup_context) {
          to_save_if_setup_context.insert(tensor.unsafeGetTensorImpl());
        }
        if (is_executable) {
          tensors_to_save.emplace_back(tensor);
        }
      } else {
        if (is_executable) {
          // TODO: We should really just ALWAYS throw an error here, but
          // doing so will break some internal tests. We should fix those.
          throw torch::TypeError(
              "save_for_backward can only save variables, but argument %ld is of "
              "type %s",
              i,
              Py_TYPE(obj)->tp_name);
        }
      }
    }
  }
}
// Save any variables that requested by to_save
static void _save_variables(
    const std::vector<c10::optional<at::Tensor>>& tensors_to_save,
    const std::shared_ptr<PyNode>& cdata_ptr,
    THPFunction* self) {
  if (!self->to_save)
    return;
  size_t num_saved = tensors_to_save.size();
  self->saved_variables.clear();
  self->saved_variables.reserve(num_saved);
  for (const auto& opt_tensor : tensors_to_save) {
    if (!opt_tensor.has_value()) {
      self->saved_variables.emplace_back();
    } else {
      bool is_output = opt_tensor.value().grad_fn().get() == cdata_ptr.get();
      self->saved_variables.emplace_back(opt_tensor.value(), is_output);
    }
  }
  // Free .to_save
  Py_CLEAR(self->to_save);
}

// Mark requires_grad = 0 on non-differentiable variables (as per
// non_differentiable)
static std::unordered_set<at::TensorImpl*> _parse_non_differentiable(
    THPFunction* self) {
  std::unordered_set<at::TensorImpl*> set;
  if (!self->non_differentiable)
    return set;

  THPFunction_assert(
      PyTuple_Check(self->non_differentiable),
      "autograd "
      "internal error: non_differentiable attribute is expected to be a "
      "tuple but is ",
      THPUtils_typename(self->non_differentiable));
  Py_ssize_t num_nondiff = PyTuple_GET_SIZE(self->non_differentiable);
  set.reserve(num_nondiff);
  for (const auto i : c10::irange(num_nondiff)) {
    PyObject* t = PyTuple_GET_ITEM(self->non_differentiable, i);
    THPFunction_assert(
        THPVariable_Check(t),
        "mark_non_differentiable "
        "only accepts variable arguments, but got ",
        THPUtils_typename(t));
    set.insert(THPVariable_Unpack(t).unsafeGetTensorImpl());
  }
  Py_CLEAR(self->non_differentiable);
  return set;
}

struct UnpackedInput {
  THPObjectPtr input_tuple;
  variable_list input_vars;
  // record_function_inputs is for RECORD_FUNCTION only
  std::vector<c10::IValue> record_function_inputs;
};

struct InputFlags {
  bool is_executable = false;
  edge_list next_edges;
  THPObjectPtr needs_input_grad;
  std::vector<bool> is_variable_input;
};

template <bool enforce_variables>
std::pair<UnpackedInput, InputFlags> unpack_input(PyObject* args) {
  UnpackedInput unpacked;
  InputFlags flags;

  auto num_args = PyTuple_GET_SIZE(args);
  unpacked.input_tuple = PyTuple_New(num_args);
  flags.needs_input_grad = PyTuple_New(num_args);
  bool profiler_need_input = torch::autograd::profiler::profilerEnabled() &&
      torch::autograd::profiler::getProfilerConfig().report_input_shapes;

  for (const auto i : c10::irange(num_args)) {
    PyObject* arg = PyTuple_GET_ITEM(args, i);

    bool is_variable = THPVariable_Check(arg);
    flags.is_variable_input.push_back(is_variable);
    if (!is_variable) {
      // TODO: remove this code path once Variable and Tensor are merged in
      // Python
      if (enforce_variables) {
        THPUtils_setError(
            "expected a Tensor argument, but got ", THPUtils_typename(arg));
        throw python_error();
      }
      Py_INCREF(Py_False);
      PyTuple_SET_ITEM(flags.needs_input_grad.get(), i, Py_False);

      if (profiler_need_input) {
        // The following conversion from PyObject to IValue is expensive
        // Only do it if profiler is enabled and needs input shapes
        auto match = torch::jit::tryToInferPrimitiveType(arg);
        if (match.success()) {
          unpacked.record_function_inputs.push_back(
              torch::jit::toIValue(arg, match.type()));
        }
      }
    } else {
      const auto& tensor = THPVariable_Unpack(arg);
      unpacked.input_vars.push_back(tensor);
      PyObject* needs_grad = tensor.requires_grad() ? Py_True : Py_False;
      Py_INCREF(needs_grad);
      PyTuple_SET_ITEM(flags.needs_input_grad.get(), i, needs_grad);
      unpacked.record_function_inputs.emplace_back(tensor);
    }
    Py_INCREF(arg);
    PyTuple_SET_ITEM(unpacked.input_tuple.get(), i, arg);
  }

  flags.is_executable =
      GradMode::is_enabled() && any_variable_requires_grad(unpacked.input_vars);
  flags.next_edges =
      (flags.is_executable ? collect_next_edges(unpacked.input_vars)
                           : edge_list());
  return std::make_pair(std::move(unpacked), std::move(flags));
}

// Given a prim::PythonOp node, _append_subgraph creates a subgraph such that:
// (1) It has the same inputs as the prim::PythonOp node
// (2) The intermediate nodes used in the PythonOp are cloned and stored in the
// subgraph (3) trace_outputs stores the Value* objects, before a new trace
// value is assigned by the prim::PythonOp node and helps to eventually route
// the outputs of the subgraph correctly This newly created subgraph is then
// added to the prim::PythonOp node as a subgraph attribute
static void _append_subgraph(
    torch::jit::Node* node,
    torch::jit::Graph* graph,
    std::vector<torch::jit::Value*> trace_outputs,
    bool unpack_output) {
  using Value = torch::jit::Value;
  node->g_(
      torch::jit::attr::Subgraph,
      std::make_shared<torch::jit::Graph>(graph->current_scope()));
  auto subgraph = node->g(torch::jit::attr::Subgraph);

  std::unordered_map<Value*, Value*> value_map;
  auto value_map_func = [&](Value* v) { return value_map.at(v); };
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    auto subgraph_input = subgraph->addInput();
    subgraph_input->copyMetadata(node->inputs().at(i));
    value_map[node->inputs().at(i)] = subgraph_input;
  }
  // Find node position in owning block, all subsequent nodes after are added to
  // subgraph
  auto owning_block = node->owningBlock();
  auto it = std::find(
      owning_block->nodes().begin(), owning_block->nodes().end(), node);
  // Skip TupleUnpack node if created
  if (!unpack_output) {
    it++;
  }
  for (it++; it != owning_block->nodes().end(); ++it) {
    torch::jit::Node* node = *it;
    auto* clone_node =
        subgraph->insertNode(subgraph->createClone(node, value_map_func));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      value_map[node->outputs()[i]] = clone_node->outputs()[i];
      auto trace_it = std::find(
          trace_outputs.begin(), trace_outputs.end(), node->outputs()[i]);
      if (trace_it != trace_outputs.end()) {
        subgraph->registerOutput(clone_node->outputs()[i]);
      }
    }
  }
}

static torch::jit::Node* _trace_pre_record(
    PyObject* op_obj,
    PyObject* input_objects,
    const variable_list& input_vars) {
  if (!jit::tracer::isTracing()) {
    return nullptr;
  }

  // Save scalar args and the calling convention
  auto num_args = PyTuple_GET_SIZE(input_objects);
  pyobj_list scalar_args;
  std::string arg_types;
  arg_types.reserve(num_args);
  scalar_args.reserve(num_args);
  for (const auto i : c10::irange(num_args)) {
    PyObject* arg_object = PyTuple_GET_ITEM(input_objects, i);
    if (THPVariable_Check(arg_object)) {
      arg_types.push_back('d');
    } else {
      arg_types.push_back('c');
      Py_INCREF(arg_object);
      scalar_args.emplace_back(arg_object);
    }
  }

  Py_INCREF(op_obj);
  auto pyobj = THPObjectPtr(op_obj);
  return jit::tracer::preRecordPythonTrace(
      std::move(pyobj), arg_types, input_vars, std::move(scalar_args));
}

static void _trace_post_record(
    torch::jit::Node* node,
    PyObject* op_obj,
    const variable_list& input_vars,
    PyObject* output_objects,
    bool is_inplace,
    bool unpack_output) {
  if (!jit::tracer::isTracing()) {
    return;
  }

  node->i_(jit::attr::inplace, is_inplace);
  if (PyObject* module_name = PyDict_GetItemString(
          ((PyTypeObject*)op_obj)->tp_dict, "__module__")) {
    if (auto ptr = PyUnicode_AsUTF8(module_name)) {
      node->s_(jit::attr::module, std::string(ptr));
    }
  }

  // Isolate C variable ptrs in a vector
  int num_outputs = PyTuple_GET_SIZE(output_objects);
  auto graph = node->owningGraph();
  node->addOutput();
  auto old_node = node;
  if (!unpack_output) {
    std::vector<at::TypePtr> tuple_values(num_outputs, at::TensorType::get());
    auto tuple_type = at::TupleType::create(std::move(tuple_values));
    // Original type is tuple of tensors "without" element type and shape.
    // The missed parts will be added below.
    node->output()->setType(std::move(tuple_type));
    auto unpacked = graph->createTupleUnpack(node->output())->insertAfter(node);
    node = unpacked;
  }

  std::vector<torch::jit::Value*> trace_outputs;
  for (const auto i : c10::irange(num_outputs)) {
    PyObject* obj = PyTuple_GET_ITEM(output_objects, i);
    if (THPVariable_Check(obj)) {
      auto value = node->outputs()[i];
      const auto& tensor = THPVariable_Unpack(obj);
      if (tensor.defined()) {
        value->inferTypeFrom(tensor);
        trace_outputs.push_back(jit::tracer::getValueTrace(tensor));
        jit::tracer::setValueTrace(tensor, value);
      }
    }
  }
  py::object onnx_globals = py::module::import("torch.onnx._globals");
  py::bool_ is_in_onnx_export =
      py::module::import("torch.onnx.__init__").attr("is_in_onnx_export");
  py::bool_ is_autograd_inlining_enabled =
      py::cast<bool>(onnx_globals.attr("GLOBALS").attr("autograd_inlining"));

  if (py::cast<bool>(is_in_onnx_export) &&
      py::cast<bool>(is_autograd_inlining_enabled)) {
    _append_subgraph(old_node, graph, std::move(trace_outputs), unpack_output);
  }

  // If TupleUnpack operator is created, we copy its output type back
  // to the original tuple type.
  if (!unpack_output) {
    std::vector<at::TypePtr> new_tuple_values;
    for (const auto i : c10::irange(num_outputs)) {
      auto ptr = node->outputs()[i]->type();
      new_tuple_values.push_back(ptr);
    }
    auto tuple_type = at::TupleType::create(std::move(new_tuple_values));
    // The i-th tuple element receives a new tensor type with element type and
    // shape.
    old_node->output()->setType(std::move(tuple_type));
  }
}

PyObject* process_outputs(
    PyObject* op_obj,
    const std::shared_ptr<PyNode>& cdata,
    THPFunction* grad_fn,
    const UnpackedInput& unpacked,
    PyObject* inputs,
    THPObjectPtr&& raw_output,
    bool is_executable,
    torch::jit::Node* node,
    bool overridden_setup_context) {
  bool unpack_output = ensure_tuple(raw_output);

  auto num_outputs = PyTuple_GET_SIZE(raw_output.get());

  THPObjectPtr outputs(PyTuple_New(num_outputs));
  if (!outputs)
    throw python_error();

  cdata->clear_input_metadata();

  // Record type, device, and size information about inputs
  if (is_executable) {
    grad_fn->input_info.clear();
    grad_fn->input_info.reserve(unpacked.input_vars.size());
    for (auto& var : unpacked.input_vars) {
      grad_fn->input_info.emplace_back(var);
    }
  }

  std::unordered_set<at::TensorImpl*> to_save_if_setup_context{};
  std::vector<c10::optional<at::Tensor>> tensors_to_save{};
  _get_tensors_to_save(
      grad_fn,
      to_save_if_setup_context,
      tensors_to_save,
      overridden_setup_context,
      is_executable);

  bool is_inplace = static_cast<bool>(grad_fn->dirty_tensors);
  _wrap_outputs(
      cdata,
      grad_fn,
      unpacked.input_vars,
      raw_output,
      outputs,
      is_executable,
      to_save_if_setup_context);
  _trace_post_record(
      node, op_obj, unpacked.input_vars, outputs, is_inplace, unpack_output);

  // It is important that creating the SavedVariables happen after the output
  // wrapping as the outputs must have their grad_fn/fw_grad properly set before
  // we save them.
  if (is_executable) {
    _save_variables(tensors_to_save, cdata, grad_fn);
  } else {
    // Remove unnecessary attributes
    Py_XDECREF(grad_fn->to_save);
    grad_fn->to_save = nullptr;
    Py_XDECREF(grad_fn->non_differentiable);
    grad_fn->non_differentiable = nullptr;
  }

  Py_XDECREF(grad_fn->saved_for_forward);
  grad_fn->saved_for_forward = nullptr;

  // Unpack the output, unless .forward() returned a tuple
  if (unpack_output) {
    PyObject* output = PyTuple_GET_ITEM(outputs.get(), 0);
    Py_INCREF(output);
    return output;
  }

  return outputs.release();
}

PyObject* THPFunction_name(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto cdata = ((THPFunction*)self)->cdata.lock();
  TORCH_CHECK(
      cdata,
      "Attribute 'name' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  return THPUtils_packString(cdata->name());
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_sequence_nr(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS;
  auto cdata = ((THPFunction*)self)->cdata.lock();
  return THPUtils_packUInt64(cdata->sequence_nr());
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_set_sequence_nr(PyObject* self, PyObject* sequence_nr) {
  HANDLE_TH_ERRORS;
  auto cdata = ((THPFunction*)self)->cdata.lock();
  cdata->set_sequence_nr(THPUtils_unpackUInt64(sequence_nr));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_maybe_clear_saved_tensors(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS;
  auto cdata = ((THPFunction*)self)->cdata.lock();
  if (!get_current_graph_task_keep_graph()) {
    cdata->release_variables();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

namespace {

THPObjectPtr make_ctx_input_tuple(
    THPFunction* ctx,
    const UnpackedInput& unpacked_input,
    int64_t num_args) {
  THPObjectPtr ctx_input_tuple(PyTuple_New(num_args + 1));
  if (!ctx_input_tuple)
    return {};
  Py_INCREF(ctx);
  PyTuple_SET_ITEM(ctx_input_tuple.get(), 0, (PyObject*)ctx);
  for (const auto i : c10::irange(num_args)) {
    PyObject* arg = PyTuple_GET_ITEM(unpacked_input.input_tuple.get(), i);
    Py_INCREF(arg);
    PyTuple_SET_ITEM(ctx_input_tuple.get(), i + 1, arg);
  }
  return ctx_input_tuple;
}

THPObjectPtr make_ctx_input_output_tuple(
    THPFunction* ctx,
    UnpackedInput& unpacked_input,
    PyObject* output) {
  THPObjectPtr result(PyTuple_New(3));
  if (!result)
    return {};
  Py_INCREF(ctx);
  Py_INCREF(unpacked_input.input_tuple.get());
  Py_INCREF(output);
  PyTuple_SET_ITEM(result.get(), 0, (PyObject*)ctx);
  PyTuple_SET_ITEM(result.get(), 1, unpacked_input.input_tuple.get());
  PyTuple_SET_ITEM(result.get(), 2, output);
  return result;
}

} // namespace

static PyObject* THPFunction_setup_context = nullptr;

static PyObject* get_base_setup_context() {
  if (THPFunction_setup_context != nullptr) {
    return THPFunction_setup_context;
  }

  auto module = THPObjectPtr(PyImport_ImportModule("torch.autograd.function"));
  if (!module)
    return nullptr;

  auto function =
      THPObjectPtr(PyObject_GetAttrString(module, "_SingleLevelFunction"));
  if (!function)
    return nullptr;

  // setup_context gets "leaked" - we return a new reference and hold onto it
  // forever.
  auto setup_context = PyObject_GetAttrString(function, "setup_context");
  if (!setup_context)
    return nullptr;
  THPFunction_setup_context = setup_context;
  return THPFunction_setup_context;
}

PyObject* THPFunction_apply(PyObject* cls, PyObject* inputs) {
  HANDLE_TH_ERRORS

  // save a local copy of seq_id before it gets incremented
  auto seq_id = at::sequence_number::peek();
  auto info_pair = unpack_input<false>(inputs);
  UnpackedInput& unpacked_input = info_pair.first;
  InputFlags& input_info = info_pair.second;

  // Call record function after all the inputs have been decoded, but
  // before context has been allocated.
  RECORD_FUNCTION(
      ((PyTypeObject*)cls)->tp_name,
      unpacked_input.record_function_inputs,
      seq_id);

  const auto& functorch_tls = at::functorch::functorchTLSAccessor();
  if (functorch_tls) {
    // autograd.Function support for functorch is handled in Python.
    // If we have gotten here, then either we are dealing with a
    // torch.autograd.function._SingleLevelFunction, or something in
    // the implementation went wrong.
    // The following code is useful for debugging when something goes wrong
    // because it'll raise a loud error (instead of being silently incorrect).
    functorch_tls->checkSupportsSingleLevelAutogradFunction();
  }

  THPObjectPtr backward_cls(PyObject_GetAttrString(cls, "_backward_cls"));
  if (!backward_cls)
    return nullptr;
  THPObjectPtr ctx_obj(PyObject_CallFunctionObjArgs(backward_cls, nullptr));
  if (!ctx_obj)
    return nullptr;
  THPFunction* ctx = (THPFunction*)ctx_obj.get();

  auto cdata =
      std::shared_ptr<PyNode>(new PyNode(std::move(ctx_obj)), deleteNode);
  ctx->cdata = cdata;

  // Record input nodes if tracing
  auto* node = _trace_pre_record(cls, inputs, unpacked_input.input_vars);

  // Initialize backward function (and ctx)
  bool is_executable = input_info.is_executable;
  cdata->set_next_edges(std::move(input_info.next_edges));
  ctx->needs_input_grad = input_info.needs_input_grad.release();
  ctx->is_variable_input = std::move(input_info.is_variable_input);

  // autograd.Function may optionally override a setup_context staticmethod.
  // In this case, autograd.Function.forward does NOT accept a ctx object.
  // Determine if this is the case.
  auto cls_setup_context =
      THPObjectPtr(PyObject_GetAttrString(cls, "setup_context"));
  if (!cls_setup_context) {
    return nullptr;
  }
  auto orig_setup_context = get_base_setup_context();
  if (!orig_setup_context) {
    return nullptr;
  }
  auto overridden_setup_context = cls_setup_context.get() != orig_setup_context;

  auto num_args = PyTuple_GET_SIZE(inputs);

  // Call forward
  THPObjectPtr output;
  {
    AutoGradMode grad_mode(false);
    at::AutoFwGradMode fw_grad_mode(false);
    THPObjectPtr forward_fn(PyObject_GetAttrString(cls, "forward"));
    if (!forward_fn)
      return nullptr;
    if (overridden_setup_context) {
      // call forward followed by setup_context
      output = PyObject_CallObject(forward_fn, unpacked_input.input_tuple);
      if (!output) {
        return nullptr;
      }
      // signature is setup_context(ctx, inputs, output)
      auto ctx_input_output_tuple =
          make_ctx_input_output_tuple(ctx, unpacked_input, output);
      if (!ctx_input_output_tuple) {
        return nullptr;
      }
      THPObjectPtr setup_context_fn(
          PyObject_GetAttrString(cls, "setup_context"));
      auto result =
          PyObject_CallObject(setup_context_fn, ctx_input_output_tuple);
      if (!result) {
        return nullptr;
      }
    } else {
      // call forward
      auto ctx_input_tuple =
          make_ctx_input_tuple(ctx, unpacked_input, num_args);
      if (!ctx_input_tuple) {
        return nullptr;
      }
      output = PyObject_CallObject(forward_fn, ctx_input_tuple);
    }
    if (!output)
      return nullptr;
  }

  return process_outputs(
      cls,
      cdata,
      ctx,
      unpacked_input,
      inputs,
      std::move(output),
      is_executable,
      node,
      overridden_setup_context);
  END_HANDLE_TH_ERRORS
}

////////////////////////////////////////////////////////////////////////////////
// Other methods / attributes
////////////////////////////////////////////////////////////////////////////////

PyObject* THPFunction__register_hook_dict(PyObject* _self, PyObject* _var) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPVariable_Check(_var), "_register_hook_dict expected a Tensor");
  THPVariable* var = reinterpret_cast<THPVariable*>(_var);
  const auto& tensor = THPVariable_Unpack(var);
  std::unique_ptr<FunctionPreHook> hook(
      new PyFunctionTensorPreHook(var->backward_hooks, tensor.output_nr()));
  auto self = (THPFunction*)_self;
  auto cdata = self->cdata.lock();
  TORCH_CHECK(
      cdata,
      "Attribute '_register_hook_dict' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  cdata->add_tensor_pre_hook(std::move(hook));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_register_hook(PyObject* _self, PyObject* hook) {
  HANDLE_TH_ERRORS
  auto self = (THPFunction*)_self;
  auto cdata = self->cdata.lock();
  TORCH_CHECK(
      cdata,
      "Attribute 'register_hook' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  return torch::autograd::registerFunctionHook(*cdata, hook);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_register_prehook(PyObject* _self, PyObject* hook) {
  HANDLE_TH_ERRORS
  auto self = (THPFunction*)_self;
  auto cdata = self->cdata.lock();
  TORCH_CHECK(
      cdata,
      "Attribute 'register_prehook' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  return torch::autograd::registerFunctionPreHook(*cdata, hook);
  END_HANDLE_TH_ERRORS
}

int THPFunction_set_materialize_grads(
    THPFunction* self,
    PyObject* value,
    void* unused) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(value)) {
    THPUtils_invalidArguments(
        value, nullptr, "set_materialize_grads", 1, "(bool)");
    return -1;
  }
  self->materialize_grads = (value == Py_True);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPFunction_get_materialize_non_diff_grads(
    THPFunction* self,
    void* _unused) {
  HANDLE_TH_ERRORS
  if (self->materialize_non_diff_grads) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

int THPFunction_set_materialize_non_diff_grads(
    THPFunction* self,
    PyObject* value,
    void* unused) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(value)) {
    THPUtils_invalidArguments(
        value, nullptr, "set_materialize_non_diff_grads", 1, "(bool)");
    return -1;
  }
  self->materialize_non_diff_grads = (value == Py_True);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPFunction_saved_tensors(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  if (self->saved_for_forward) {
    Py_INCREF(self->saved_for_forward);
    return self->saved_for_forward;
  } else {
    return unpack_saved_variables(
        self, [](const Variable& var) { return THPVariable_Wrap(var); });
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_saved_variables(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  auto r = PyErr_WarnEx(
      PyExc_DeprecationWarning,
      "'saved_variables' is deprecated; use 'saved_tensors'",
      0);
  if (r != 0)
    throw python_error();
  return unpack_saved_variables(
      self, [](const Variable& var) { return THPVariable_Wrap(var); });
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_is_compiled_autograd_tracing(
    PyObject* self,
    PyObject* _unused) {
  HANDLE_TH_ERRORS
  if (((THPFunction*)self)->compiled_autograd_tracing) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_get_compiled_autograd_symints(
    PyObject* _self,
    PyObject* _unused) {
  HANDLE_TH_ERRORS
  auto self = (THPFunction*)_self;
  auto size = self->compiled_autograd_symints.size();
  PyObject* result = PyTuple_New(static_cast<Py_ssize_t>(size));
  if (!result) {
    throw python_error();
  }
  for (const auto i : c10::irange(size)) {
    PyTuple_SET_ITEM(
        result,
        i,
        py::cast(self->compiled_autograd_symints[i]).release().ptr());
  }
  return result;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_get_compiled_autograd_backward_state(
    PyObject* _self,
    void* _unused) {
  HANDLE_TH_ERRORS
  auto self = (THPFunction*)_self;
  PyObject* bw_state = self->compiled_autograd_backward_state;
  if (bw_state == nullptr) {
    bw_state = Py_None;
  }
  Py_INCREF(bw_state);
  return bw_state;
  END_HANDLE_TH_ERRORS
}

int THPFunction_set_compiled_autograd_backward_state(
    PyObject* _self,
    PyObject* bw_state,
    void* _unused) {
  HANDLE_TH_ERRORS
  auto self = (THPFunction*)_self;
  TORCH_INTERNAL_ASSERT(self->compiled_autograd_backward_state == nullptr);
  Py_INCREF(bw_state);
  self->compiled_autograd_backward_state = bw_state;
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPFunction_raw_saved_tensors(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  // User tries to access saved variables after they have been freed
  TORCH_CHECK(!self->has_freed_buffers, ERR_BACKWARD_TWICE);
  const auto& saved_variables = self->saved_variables;
  if (saved_variables.empty())
    return PyTuple_New(0);
  size_t num_saved = saved_variables.size();
  THPObjectPtr saved(PyTuple_New(static_cast<Py_ssize_t>(num_saved)));
  if (!saved) {
    return nullptr;
  }
  for (const auto i : c10::irange(num_saved)) {
    py::object obj =
        py::cast(saved_variables[i], py::return_value_policy::reference);
    PyTuple_SET_ITEM(saved.get(), i, obj.release().ptr());
  }
  return saved.release();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_next_functions(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  auto cdata = self->cdata.lock();
  TORCH_CHECK(
      cdata,
      "Attribute 'next_functions' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  const auto num_outputs = cdata->num_outputs();
  THPObjectPtr result(PyTuple_New(num_outputs));
  if (!result)
    return nullptr;
  for (const auto i : c10::irange(num_outputs)) {
    THPObjectPtr fn_tuple(PyTuple_New(2));
    if (!fn_tuple)
      return nullptr;
    const auto& edge = cdata->next_edge(i);
    PyObject* fn = functionToPyObject(edge.function);
    if (!fn)
      return nullptr;
    PyTuple_SET_ITEM(fn_tuple.get(), 0, fn);
    PyTuple_SET_ITEM(fn_tuple.get(), 1, THPUtils_packInt64(edge.input_nr));
    PyTuple_SET_ITEM(result.get(), i, fn_tuple.release());
  }
  return result.release();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_metadata(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  auto cdata = self->cdata.lock();
  // The correct way to solve this problem is to stop exposing grad_fn
  // of PyFunctions as THPFunction; instead, we should use THPCppFunction
  // like everyone else.  But this is a BC-breaking change as it would
  // mean that you no longer get the property that grad_fn is a subclass
  // of the autograd function class that you defined in the custom case,
  // so I didn't fix it here.
  TORCH_CHECK(
      cdata,
      "You attempted to access the anomaly metadata of a custom autograd function "
      "but the underlying PyNode has already been deallocated.  The most likely "
      "reason this occurred is because you assigned x.grad_fn to a local variable "
      "and then let the original variable get deallocated.  Don't do that!  If "
      "you really have no way of restructuring your code so this is the case, "
      "please file an issue reporting that you are affected by this.");
  auto metadata = static_cast<PyAnomalyMetadata*>(cdata->metadata())->dict();

  Py_INCREF(metadata);
  return metadata;
  END_HANDLE_TH_ERRORS
}

typedef PyObject* (*getter)(PyObject*, void*);
typedef int (*setter)(PyObject*, PyObject*, void*);

namespace {

template <PyObject* THPFunction::*ptr>
PyObject* getObject(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  PyObject* value = self->*ptr;
  if (!value) {
    Py_RETURN_NONE;
  }
  Py_INCREF(value);
  return value;
}

template <PyObject* THPFunction::*ptr>
int setObject(PyObject* obj, PyObject* value, void* _unused) {
  auto self = (THPFunction*)obj;
  if (value == Py_None) {
    value = nullptr;
  }
  Py_XDECREF((self->*ptr));
  Py_XINCREF(value);
  self->*ptr = value;
  return 0;
}

template <typename M, M THPFunction::*ptr, PyObject* (*Convert)(long)>
PyObject* getMember(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  return Convert(self->*ptr);
}

template <typename M, M autograd::Node::*ptr, PyObject* (*Convert)(long)>
PyObject* getImplMember(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  return Convert(self->cdata.*ptr);
}

PyObject* getRequiresGrad(PyObject* obj, void* _unused) {
  Py_RETURN_TRUE;
}

} // namespace

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPFunction_properties[] = {
    {"saved_tensors",
     (getter)THPFunction_saved_tensors,
     nullptr,
     nullptr,
     nullptr},
    {"saved_variables",
     (getter)THPFunction_saved_variables,
     nullptr,
     nullptr,
     nullptr},
    {"_raw_saved_tensors",
     (getter)THPFunction_raw_saved_tensors,
     nullptr,
     nullptr,
     nullptr},
    {"next_functions",
     (getter)THPFunction_next_functions,
     nullptr,
     nullptr,
     nullptr},
    {"to_save",
     &getObject<&THPFunction::to_save>,
     &setObject<&THPFunction::to_save>,
     nullptr,
     nullptr},
    {"non_differentiable",
     &getObject<&THPFunction::non_differentiable>,
     &setObject<&THPFunction::non_differentiable>,
     nullptr,
     nullptr},
    {"dirty_tensors",
     &getObject<&THPFunction::dirty_tensors>,
     &setObject<&THPFunction::dirty_tensors>,
     nullptr,
     nullptr},
    {"saved_for_forward",
     &getObject<&THPFunction::saved_for_forward>,
     &setObject<&THPFunction::saved_for_forward>,
     nullptr,
     nullptr},
    {"needs_input_grad",
     &getObject<&THPFunction::needs_input_grad>,
     nullptr,
     nullptr,
     nullptr},
    {"requires_grad", getRequiresGrad, nullptr, nullptr, nullptr},
    {"metadata", (getter)THPFunction_metadata, nullptr, nullptr, nullptr},
    {"materialize_grads",
     nullptr,
     (setter)THPFunction_set_materialize_grads,
     nullptr,
     nullptr},
    {"_materialize_non_diff_grads",
     (getter)THPFunction_get_materialize_non_diff_grads,
     (setter)THPFunction_set_materialize_non_diff_grads,
     nullptr,
     nullptr},
    {"_compiled_autograd_backward_state",
     (getter)THPFunction_get_compiled_autograd_backward_state,
     (setter)THPFunction_set_compiled_autograd_backward_state,
     nullptr,
     nullptr},
    {nullptr}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyMethodDef THPFunction_methods[] = {
    {(char*)"name", THPFunction_name, METH_NOARGS, nullptr},
    {(char*)"_sequence_nr", THPFunction_sequence_nr, METH_NOARGS, nullptr},
    {(char*)"_set_sequence_nr", THPFunction_set_sequence_nr, METH_O, nullptr},
    {(char*)"maybe_clear_saved_tensors",
     THPFunction_maybe_clear_saved_tensors,
     METH_NOARGS,
     nullptr},
    {(char*)"apply", THPFunction_apply, METH_CLASS | METH_VARARGS, nullptr},
    {(char*)"_register_hook_dict",
     THPFunction__register_hook_dict,
     METH_O,
     nullptr},
    {(char*)"register_hook", THPFunction_register_hook, METH_O, nullptr},
    {(char*)"register_prehook", THPFunction_register_prehook, METH_O, nullptr},
    {(char*)"_is_compiled_autograd_tracing",
     THPFunction_is_compiled_autograd_tracing,
     METH_NOARGS,
     nullptr},
    {(char*)"_get_compiled_autograd_symints",
     THPFunction_get_compiled_autograd_symints,
     METH_NOARGS,
     nullptr},
    {nullptr}};

PyTypeObject THPFunctionType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._FunctionBase", /* tp_name */
    sizeof(THPFunction), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THPFunction_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    (traverseproc)THPFunction_traverse, /* tp_traverse */
    (inquiry)THPFunction_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPFunction_methods, /* tp_methods */
    nullptr, /* tp_members */
    THPFunction_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPFunction_new /* tp_new */
};

bool THPFunction_initModule(PyObject* module) {
  if (PyType_Ready(&THPFunctionType) < 0)
    return false;
  Py_INCREF(&THPFunctionType);
  PyModule_AddObject(module, "_FunctionBase", (PyObject*)&THPFunctionType);
  return true;
}
