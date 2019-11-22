#include <torch/csrc/autograd/python_function.h>

#include <torch/csrc/python_headers.h>
#include <structmember.h>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <ATen/ATen.h>

#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/python_tracer.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/Exceptions.h>

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace torch;
using namespace torch::autograd;
using namespace torch::jit;
using at::Tensor;

PyObject *THPFunctionClass = nullptr;

#define THPFunction_assert(condition, ...)                                     \
  if (!(condition)) { THPUtils_setError(__VA_ARGS__); throw python_error(); }

namespace torch { namespace autograd {

auto PyNode::legacy_apply(const variable_list& inputs) -> variable_list {
  AutoGIL gil;

  THPObjectPtr pyInputs(PyTuple_New(inputs.size()));
  if (!pyInputs) throw python_error();

  for (size_t i = 0; i != inputs.size(); ++i) {
    PyTuple_SET_ITEM(pyInputs.get(), i, THPVariable_Wrap(inputs[i]));
  }

  THPObjectPtr r(PyObject_CallMethod(
      obj, "_do_backward", "OO", pyInputs.get(), Py_True));
  if (!r) throw python_error();

  auto num_outputs = PyTuple_GET_SIZE(r.get());
  tensor_list tensor_results(num_outputs);
  for (int i = 0; i != num_outputs; ++i) {
    PyObject* obj = PyTuple_GET_ITEM(r.get(), i);
    if (obj != Py_None) {
      if (!THPVariable_Check(obj)) {
        std::string msg("expected Variable (got '");
        msg += THPUtils_typename(obj);
        msg += "')'";
        throw std::runtime_error(msg);
      }
      tensor_results[i] = ((THPVariable*)obj)->cdata.tensor_data();
    }
  }

  // XXX: this might get requires_grad wrong - there's no way to figure out
  // if _do_backward didn't use ctx.saved_tensors and as a result some
  // Variables might require grad, even if no args do. Unfortunately, this
  // leads to unexpected error messages ("no nodes require computing gradients"),
  // but I don't have a better idea. These functions would raise an error
  // in backward anyway.
  return wrap_outputs(
      inputs,
      std::move(tensor_results),
      [this](edge_list&& next_edges) {
        return std::make_shared<Error>(
            name() + " is not differentiable twice", std::move(next_edges));
      });
}

// NOTE: this function is written in a way that assumes it's only called for backward;
// it's used by engine.cpp.  This is responsible for forwarding a call from
// C++'s Node::apply to a Python method "apply".
auto PyNode::apply(variable_list&& inputs) -> variable_list {
  AutoGIL gil;
  at::OptionalDeviceGuard _device_guard;
  THPFunction* py_fn = (THPFunction*)obj;

  THPObjectPtr _legacy(PyObject_GetAttrString(obj, "_is_legacy"));
  if (_legacy == Py_True) {
    return legacy_apply(inputs);
  }

  // Massage a C++ variable_list into a Python arguments tuple
  auto num_inputs = inputs.size();
  THPObjectPtr pyInputs(PyTuple_New(num_inputs));
  if (!pyInputs) throw python_error();
  auto& output_info = py_fn->output_info;
  for (size_t i = 0; i < num_inputs; ++i) {
    PyObject* input;
    if (inputs[i].defined()) {
      input = THPVariable_Wrap(inputs[i]);
    } else {
      input = THPVariable_Wrap(output_info[i].zeros(_device_guard));
    }
    if (!input) throw python_error();
    PyTuple_SET_ITEM(pyInputs.get(), i, input);
  }

  THPObjectPtr apply_fn(PyObject_GetAttrString(obj, "apply"));
  if (!apply_fn) throw python_error();
  THPObjectPtr r(PyObject_CallObject(apply_fn, pyInputs.get()));
  if (!r) throw python_error();
  ensure_tuple(r);

  auto& is_variable_input = py_fn->is_variable_input;
  int num_outputs = PyTuple_GET_SIZE(r.get());
  int num_forward_inputs = is_variable_input.size();
  // Returning too many results is ok, but only as long as they're all None.
  // Truncate the result tuple in that case.
  if (num_outputs > num_forward_inputs) {
    bool all_none = true;
    for (int i = num_forward_inputs; i < num_outputs; i++) {
      all_none &= PyTuple_GET_ITEM(r.get(), i) == Py_None;
    }
    if (all_none) {
      num_outputs = num_forward_inputs;
      r = PyTuple_GetSlice(r.get(), 0, num_forward_inputs);
      if (!r) throw python_error();
    }
  }

  // Now the number of gradients should match
  if (num_outputs != num_forward_inputs) {
    std::string msg("function ");
    msg += name() + " returned an incorrect number of gradients (expected ";
    msg += std::to_string(num_forward_inputs) + ", got " ;
    msg += std::to_string(num_outputs) + ")";
    throw std::runtime_error(msg);
  }

  // Massage the Python results tuple back into a C++ variable_list
  variable_list results;
  results.reserve(num_outputs);
  auto& input_info = py_fn->input_info;
  for (int i = 0; i != num_outputs; ++i) {
    PyObject* output = PyTuple_GET_ITEM(r.get(), i);
    bool was_variable = is_variable_input[i];
    if (!was_variable) {
      if (output != Py_None) {
        std::string msg("function ");
        msg += name() + " returned a gradient different than None at position ";
        msg += std::to_string(i + 1) + ", but the corresponding forward input was not a Variable";
        throw std::runtime_error(msg);
      }
      continue;
    }
    if (output == Py_None) {
      auto& info = input_info[results.size()];
      if (info.requires_grad) {
        results.emplace_back(info.zeros(_device_guard));
      } else {
        results.emplace_back();
      }
    } else {
      if (!THPVariable_Check(output)) {
        std::string msg("expected Variable or None (got ");
        msg += THPUtils_typename(output);
        msg += ")";
        throw std::runtime_error(msg);
      }
      results.emplace_back(((THPVariable*)output)->cdata);
    }
  }

  return results;
}

auto PyNode::is_traceable() -> bool {
  AutoGIL gil;
  THPObjectPtr forward_class {PyObject_GetAttrString(obj, "_forward_cls")};
  if (!forward_class) throw python_error();
  THPObjectPtr traceable_py_bool {PyObject_GetAttrString(forward_class, "is_traceable")};
  if (!traceable_py_bool) throw python_error();
  return traceable_py_bool == Py_True;
}

auto PyNode::release_variables() -> void {
  AutoGIL gil;
  auto f = (THPFunction*) obj;
  f->saved_variables.clear();
  f->has_freed_buffers = 1;
}

auto PyNode::name() const -> std::string {
  AutoGIL gil;
  auto f = (THPFunction*) obj;
  auto name = std::string(Py_TYPE(f)->tp_name);
  // Python API functions are not const-correct
  THPObjectPtr _legacy(PyObject_GetAttrString(const_cast<PyObject*>(obj), "_is_legacy")); // NOLINT
  if (_legacy == Py_True) {
    name += "LegacyBackward";
  }
  return name;
}

}} // namespace torch::autograd

// Traverse and clear are required for supporting Python's GC cycle handling.
static int THPFunction_traverse(THPFunction *self, visitproc visit, void *arg)
{
  // cdata could be null if someone constructed a legacy function but haven't
  // actually called backward() on it yet, or if the PyNode has already
  // gone out of scope by the time we're GC'ing this THPFunction (e.g., the
  // user saved grad_fn only).
  //
  // TODO: I'm not really sure if we're actually obligated to traverse PyObject
  // that is stored in PyNode, since we don't really own that C++ object.
  if (auto cdata = self->cdata.lock()) {
    for (const auto& hook : cdata->pre_hooks()) {
      if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
    for (const auto& hook : cdata->post_hooks()) {
      if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
  }
  Py_VISIT(self->to_save);
  Py_VISIT(self->non_differentiable);
  Py_VISIT(self->dirty_tensors);
  return 0;
}

static int THPFunction_clear(THPFunction *self)
{
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

  Py_CLEAR(self->needs_input_grad);

  Py_CLEAR(self->to_save);
  Py_CLEAR(self->non_differentiable);
  Py_CLEAR(self->dirty_tensors);

  self->output_info.clear();
  self->input_info.clear();
  self->saved_variables.clear();
  self->is_variable_input.clear();

  return 0;
}

static void THPFunction_dealloc(THPFunction* self)
{
  PyObject_GC_UnTrack(self);
  THPFunction_clear(self);
  self->cdata.~weak_ptr<PyNode>();
  self->output_info.~vector();
  self->input_info.~vector();
  self->saved_variables.~vector();
  self->is_variable_input.~vector();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *THPFunction_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (!obj) return nullptr;
  // Python zero-initializes the object memory, so there's no need to initialize
  // most fields
  THPFunction* self = (THPFunction*)obj;
  // Setup the PyNode later; we can't keep it live here
  new (&self->cdata) std::weak_ptr<PyNode>();
  new (&self->output_info) std::vector<VariableInfo>();
  new (&self->input_info) std::vector<VariableInfo>();
  new (&self->saved_variables) std::vector<SavedVariable>();
  new (&self->is_variable_input) std::vector<bool>();
  return obj;
}

////////////////////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////////////////////

using t2var_type = std::unordered_map<PyObject *, THPVariable *>;

// Bump the counters of all recorded dirty input tensors, adding each of them
// into dirty_inputs.  Also does some sanity checking.
static std::unordered_set<at::TensorImpl*> _mark_dirty(THPFunction *self)
{
  // Increase versions of modified tensors
  std::unordered_set<at::TensorImpl*> dirty_inputs;
  if (!self->dirty_tensors) return dirty_inputs;

  THPFunction_assert(PyTuple_Check(self->dirty_tensors), "autograd "
      "internal error: dirty_tensors attribute is expected to be a tuple "
      "but is %s", THPUtils_typename(self->dirty_tensors));
  Py_ssize_t num_dirty = PyTuple_GET_SIZE(self->dirty_tensors);
  dirty_inputs.reserve(num_dirty);
  for (int i = 0; i < num_dirty; i++) {
    PyObject *obj = PyTuple_GET_ITEM(self->dirty_tensors, i);
    THPFunction_assert(THPVariable_Check(obj), "mark_dirty can "
        "only accept variables, but argument %d is of type %s", i,
        THPUtils_typename(obj));

    dirty_inputs.insert(((THPVariable*)obj)->cdata.unsafeGetTensorImpl());
    auto variable = (THPVariable*)obj;
    torch::autograd::impl::bump_version(variable->cdata);
  }
  // We're not going to ever need this so let's remove references now
  Py_CLEAR(self->dirty_tensors);
  return dirty_inputs;
}

static std::unordered_set<at::TensorImpl*> _parse_non_differentiable(THPFunction *self);

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
static void _wrap_outputs(const std::shared_ptr<PyNode>& cdata, THPFunction *self,
    const variable_list &input_vars, PyObject *raw_output, PyObject *outputs, bool is_executable)
{
  auto cdata_if_executable = is_executable ? cdata : nullptr;
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(raw_output);
  if (is_executable) {
    self->output_info.clear();
    self->output_info.reserve(num_outputs);
  }

  auto as_variable = [&](PyObject* obj, int i) -> Variable {
    if (THPVariable_Check(obj)) {
      return ((THPVariable*)obj)->cdata;
    }
    throw TypeError("%s.forward: expected Variable (got %s) for return value %d",
        Py_TYPE(self)->tp_name, Py_TYPE(obj)->tp_name, i);
  };

  auto non_differentiable = _parse_non_differentiable(self);
  auto dirty_inputs = _mark_dirty(self);

  std::vector<Variable> raw_output_vars;
  raw_output_vars.reserve(num_outputs);
  for(int i = 0; i < num_outputs; ++i){
    PyObject* obj = PyTuple_GET_ITEM(raw_output, i);
    raw_output_vars.push_back(as_variable(obj,i));
  }

  auto wrapped_outputs = _wrap_outputs(input_vars, non_differentiable, dirty_inputs, raw_output_vars, cdata_if_executable);
  for (int i = 0; i < num_outputs; i++) {
    if (is_executable) {
      self->output_info.emplace_back(wrapped_outputs[i]);
    }
    PyTuple_SET_ITEM(outputs, i, THPVariable_Wrap(wrapped_outputs[i]));
  }
}

// Save any variables that requested by to_save
static void _save_variables(const std::shared_ptr<PyNode>& cdata_ptr, THPFunction* self)
{
  if (!self->to_save) return;

  THPFunction_assert(PyTuple_Check(self->to_save), "autograd internal "
      "error: to_save attribute is expected to be a tuple but is %s",
      THPUtils_typename(self->to_save));
  Py_ssize_t num_saved = PyTuple_GET_SIZE(self->to_save);
  self->saved_variables.clear();
  self->saved_variables.reserve(num_saved);
  for (int i = 0; i < num_saved; i++) {
    PyObject *obj = PyTuple_GET_ITEM(self->to_save, i);
    if (obj == Py_None) {
      self->saved_variables.emplace_back();
      continue;
    } else if (THPVariable_Check(obj)) {
      auto variable = (THPVariable*)obj;
      bool is_output = variable->cdata.grad_fn().get() == cdata_ptr.get();
      self->saved_variables.emplace_back(variable->cdata, is_output);
    } else {
      throw TypeError(
          "save_for_backward can only save variables, but argument %d is of "
          "type %s", i, Py_TYPE(obj)->tp_name);
    }
  }
  // Free .to_save
  Py_CLEAR(self->to_save);
}

// Mark requires_grad = 0 on non-differentiable variables (as per non_differentiable)
static std::unordered_set<at::TensorImpl*>
_parse_non_differentiable(THPFunction *self)
{
  std::unordered_set<at::TensorImpl*> set;
  if (!self->non_differentiable) return set;

  THPFunction_assert(PyTuple_Check(self->non_differentiable), "autograd "
      "internal error: non_differentiable attribute is expected to be a "
      "tuple but is %s", THPUtils_typename(self->non_differentiable));
  Py_ssize_t num_nondiff = PyTuple_GET_SIZE(self->non_differentiable);
  set.reserve(num_nondiff);
  for (int i = 0; i < num_nondiff; i++) {
    PyObject *t = PyTuple_GET_ITEM(self->non_differentiable, i);
    THPFunction_assert(THPVariable_Check(t), "mark_non_differentiable "
        "only accepts variable arguments, but got %s", THPUtils_typename(t));
    set.insert(((THPVariable*)t)->cdata.unsafeGetTensorImpl());
  }
  Py_CLEAR(self->non_differentiable);
  return set;
}

struct UnpackedInput {
  THPObjectPtr input_tuple;
  variable_list input_vars;
};

struct InputFlags {
  bool is_executable = false;
  edge_list next_edges;
  THPObjectPtr needs_input_grad;
  std::vector<bool> is_variable_input;
};

template<bool enforce_variables>
std::pair<UnpackedInput, InputFlags> unpack_input(PyObject *args) {
  UnpackedInput unpacked;
  InputFlags flags;

  auto num_args = PyTuple_GET_SIZE(args);
  unpacked.input_tuple = PyTuple_New(num_args);
  flags.needs_input_grad = PyTuple_New(num_args);
  for (int i = 0; i < num_args; i++) {
    PyObject *arg = PyTuple_GET_ITEM(args, i);

    bool is_variable = THPVariable_Check(arg);
    flags.is_variable_input.push_back(is_variable);
    if (!is_variable) {
      // TODO: remove this code path once Variable and Tensor are merged in Python
      if (enforce_variables) {
        THPUtils_setError("expected a Variable argument, but got %s",
                          THPUtils_typename(arg));
        throw python_error();
      }
      Py_INCREF(Py_False);
      PyTuple_SET_ITEM(flags.needs_input_grad.get(), i, Py_False);
    } else {
      THPVariable* variable = (THPVariable*)arg;
      unpacked.input_vars.push_back(variable->cdata);
      PyObject* needs_grad = variable->cdata.requires_grad() ? Py_True : Py_False;
      Py_INCREF(needs_grad);
      PyTuple_SET_ITEM(flags.needs_input_grad.get(), i, needs_grad);
    }
    Py_INCREF(arg);
    PyTuple_SET_ITEM(unpacked.input_tuple.get(), i, arg);
  }

  flags.is_executable = GradMode::is_enabled() && any_variable_requires_grad(unpacked.input_vars);
  flags.next_edges = collect_next_edges(unpacked.input_vars);
  return std::make_pair(std::move(unpacked), std::move(flags));
}

static void _assert_not_tracing(const char* name, const variable_list& input_vars) {
  if (tracer::isTracing()) {
    std::ostringstream oss;
    oss << "Attempted to trace " << name;
    oss << ", but tracing of legacy functions is not supported";
    throw std::runtime_error(oss.str());
  }
}

static torch::jit::Node* _trace_pre_record(
    PyObject* op_obj,
    PyObject *input_objects,
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
  for (int i = 0; i < num_args; i++) {
    PyObject *arg_object = PyTuple_GET_ITEM(input_objects, i);
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
    PyObject *output_objects,
    bool is_inplace,
    bool unpack_output) {
  if (!jit::tracer::isTracing()) {
    return;
  }

  node->i_(attr::inplace, is_inplace);

  // Isolate C variable ptrs in a vector
  int num_outputs = PyTuple_GET_SIZE(output_objects);
  variable_list output_vars(num_outputs);
  auto graph = node->owningGraph();
  node->addOutput();
  if (!unpack_output) {
    std::vector<TypePtr> tuple_values(num_outputs, TensorType::get());
    TypePtr tuple_type = TupleType::create(std::move(tuple_values));
    node->output()->setType(tuple_type);
    auto unpacked = graph->createTupleUnpack(node->output())->insertAfter(node);
    node = unpacked;
  }
  for (int i = 0; i < num_outputs; ++i) {
    auto var = (THPVariable*)PyTuple_GET_ITEM(output_objects, i);
    Value* value = node->outputs()[i];
    if (var->cdata.defined()) {
      value->inferTypeFrom(var->cdata);
      jit::tracer::setValueTrace(autograd::as_variable_ref(var->cdata), value);
    }
  }
}

PyObject* process_outputs(PyObject *op_obj, const std::shared_ptr<PyNode>& cdata,
                          THPFunction* grad_fn, const UnpackedInput& unpacked,
                          PyObject *inputs, THPObjectPtr&& raw_output, bool is_executable,
                          torch::jit::Node* node) {
  bool unpack_output = ensure_tuple(raw_output);

  auto num_outputs = PyTuple_GET_SIZE(raw_output.get());

  THPObjectPtr outputs(PyTuple_New(num_outputs));
  if (!outputs) throw python_error();

  cdata->clear_input_metadata();

  // Record type, device, and size information about inputs
  if (is_executable) {
    grad_fn->input_info.clear();
    grad_fn->input_info.reserve(unpacked.input_vars.size());
    for (auto& var : unpacked.input_vars) {
      grad_fn->input_info.emplace_back(var);
    }
  }

  bool is_inplace = static_cast<bool>(grad_fn->dirty_tensors);
  _wrap_outputs(cdata, grad_fn, unpacked.input_vars, raw_output, outputs, is_executable);
  _trace_post_record(node, op_obj, unpacked.input_vars, outputs, is_inplace, unpack_output);
  if (is_executable) {
    _save_variables(cdata, grad_fn);
  } else {
    // Remove unnecessary attributes
    Py_XDECREF(grad_fn->to_save);
    grad_fn->to_save = nullptr;
    Py_XDECREF(grad_fn->non_differentiable);
    grad_fn->non_differentiable = nullptr;
  }

  // Unpack the output, unless .forward() returned a tuple
  if (unpack_output) {
    PyObject *output = PyTuple_GET_ITEM(outputs.get(), 0);
    Py_INCREF(output);
    return output;
  }

  return outputs.release();
}

// Legacy codepath
PyObject *THPFunction_do_forward(THPFunction *self, PyObject *_inputs)
{
  HANDLE_TH_ERRORS
  RECORD_FUNCTION(
    Py_TYPE(self)->tp_name,
    std::vector<c10::IValue>(),
    autograd::Node::peek_at_next_sequence_nr());

  TORCH_WARN("Legacy autograd function with non-static forward method is deprecated and will be removed in 1.3. ",
             "Please use new-style autograd function with static forward method. ",
             "(Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)");

  auto info_pair = unpack_input<true>(_inputs);
  auto& unpacked_input = info_pair.first;
  auto& input_info = info_pair.second;
  bool is_executable = input_info.is_executable;
  std::shared_ptr<PyNode> cdata = self->cdata.lock();
  if (cdata) {
    // In some pathological cases, self->cdata can already be set on entry to
    // this function.  This occurs on misuse of the legacy autograd API in the
    // following way:
    //
    //    f = MyFunction()
    //    y1 = f(x1)
    //    y2 = f(x2)  # bad!!
    //
    // Historically, we did something very nutty: we set y1.grad_fn ==
    // y2.grad_fn (even though these variables really have nothing to do with
    // each other.)  At least now we have a warning.  All of this hoo-ha will
    // go away when we delete the implementation of legacy autograd.
    TORCH_WARN(
      "Legacy autograd function object was called twice.  You will probably "
      "get incorrect gradients from this computation, as the saved tensors "
      "from the second invocation will clobber the saved tensors from the "
      "first invocation.  Please consider rewriting your autograd function "
      "in the modern style; for information on the new format, please see: "
      "https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd");
  } else {
    Py_INCREF(self);
    cdata = std::shared_ptr<PyNode>(new PyNode(THPObjectPtr((PyObject*)self)), deleteNode);
    self->cdata = cdata;
  }
  cdata->set_next_edges(std::move(input_info.next_edges));
  self->needs_input_grad = input_info.needs_input_grad.release();

  // We don't support tracing in the legacy code path
  _assert_not_tracing(Py_TYPE(self)->tp_name, unpacked_input.input_vars);

  // Now we're ready to call a forward (implemented in Python)
  THPObjectPtr raw_output;
  {
    AutoGradMode grad_mode(false);
    THPObjectPtr forward_fn(PyObject_GetAttrString((PyObject*)self, "forward"));
    if (!forward_fn) return nullptr;
    raw_output = PyObject_CallObject(forward_fn, unpacked_input.input_tuple);
    if (!raw_output) return nullptr;
  }

  return process_outputs(nullptr, cdata, self, unpacked_input, _inputs, std::move(raw_output),
                         is_executable, nullptr);
  END_HANDLE_TH_ERRORS
}

PyObject *THPFunction_apply(PyObject *cls, PyObject *inputs)
{
  HANDLE_TH_ERRORS
  RECORD_FUNCTION(
    ((PyTypeObject*)cls)->tp_name,
    std::vector<c10::IValue>(),
    autograd::Node::peek_at_next_sequence_nr());

  THPObjectPtr backward_cls(PyObject_GetAttrString(cls, "_backward_cls"));
  if (!backward_cls) return nullptr;
  THPObjectPtr ctx_obj(PyObject_CallFunctionObjArgs(backward_cls, nullptr));
  if (!ctx_obj) return nullptr;
  THPFunction* ctx = (THPFunction*)ctx_obj.get();

  auto cdata = std::shared_ptr<PyNode>(new PyNode(std::move(ctx_obj)), deleteNode);
  ctx->cdata = cdata;

  // Prepare inputs and allocate context (grad fn)
  auto info_pair = unpack_input<false>(inputs);
  UnpackedInput& unpacked_input = info_pair.first;
  InputFlags& input_info = info_pair.second;

  // Record input nodes if tracing
  auto* node = _trace_pre_record(cls, inputs, unpacked_input.input_vars);

  // Initialize backward function (and ctx)
  bool is_executable = input_info.is_executable;
  cdata->set_next_edges(std::move(input_info.next_edges));
  ctx->needs_input_grad = input_info.needs_input_grad.release();
  ctx->is_variable_input = std::move(input_info.is_variable_input);

  // Prepend ctx to input_tuple, in preparation for static method call
  auto num_args = PyTuple_GET_SIZE(inputs);
  THPObjectPtr ctx_input_tuple(PyTuple_New(num_args + 1));
  if (!ctx_input_tuple) return nullptr;
  Py_INCREF(ctx);
  PyTuple_SET_ITEM(ctx_input_tuple.get(), 0, (PyObject*)ctx);
  for (int i = 0; i < num_args; ++i) {
    PyObject *arg = PyTuple_GET_ITEM(unpacked_input.input_tuple.get(), i);
    Py_INCREF(arg);
    PyTuple_SET_ITEM(ctx_input_tuple.get(), i + 1, arg);
  }

  // Call forward
  THPObjectPtr tensor_outputs;
  {
    AutoGradMode grad_mode(false);
    THPObjectPtr forward_fn(PyObject_GetAttrString(cls, "forward"));
    if (!forward_fn) return nullptr;
    tensor_outputs = PyObject_CallObject(forward_fn, ctx_input_tuple);
    if (!tensor_outputs) return nullptr;
  }

  return process_outputs(cls, cdata, ctx, unpacked_input, inputs, std::move(tensor_outputs),
                         is_executable, node);
  END_HANDLE_TH_ERRORS
}


////////////////////////////////////////////////////////////////////////////////
// Backward
////////////////////////////////////////////////////////////////////////////////

static void _prepare_grads(THPFunction *self, THPObjectPtr& raw_grads, bool is_grad_output)
{
  at::OptionalDeviceGuard device_guard;
  int num_grads = PyTuple_GET_SIZE(raw_grads.get());
  // First, check if any of grads is None. If not, there's nothing to do
  bool has_none = false;
  for (int i = 0; i < num_grads; i++) {
    has_none |= PyTuple_GET_ITEM(raw_grads.get(), i) == Py_None;
  }
  if (!has_none)
      return;

  THPObjectPtr grads;
  grads = PyTuple_New(num_grads);
  if (!grads) throw python_error();

  // Look for Nones and replace them with new buffers
  auto& grads_info = is_grad_output ? self->output_info : self->input_info;
  AT_ASSERT(grads_info.size() == (size_t)num_grads);
  for (int i = 0; i < num_grads; i++) {
    PyObject *grad = PyTuple_GET_ITEM(raw_grads.get(), i);
    if (grad == Py_None) {
      grad = THPVariable_Wrap(grads_info[i].zeros(device_guard));
      if (!grad) throw python_error();
    } else {
      Py_INCREF(grad);
    }
    PyTuple_SET_ITEM(grads.get(), i, grad);
  }
  raw_grads = grads.release();
}

static void _trim_grad_input(const std::shared_ptr<PyNode>& cdata, THPFunction *self, THPObjectPtr& grad_input)
{
  int num_grads = PyTuple_GET_SIZE(grad_input.get());
  const int num_outputs = cdata->num_outputs();
  if (num_grads > num_outputs) {
    // Check that all extra grads are none
    bool all_none = true;
    for (int i = num_outputs; i < num_grads; i++) {
      all_none = (PyTuple_GET_ITEM(grad_input.get(), i) == Py_None);
      if (!all_none) break;
    }
    // If yes, slice the tuple
    if (all_none) {
      num_grads = num_outputs;
      grad_input = PyTuple_GetSlice(grad_input.get(), 0, num_grads);
      if (!grad_input) throw python_error();
    }
  }
}

PyObject * THPFunction_do_backward(THPFunction *self, PyObject *args)
{
  try {
    Py_ssize_t num_args = args ? PyTuple_GET_SIZE(args) : 0;
    THPUtils_assert(num_args == 2, "_do_backward expects exactly two arguments");
    PyObject *raw_grad_output = PyTuple_GET_ITEM(args, 0);
    PyObject *retain_variables = PyTuple_GET_ITEM(args, 1);
    if (!PyTuple_Check(raw_grad_output) || !PyBool_Check(retain_variables)) {
      THPUtils_invalidArguments(args, nullptr, "_do_backward", 1, "(tuple, bool)");
      return nullptr;
    }
    auto cdata = self->cdata.lock();
    // In obscure situations, cdata might be nullptr because it's expired.  THAT
    // is an internal error and I'd like to know about it, but since this is
    // all dead soon I didn't bother implementing a sanity check here.  See
    // https://stackoverflow.com/questions/45507041/how-to-check-if-weak-ptr-is-empty-non-assigned
    // for how to do it.
    TORCH_CHECK(cdata,
      "Legacy autograd function attempted to call backward before forward "
      "was called.  This could occur if you manually called _do_backward on Function.  "
      "In any case, this is very naughty!  If you absolutely need this to work, "
      "try porting your code to use non-legacy autograd function, see: "
      "https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd");
    THPUtils_assert(PyTuple_GET_SIZE(raw_grad_output) == cdata->num_inputs(),
                    "%s got an invalid number of gradients (expected %d got %d)",
                    THPUtils_typename(self), cdata->num_inputs(),
                    PyTuple_GET_SIZE(raw_grad_output));

    // Some of the output might have been unused, so we have to allocate
    // zero-filled buffers instead
    Py_INCREF(raw_grad_output);
    THPObjectPtr grad_output(raw_grad_output);
    _prepare_grads(self, grad_output, true);

    // self.backward(*grad_output)
    THPObjectPtr backward_fn(PyObject_GetAttrString((PyObject*)self, "backward"));
    THPUtils_assert(backward_fn.get(), "function %s doesn't implement a required "
        "'backward' method", THPUtils_typename((PyObject*)self));
    THPObjectPtr grad_input(PyObject_CallObject(backward_fn, grad_output.get()));
    if (!grad_input) return nullptr;
    ensure_tuple(grad_input);

    // We allow functions to return more gradients, than there were outputs,
    // if and only if the additional ones are all None
    _trim_grad_input(cdata, self, grad_input);
    int num_grads = PyTuple_GET_SIZE(grad_input.get());
    int num_outputs = cdata->num_outputs();
    THPUtils_assert(num_grads == num_outputs, "%s returned an invalid number of "
        "gradient tensors (expected %d, but got %d)", THPUtils_typename(self),
        num_outputs, num_grads);

    // If any of the remaining grad_inputs are None, zero them.
    _prepare_grads(self, grad_input, false);
    return grad_input.release();

  } catch (python_error& e) {
    return nullptr;
  } catch (std::exception& e) {
    THPUtils_setError(e.what());
    return nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Other methods / attributes
////////////////////////////////////////////////////////////////////////////////

PyObject* THPFunction__register_hook_dict(THPFunction *self, PyObject *_var)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPVariable_Check(_var), "_register_hook_dict expected a variable");
  THPVariable *var = (THPVariable*)_var;
  std::unique_ptr<FunctionPreHook> hook(new PyFunctionPreHook(
      var->backward_hooks, var->cdata.output_nr()));
  auto cdata = self->cdata.lock();
  TORCH_CHECK(cdata,
    "Legacy autograd function had register_hook called before the function was "
    "invoked.  This usage pattern is no longer supported: please call register_hook "
    "AFTER calling your function, or port your code to use non-legacy autograd function, see: "
    "https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd")
  cdata->add_pre_hook(std::move(hook));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_register_hook(THPFunction *self, PyObject *hook)
{
  HANDLE_TH_ERRORS
  auto cdata = self->cdata.lock();
  TORCH_CHECK(cdata,
    "Legacy autograd function had _register_hook called before the function was "
    "invoked.  This usage pattern is no longer supported: please call _register_hook "
    "AFTER calling your function, or port your code to use non-legacy autograd function, see: "
    "https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd")
  return torch::autograd::registerFunctionHook(*cdata, hook);
  END_HANDLE_TH_ERRORS
}

static PyObject *unpack_saved_variables(
    THPFunction *self,
    const std::function<PyObject*(const Variable&)>& unpack_fn)
{
  THPUtils_assert(!self->has_freed_buffers, ERR_BACKWARD_TWICE);
  auto& saved_variables = self->saved_variables;
  if (saved_variables.empty())
    return PyTuple_New(0);

  int num_saved = saved_variables.size();
  THPObjectPtr saved(PyTuple_New(num_saved));
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
  for (int i = 0; i < num_saved; i++) {
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
}

PyObject *THPFunction_saved_tensors(THPFunction *self, void *_unused)
{
  HANDLE_TH_ERRORS
  return unpack_saved_variables(self, [](const Variable& var) {
    return THPVariable_Wrap(var);
  });
  END_HANDLE_TH_ERRORS
}

PyObject *THPFunction_saved_variables(THPFunction *self, void *_unused)
{
  HANDLE_TH_ERRORS
  auto r = PyErr_WarnEx(PyExc_DeprecationWarning,
      "'saved_variables' is deprecated; use 'saved_tensors'", 0);
  if (r != 0) throw python_error();
  return unpack_saved_variables(self, [](const Variable& var) {
    return THPVariable_Wrap(var);
  });
  END_HANDLE_TH_ERRORS
}

PyObject *THPFunction_next_functions(THPFunction *self, void *_unused)
{
  HANDLE_TH_ERRORS
  auto cdata = self->cdata.lock();
  TORCH_CHECK(cdata,
    "Legacy autograd function had next_functions accessed before the function was "
    "invoked.  This doesn't make any sense: we have no idea what the next "
    "functions are, because you haven't actually inserted this grad_fn inside "
    "a graph.  Try invoking your function first before accessing this field.")
  const auto num_outputs = cdata->num_outputs();
  THPObjectPtr result(PyTuple_New(num_outputs));
  if (!result)
    return nullptr;
  for (uint32_t i = 0; i < num_outputs; i++) {
    THPObjectPtr fn_tuple(PyTuple_New(2));
    if (!fn_tuple) return nullptr;
    const auto& edge = cdata->next_edge(i);
    PyObject* fn = functionToPyObject(edge.function);
    if (!fn) return nullptr;
    PyTuple_SET_ITEM(fn_tuple.get(), 0, fn);
    PyTuple_SET_ITEM(fn_tuple.get(), 1, THPUtils_packInt64(edge.input_nr));
    PyTuple_SET_ITEM(result.get(), i, fn_tuple.release());
  }
  return result.release();
  END_HANDLE_TH_ERRORS
}

PyObject *THPFunction_metadata(THPFunction *self, void *_unused)
{
  HANDLE_TH_ERRORS
  auto cdata = self->cdata.lock();
  // The correct way to solve this problem is to stop exposing grad_fn
  // of PyFunctions as THPFunction; instead, we should use THPCppFunction
  // like everyone else.  But this is a BC-breaking change as it would
  // mean that you no longer get the property that grad_fn is a subclass
  // of the autograd function class that you defined in the custom case,
  // so I didn't fix it here.
  TORCH_CHECK(cdata,
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

typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

namespace {

template<PyObject* THPFunction::*ptr>
PyObject* getObject(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  PyObject* value = self->*ptr;
  if (!value) {
    Py_RETURN_NONE;
  }
  Py_INCREF(value);
  return value;
}

template<PyObject* THPFunction::*ptr>
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

template<typename M, M THPFunction::*ptr, PyObject* (*Convert)(long)>
PyObject* getMember(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  return Convert(self->*ptr);
}

template<typename M, M autograd::Node::*ptr, PyObject* (*Convert)(long)>
PyObject* getImplMember(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  return Convert(self->cdata.*ptr);
}

PyObject* getRequiresGrad(PyObject* obj, void* _unused) {
  Py_RETURN_TRUE;
}

}

static struct PyGetSetDef THPFunction_properties[] = {
  {"saved_tensors", (getter)THPFunction_saved_tensors, nullptr, nullptr, nullptr},
  {"saved_variables", (getter)THPFunction_saved_variables, nullptr, nullptr, nullptr},
  {"next_functions", (getter)THPFunction_next_functions, nullptr, nullptr, nullptr},
  {"to_save", &getObject<&THPFunction::to_save>, &setObject<&THPFunction::to_save>, nullptr, nullptr},
  {"non_differentiable", &getObject<&THPFunction::non_differentiable>, &setObject<&THPFunction::non_differentiable>, nullptr, nullptr},
  {"dirty_tensors", &getObject<&THPFunction::dirty_tensors>, &setObject<&THPFunction::dirty_tensors>, nullptr, nullptr},
  {"needs_input_grad", &getObject<&THPFunction::needs_input_grad>, nullptr, nullptr, nullptr},
  {"requires_grad", getRequiresGrad, nullptr, nullptr, nullptr},
  {"metadata", (getter)THPFunction_metadata, nullptr, nullptr, nullptr},
  {nullptr}
};

static struct PyMethodDef THPFunction_methods[] = {
  {(char*)"apply", (PyCFunction)THPFunction_apply, METH_CLASS | METH_VARARGS, nullptr},
  {(char*)"_do_forward", (PyCFunction)THPFunction_do_forward, METH_VARARGS, nullptr},
  {(char*)"_do_backward", (PyCFunction)THPFunction_do_backward, METH_VARARGS, nullptr},
  {(char*)"_register_hook_dict", (PyCFunction)THPFunction__register_hook_dict, METH_O, nullptr},
  {(char*)"register_hook", (PyCFunction)THPFunction_register_hook, METH_O, nullptr},
  {nullptr}
};

PyTypeObject THPFunctionType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._FunctionBase",                    /* tp_name */
  sizeof(THPFunction),                         /* tp_basicsize */
  0,                                           /* tp_itemsize */
  (destructor)THPFunction_dealloc,             /* tp_dealloc */
  0,                                           /* tp_vectorcall_offset */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  nullptr,                                     /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  nullptr,                                     /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  nullptr,                                     /* tp_doc */
  (traverseproc)THPFunction_traverse,          /* tp_traverse */
  (inquiry)THPFunction_clear,                  /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                           /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  THPFunction_methods,                         /* tp_methods */
  nullptr,                                     /* tp_members */
  THPFunction_properties,                      /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  THPFunction_new                              /* tp_new */
};

bool THPFunction_initModule(PyObject *module)
{
  if (PyType_Ready(&THPFunctionType) < 0)
    return false;
  Py_INCREF(&THPFunctionType);
  PyModule_AddObject(module, "_FunctionBase", (PyObject *)&THPFunctionType);
  return true;
}
