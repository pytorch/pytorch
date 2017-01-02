#include <Python.h>
#include <structmember.h>

#include <unordered_map>
#include <unordered_set>
#include <exception>

#include "THP.h"

#ifdef WITH_CUDA
#include "cuda/AutoGPU.h"
#endif

// Throwing this exception means that the python error flags have been already
// set and control should be immediately returned to the interpreter.
class python_error : public std::exception {};

#define THPFunction_assert(condition, ...)                                     \
  if (!(condition)) { THPUtils_setError(__VA_ARGS__); throw python_error(); }


PyObject *THPFunctionClass = NULL;
PyObject *THPStochasticFunctionClass = NULL;

// Traverse and clear are required for supporting Python's GC cycle handling.
static int THPFunction_traverse(THPFunction *self, visitproc visit, void *arg)
{
  Py_VISIT(self->needs_input_grad);
  Py_VISIT(self->backward_hooks);
  for (int i = 0; i < self->num_inputs; i++)
      Py_VISIT(self->previous_functions[i].get());
  if (self->saved_variables) {
    for (unsigned int i = 0; i < self->saved_variables->size(); i++)
      Py_VISIT(std::get<0>(self->saved_variables->at(i)));
  }
  if (self->output_backward_hooks) {
    for (int i = 0; i < self->num_inputs; i++)
      Py_VISIT(self->output_backward_hooks[i].get());
  }

  Py_VISIT(self->to_save);
  Py_VISIT(self->shared_pairs);
  Py_VISIT(self->non_differentiable);
  Py_VISIT(self->dirty_tensors);

  return 0;
}

static int THPFunction_clear(THPFunction *self)
{
  self->num_inputs = 0;
  self->num_outputs = 0;

  Py_CLEAR(self->needs_input_grad);
  Py_CLEAR(self->backward_hooks);

  Py_CLEAR(self->to_save);
  Py_CLEAR(self->shared_pairs);
  Py_CLEAR(self->non_differentiable);
  Py_CLEAR(self->dirty_tensors);

  THPFunctionPtr *previous_functions = self->previous_functions;
  self->previous_functions = NULL;
  delete[] previous_functions;

  auto saved_variables = self->saved_variables;
  self->saved_variables = NULL;
  delete saved_variables;

  auto output_backward_hooks = self->output_backward_hooks;
  self->output_backward_hooks = NULL;
  delete[] output_backward_hooks;

  auto output_info = self->output_info;
  self->output_info = NULL;
  delete output_info;

  return 0;
}

static void THPFunction_dealloc(THPFunction* self)
{
  PyObject_GC_UnTrack(self);
  THPFunction_clear(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *THPFunction_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  THPFunction *self = (THPFunction*)type->tp_alloc(type, 0);
  if (!self)
    return NULL;
  // Python zero-initializes the object memory, so there's no need to initialize
  // most fields
  self->num_outputs = -1;
  return (PyObject*)self;
}

////////////////////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////////////////////

using t2var_type = std::unordered_map<PyObject *, THPVariable *>;

static void _mark_dirty(THPFunction *self, t2var_type &t2var,
        std::unordered_set<PyObject *> &dirty_inputs)
{
  // Increase versions of modified tensors
  if (!self->dirty_tensors) return;

  THPFunction_assert(PyTuple_Check(self->dirty_tensors), "autograd "
      "internal error: dirty_tensors attribute is expected to be a tuple "
      "but is %s", THPUtils_typename(self->dirty_tensors));
  Py_ssize_t num_dirty = PyTuple_GET_SIZE(self->dirty_tensors);
  for (int i = 0; i < num_dirty; i++) {
    PyObject *tensor = PyTuple_GET_ITEM(self->dirty_tensors, i);
    dirty_inputs.insert(tensor);
    THPVariable *variable;
    try {
      variable = t2var.at(tensor);
    } catch (std::out_of_range &e) {
      THPFunction_assert(THPModule_isTensor(tensor), "mark_dirty can "
          "only accept tensors, but argument %d is of type %s", i,
          THPUtils_typename(tensor));
      THPFunction_assert(false, "mark_dirty only accepts input tensors, but "
          "argument %d isn't one", i);
    }
    auto &v_counter = *variable->version_counter;
    THPFunction_assert(v_counter.var_refcnt() == 1, "in-place operations can be "
        "only used on variables that don't share storage with any other "
        "variables, but detected that there are %d objects sharing it",
        v_counter.var_refcnt());
    v_counter++;
  }
  // We're not going to ever need this so let's remove references now
  Py_DECREF(self->dirty_tensors);
  self->dirty_tensors = NULL;
}

static void _wrap_outputs(THPFunction *self, t2var_type &t2var,
    std::unordered_set<PyObject *> &dirty_inputs, PyObject *raw_output,
    PyObject *outputs)
{
  // Wrap outputs in Variables
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(raw_output);
  self->output_info = new std::vector<output_info_type>(num_outputs);
  auto &output_info = *self->output_info;
  for (int i = 0; i < num_outputs; i++) {
    PyObject *output = PyTuple_GET_ITEM(raw_output, i);
    THPVariable *output_var;
    auto it = t2var.find(output);
    if (it == t2var.end()) {
      // A completely new tensor - just wrap it and continue
      output_var = (THPVariable*)THPVariable_New(output, (PyObject*)self, self->requires_grad);
    } else {
      // If one of the outputs was also an input tensor it's a bit more complicated.
      THPVariable *input_var = it->second;
      if (input_var->creator) {
        // If it's not a leaf we want to move it in the graph so backprop
        // will be computed correctly:
        // creator <- variable <- self  ==>  creator <- self <- variable
        Py_INCREF(input_var);
        output_var = input_var;
        Py_DECREF(input_var->creator);
        Py_INCREF(self);
        input_var->creator = (PyObject*)self;
      } else {
        // If the Variable has been changed, we have to move it after the
        // current function to ensure the gradient is computed correctly.
        // There are two cases now:
        // 1. If it requires grad, it is an error, and this will be caught
        // when its _do_backward is called, because it won't be a leaf anymore.
        // Also we'll change its version.
        // 2. If it doesn't require grad, we can safely move it in the graph,
        // because its _do_backward will never be called.
        if (dirty_inputs.count(output) > 0) {
          Py_INCREF(input_var);
          output_var = input_var;
          Py_INCREF(self);
          output_var->creator = (PyObject*)self;
          if (!output_var->requires_grad && self->requires_grad) {
            // Now, there's another subtlety. We move the input in the graph
            // and we change its requires_grad to True. However, remember
            // that we're still holding a reference to is as a previous
            // function. Backward engine will think that it was really a
            // leaf that initialy did require grad and call its _do_backward
            // and that will throw. Because of this, we need to allocate
            // a dummy leaf that doesn't require grad and put it as our
            // previous function.
            output_var->requires_grad = self->requires_grad;
            PyObject* dummy_prev_fn = THPVariable_New(output, NULL, false);
            if (!dummy_prev_fn) throw python_error();
            self->previous_functions[i] = THPFunctionPtr(dummy_prev_fn, 0);
          }
        } else {
          // An input has been returned, but it wasn't modified. It's better
          // not to move the Variable, because there are some legitimate cases
          // where making it non-leaf would break stuff (e.g. broadcast). Also,
          // returning the input Variable is not a good option either,
          // because if someone registers hooks on it, they will fire with grads
          // from all usages, not only from usages of this output. This is why
          // we'll return a copy and join their version counters. This has
          // a side-effect of making in-place ops on any of these Variables an
          // immediate error, but it would be raised anyway once someone
          // calls backward.
          output_var = (THPVariable*)THPVariable_New(output, (PyObject*)self,
                  self->requires_grad);
          if (!output_var) throw python_error();
          output_var->version_counter->join_with(*input_var->version_counter);
        }
      }
    }
    if (!output_var) throw python_error();

    torch::THPVoidTensor *output_obj = (torch::THPVoidTensor*)output_var->data;
    torch::THVoidTensor *output_tensor = output_obj->cdata;
    long ndim = output_tensor->nDimension;
    int device_id = -1;
    THPObjectPtr is_cuda = PyObject_GetAttrString(output_var->data, "is_cuda");
    if (is_cuda.get() == Py_True) {
      THPObjectPtr device_id_obj = PyObject_CallMethod(output_var->data,
          "get_device", "");
      THPFunction_assert(THPUtils_checkLong(device_id_obj), "get_device "
          "should return an int, but got %s", THPUtils_typename(device_id_obj));
      device_id = THPUtils_unpackLong(device_id_obj);
    }
    output_info[i] = std::make_tuple(
      (PyObject*)Py_TYPE(output_var->data),
      device_id,
      std::vector<long>(output_tensor->size, output_tensor->size + ndim)
    );
    t2var[output] = output_var;
    output_var->output_nr = i;
    PyTuple_SET_ITEM(outputs, i, (PyObject*)output_var);
  }
}

static void _save_variables(THPFunction*self, t2var_type &t2var)
{
  if (!self->to_save) return;

  THPFunction_assert(PyTuple_Check(self->to_save), "autograd internal "
      "error: to_save attribute is expected to be a tuple but is %s",
      THPUtils_typename(self->to_save));
  Py_ssize_t num_saved = PyTuple_GET_SIZE(self->to_save);
  self->saved_variables = new std::vector<saved_var_info_type>();
  self->saved_variables->reserve(num_saved);
  for (int i = 0; i < num_saved; i++) {
    PyObject *tensor = PyTuple_GET_ITEM(self->to_save, i);
    if (tensor == Py_None) {
      Py_INCREF(tensor);
      self->saved_variables->emplace_back(tensor, 0, nullptr);
      continue;
    }

    THPVariable *variable;
    try {
      variable = t2var.at(tensor);
    } catch(std::out_of_range &e) {
      THPFunction_assert(THPModule_isTensor(tensor),
          "save_for_backward can only save tensors, but argument %d is of "
          "type %s", i, THPUtils_typename(tensor));
      THPFunction_assert(false, "save_for_backward can only save input or output "
          "tensors, but argument %d doesn't satisfy this condition", i);
    }

    Py_INCREF(tensor);
    self->saved_variables->emplace_back(
      tensor,
      **variable->version_counter,
      std::unique_ptr<THPVariableVersion>(variable->version_counter->new_saved_ref())
    );
  }
  // Free .to_save
  Py_DECREF(self->to_save);
  self->to_save = NULL;
}

static void _join_version_counters(THPFunction *self, t2var_type &t2var)
{
  if (!self->shared_pairs) return;
  THPFunction_assert(PyTuple_Check(self->shared_pairs), "autograd internal "
      "error: shared_pairs attribute is expected to be a tuple but is %s",
      THPUtils_typename(self->shared_pairs));
  Py_ssize_t num_shared = PyTuple_GET_SIZE(self->shared_pairs);
  for (int i = 0; i < num_shared; i++) {
    PyObject *shared_tuple = PyTuple_GET_ITEM(self->shared_pairs, i);
    THPFunction_assert(PyTuple_Check(shared_tuple), "mark_shared_storages "
        "accepts a number of pairs, but one of the arguments is of type %s",
        THPUtils_typename(shared_tuple));
    THPFunction_assert(PyTuple_GET_SIZE(shared_tuple) == 2,
        "mark_shared_storages accepts pairs, but argument %d is a tuple of "
        "%d elements", i, PyTuple_GET_SIZE(shared_tuple));

    // Now we're sure it's really a pair!
    THPVariable *v1, *v2;
    try {
      v1 = t2var.at(PyTuple_GET_ITEM(shared_tuple, 0));
      v2 = t2var.at(PyTuple_GET_ITEM(shared_tuple, 1));
    } catch(std::out_of_range &e) {
      // One tuple items wasn't present in t2var, so there are two cases:
      // 1. it's not a tensor
      // 2. it's not an input nor an output
      PyObject *t1 = PyTuple_GET_ITEM(shared_tuple, 0);
      PyObject *t2 = PyTuple_GET_ITEM(shared_tuple, 1);
      THPFunction_assert(THPModule_isTensor(t1) && THPModule_isTensor(t2),
        "mark_shared_storages accepts pairs of tensors, but one of them "
        "contains %s and %s", THPUtils_typename(t1), THPUtils_typename(t2));
      THPFunction_assert(false, "mark_shared_storages only accepts pairs of input "
          "and output tensors, but argument %d doesn't satify this "
          "condition", i);
    }
    v2->version_counter->join_with(*v1->version_counter);
  }
  // Free .shared_pairs
  Py_DECREF(self->shared_pairs);
  self->shared_pairs = NULL;
}

static void _mark_non_differentiable(THPFunction *self, t2var_type &t2var)
{
  if (!self->non_differentiable) return;

  THPFunction_assert(PyTuple_Check(self->non_differentiable), "autograd "
      "internal error: non_differentiable attribute is expected to be a "
      "tuple but is %s", THPUtils_typename(self->non_differentiable));
  Py_ssize_t num_nondiff = PyTuple_GET_SIZE(self->non_differentiable);
  for (int i = 0; i < num_nondiff; i++) {
    PyObject *t = PyTuple_GET_ITEM(self->non_differentiable, i);
    THPVariable *var;
    try {
      var = t2var.at(t);
      THPFunction_assert(var->creator == (PyObject*)self,
          "mark_non_differentiable only accepts output tensors, but "
          "argument %d isn't an output", i);
    } catch (std::out_of_range &e) {
      THPFunction_assert(THPModule_isTensor(t), "mark_non_differentiable "
          "only accepts tensor arguments, but got %s", THPUtils_typename(t));
      THPFunction_assert(false, "mark_non_differentiable only accepts function "
          "outputs");
    }
    var->requires_grad = 0;
  }
  Py_DECREF(self->non_differentiable);
  self->non_differentiable = NULL;
}

static bool _ensure_tuple(THPObjectPtr& obj)
{
  if (PyTuple_Check(obj.get()))
    return false;

  PyObject *tuple = PyTuple_New(1);
  if (!tuple) throw python_error();
  PyTuple_SET_ITEM(tuple, 0, obj.release());
  obj = tuple;
  return true;
}

PyObject *THPFunction_do_forward(THPFunction *self, PyObject *inputs)
{
  try {
    Py_ssize_t num_inputs = inputs ? PyTuple_GET_SIZE(inputs) : 0;

    // Unpack inputs and check if they require gradients or are volatile
    THPObjectPtr unpacked_inputs = PyTuple_New(num_inputs);
    self->needs_input_grad = PyTuple_New(num_inputs);
    self->requires_grad = false;
    bool is_volatile = false;
    for (int i = 0; i < num_inputs; i++) {
      PyObject *input = PyTuple_GET_ITEM(inputs, i);
      THPUtils_assert(THPVariable_Check(input), "expected a Variable argument, "
          "but got %s", THPUtils_typename(input));
      THPVariable *variable = (THPVariable*)input;

      // Unpack the variable - SET_ITEM steals a reference so INCREF it
      Py_INCREF(variable->data);
      PyTuple_SET_ITEM(unpacked_inputs.get(), i, variable->data);

      // We can't move this to C, because it's going to be accessed from user code.
      PyTuple_SET_ITEM(self->needs_input_grad, i, PyBool_FromLong(variable->requires_grad));

      is_volatile = is_volatile || variable->is_volatile;
      self->requires_grad = self->requires_grad || variable->requires_grad;
    }


    // Now we're ready to call a forward (implemented in Python)
    THPObjectPtr forward_fn = PyObject_GetAttrString((PyObject*)self, "forward");
    THPUtils_assert(forward_fn.get(), "function %s doesn't implement a required "
        "'forward' method", THPUtils_typename((PyObject*)self));
    THPObjectPtr raw_output = PyObject_CallObject(forward_fn, unpacked_inputs);
    if (!raw_output) return NULL;
    // Wrap output in a tuple, if it's not one already
    bool unpack_output = _ensure_tuple(raw_output);
    int num_outputs = PyTuple_GET_SIZE(raw_output.get());


    THPObjectPtr outputs = PyTuple_New(num_outputs);
    if (!outputs) return NULL;
    if (is_volatile) {
      // If one of the inputs is volatile let's take a fast path - we want
      // minimize the overhead of inference
      for (int i = 0; i < num_outputs; i++) {
        PyObject *output = PyTuple_GET_ITEM(raw_output.get(), i);
        THPVariable *output_var = (THPVariable*)THPVariable_NewVolatile(output);
        if (!output_var) return NULL;
        output_var->output_nr = i;
        PyTuple_SET_ITEM(outputs.get(), i, (PyObject*)output_var);
      }
    } else {
      // We're not volatile, so there's a lot of bookkeeping to do...
      self->num_inputs = num_inputs;
      self->num_outputs = num_outputs;
      t2var_type t2var;

      // Save previous functions and initialize t2var map
      self->previous_functions = new THPFunctionPtr[num_inputs];
      for (int i = 0; i < num_inputs; i++) {
        THPVariable *input_var = (THPVariable*)PyTuple_GET_ITEM(inputs, i);
        t2var.emplace(input_var->data, input_var);

        // Save previous function in a helper class (that has a smart pointer to
        // the object and remembers which output did we use).
        PyObject *prev_fn = input_var->creator ? input_var->creator : (PyObject*)input_var;
        Py_INCREF(prev_fn);
        self->previous_functions[i] = THPFunctionPtr(prev_fn, input_var->output_nr);
      }

      std::unordered_set<PyObject *> dirty_inputs;
      _mark_dirty(self, t2var, dirty_inputs);
      _wrap_outputs(self, t2var, dirty_inputs, raw_output, outputs);
      _join_version_counters(self, t2var);
      if (self->requires_grad ||
          PyObject_IsInstance((PyObject*)self, THPStochasticFunctionClass)) {
        _save_variables(self, t2var);
        _mark_non_differentiable(self, t2var);
      }
    }

    // Unpack the output, unless .forward() returned a tuple
    if (unpack_output) {
      PyObject *output = PyTuple_GET_ITEM(outputs.get(), 0);
      Py_INCREF(output);
      return output;
    }

    return outputs.release();

  } catch (python_error& e) {
    return NULL;
  } catch (std::exception& e) {
    THPUtils_setError(e.what());
    return NULL;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Backward
////////////////////////////////////////////////////////////////////////////////

// We need a reference to a smart pointer that will outlive the duration of
// a function call, so that the char* pointer is valid even after it returns
static char* _try_get_name(PyObject *key, THPObjectPtr& tmp) {
#if PY_MAJOR_VERSION == 2
  if (PyString_Check(key)) {
    return PyString_AS_STRING(key);
  }
#else
  if (PyUnicode_Check(key)) {
    tmp = PyUnicode_AsASCIIString(key);
    return PyBytes_AS_STRING(tmp.get());
  }
#endif
  return NULL;
}

#define OPTIONAL_HOOK_NAME                                                     \
  hook_name ? "'" : "",                                                        \
  hook_name ? hook_name : "",                                                  \
  hook_name ? "' " : ""

static void _ensure_correct_hook_result_single(PyObject *original,
    PyObject *returned, PyObject *key)
{
#if PY_MAJOR_VERSION == 2
  static PyObject *IS_SAME_SIZE_NAME = PyString_FromString("is_same_size");
#else
  static PyObject *IS_SAME_SIZE_NAME = PyUnicode_FromString("is_same_size");
#endif
  THPObjectPtr tmp;
  // Check that the type matches
  if(Py_TYPE(original) != Py_TYPE(returned)) {
    char *hook_name = _try_get_name(key, tmp);
    THPUtils_setError("backward hook %s%s%shas changed the type of "
        "grad_input (was %s, but got %s)",
        OPTIONAL_HOOK_NAME,
        THPUtils_typename(original),
        THPUtils_typename(returned)
    );
    throw python_error();
  }

  // Check that the size matches
  THPObjectPtr is_same_size = PyObject_CallMethodObjArgs(original,
      IS_SAME_SIZE_NAME, returned, NULL);
  if(is_same_size.get() != Py_True) {
    char *hook_name = _try_get_name(key, tmp);
    THPUtils_setError("backward hook %s%s%shas changed the size of "
        "grad_input",
        OPTIONAL_HOOK_NAME
    );
    throw python_error();
  }
}

static void _ensure_correct_hook_result(THPObjectPtr& grad_input,
    THPObjectPtr& result, PyObject *key)
{
  THPObjectPtr tmp;
  // Check that the tuple sizes match
  if (PyTuple_GET_SIZE(result.get()) != PyTuple_GET_SIZE(grad_input.get())) {
    char *hook_name = _try_get_name(key, tmp);
    THPUtils_setError("backward hook %s%s%sreturned an incorrect number "
        "of gradients (got %ld, but expected %ld)",
        OPTIONAL_HOOK_NAME,
        PyTuple_GET_SIZE(result.get()),
        PyTuple_GET_SIZE(grad_input.get())
    );
    throw python_error();
  }

  Py_ssize_t size = PyTuple_GET_SIZE(grad_input.get());
  for (int i = 0; i < size; i++) {
    PyObject *original = PyTuple_GET_ITEM(grad_input.get(), i);
    PyObject *returned = PyTuple_GET_ITEM(result.get(), i);
    _ensure_correct_hook_result_single(original, returned, key);
  }
}

static void _call_output_hooks(THPFunction *self, THPObjectPtr& grad_output)
{
  if (!self->output_backward_hooks) return;

  PyObject *key, *value;
  Py_ssize_t pos = 0;
  // We can't reuse the tuple we got, so allocate a new one.
  THPObjectPtr new_grad_output = PyTuple_New(self->num_outputs);
  if (!new_grad_output) throw python_error();

  for (int i = 0; i < self->num_outputs; i++) {
    // Copy grad to a new tuple
    PyObject *old_grad = PyTuple_GET_ITEM(grad_output.get(), i);
    Py_INCREF(old_grad);
    PyTuple_SET_ITEM(new_grad_output.get(), i, old_grad);

    // Make sure that we're really going to operate on a dict
    PyObject *hook_dict = self->output_backward_hooks[i];
    if (!hook_dict) continue;
    THPFunction_assert(PyDict_Check(hook_dict), "backward_hooks "
        "attribute has to be a dictionary");

    while (PyDict_Next(hook_dict, &pos, &key, &value)) {
      THPObjectPtr result = PyObject_CallFunctionObjArgs(value,
          old_grad, NULL);
      if (!result) throw python_error();

      // If the hook returns a something else than None, we treat that as a sign
      // to replace this grad with the return value.
      if (result.get() != Py_None) {
        // Check all possible inconsistencies of the output that we can detect
        // (sizes, types, etc.)
        _ensure_correct_hook_result_single(old_grad, result, key);

        // Replace the old gradient
        PyTuple_SET_ITEM(new_grad_output.get(), i, result.release());
        Py_XDECREF(old_grad);
        old_grad = PyTuple_GET_ITEM(new_grad_output.get(), i);
      }
    }
  }
  grad_output = new_grad_output.release();
}

static void _call_function_hooks(THPFunction *self, THPObjectPtr& grad_input, THPObjectPtr& grad_output)
{
  if (!self->backward_hooks) return;

  PyObject *key, *value;
  Py_ssize_t pos = 0;

  THPFunction_assert(PyDict_Check(self->backward_hooks), "backward_hooks "
      "attribute has to be a dictionary");
  while (PyDict_Next(self->backward_hooks, &pos, &key, &value)) {
    THPObjectPtr result = PyObject_CallFunctionObjArgs(value,
        grad_input.get(), grad_output.get(), NULL);
    if (!result) throw python_error();

    // If the hook returns a something else than None, we treat that as a sign
    // to replace grad_input with its return value.
    if (result.get() != Py_None) {
      // Make sure we're working with a tuple
      _ensure_tuple(result);
      // Check all possible inconsistencies of the output that we can detect
      // (sizes, types, etc.)
      _ensure_correct_hook_result(grad_input, result, key);
      grad_input = result.release();
    }
  }
}

static void _prepare_grad_output(THPFunction *self, THPObjectPtr& raw_grad_output)
{
#ifdef WITH_CUDA
  THCPAutoGPU gpu_guard(-1);
#endif
  int num_grad_output = PyTuple_GET_SIZE(raw_grad_output.get());
  // First, check if any of grad_outputs is None. If not, there's nothing to do
  bool has_none = false;
  for (int i = 0; i < num_grad_output; i++) {
    if (PyTuple_GET_ITEM(raw_grad_output.get(), i) == Py_None) {
      has_none = true;
      break;
    }
  }
  if (!has_none)
      return;

  THPObjectPtr grad_output;
  grad_output = PyTuple_New(num_grad_output);
  if (!grad_output) throw python_error();

  // Look for Nones and replace them with new buffers
  for (int i = 0; i < num_grad_output; i++) {
    PyObject *grad = PyTuple_GET_ITEM(raw_grad_output.get(), i);
    if (grad == Py_None) {
      auto &info = (*self->output_info)[i];
      PyObject *tensor_cls = std::get<0>(info);
#ifdef WITH_CUDA
      gpu_guard.setDevice(std::get<1>(info));
#endif
      std::vector<long> &sizes = std::get<2>(info);
      THPObjectPtr grad_size = THPSize_New(sizes.size(), sizes.data());
      THPObjectPtr new_grad = PyObject_CallFunctionObjArgs(tensor_cls, grad_size.get(), NULL);
      if (!new_grad) throw python_error();
      THPObjectPtr result = PyObject_CallMethod(new_grad.get(), "zero_", "");
      if (!result) throw python_error();
      grad = new_grad.release();
    } else {
      Py_INCREF(grad);
    }
    PyTuple_SET_ITEM(grad_output.get(), i, grad);
  }
  raw_grad_output = grad_output.release();
}

static void _trim_grad_input(THPFunction *self, THPObjectPtr& grad_input)
{
  int num_grads = PyTuple_GET_SIZE(grad_input.get());
  int num_prev_fns = self->num_inputs;
  if (num_grads > num_prev_fns) {
    // Check that all extra grads are none
    bool all_none = true;
    for (int i = num_prev_fns; i < num_grads; i++) {
      all_none = (PyTuple_GET_ITEM(grad_input.get(), i) == Py_None);
      if (!all_none) break;
    }
    // If yes, slice the tuple
    if (all_none) {
      num_grads = num_prev_fns;
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
      THPUtils_invalidArguments(args, "_do_backward", 1, "(tuple, bool)");
      return NULL;
    }

    // Some of the output might have been unused, so we have to allocate
    // zero-filled buffers instead
    Py_INCREF(raw_grad_output);
    THPObjectPtr grad_output = raw_grad_output;
    _prepare_grad_output(self, grad_output);

    // Call output hooks (this can modify grad_output!)
    _call_output_hooks(self, grad_output);

    // self.backward(*grad_output)
    THPObjectPtr backward_fn = PyObject_GetAttrString((PyObject*)self, "backward");
    THPUtils_assert(backward_fn.get(), "function %s doesn't implement a required "
        "'backward' method", THPUtils_typename((PyObject*)self));
    THPObjectPtr grad_input = PyObject_CallObject(backward_fn, grad_output.get());
    if (!grad_input) return NULL;
    _ensure_tuple(grad_input);

    // We allow functions to return more gradients, than there were outputs,
    // if and only if the additional ones are all None
    _trim_grad_input(self, grad_input);
    int num_grads = PyTuple_GET_SIZE(grad_input.get());
    int num_prev_fns = self->num_inputs;
    THPUtils_assert(num_grads == num_prev_fns, "%s returned an invalid number of "
        "gradient tensors (expected %d, but got %d)", THPUtils_typename(self),
        num_prev_fns, num_grads);

    // Call function hooks (this can modify grad_input!)
    _call_function_hooks(self, grad_input, grad_output);

    // Free buffers only if they're not going to be ever used again
    if (retain_variables == Py_False) {
      delete self->saved_variables;
      self->saved_variables = nullptr;
      self->has_freed_buffers = 1;
    }

    return grad_input.release();

  } catch (python_error& e) {
    return NULL;
  } catch (std::exception& e) {
    THPUtils_setError(e.what());
    return NULL;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Other methods / attributes
////////////////////////////////////////////////////////////////////////////////

PyObject* THPFunction__register_hook_dict(THPFunction *self, PyObject *_var)
{
  THPUtils_assert(THPVariable_Check(_var), "_register_hook_dict expected a variable");
  THPVariable *var = (THPVariable*)_var;

  if (!self->output_backward_hooks)
    self->output_backward_hooks = new THPObjectPtr[self->num_inputs];
  Py_INCREF(var->backward_hooks);
  self->output_backward_hooks[var->output_nr] = var->backward_hooks;

  Py_RETURN_NONE;
}


PyObject *THPFunction_saved_tensors(THPFunction *self, void *_unused)
{
  THPUtils_assert(!self->has_freed_buffers, "Trying to backward through the "
      "graph second time, but the buffers have already been freed. Please "
      "specify retain_variables=True when calling backward for the first time.");
  if (!self->saved_variables)
    return PyTuple_New(0);

  int num_saved = self->saved_variables->size();
  THPObjectPtr saved_tensors = PyTuple_New(num_saved);
  if (!saved_tensors)
    return NULL;
  for (int i = 0; i < num_saved; i++) {
    saved_var_info_type &tuple = (*self->saved_variables)[i];
    PyObject *tensor = std::get<0>(tuple);
    if (tensor != Py_None) {
      int expected_version = std::get<1>(tuple);
      int current_version = **(std::get<2>(tuple));
      THPUtils_assert(expected_version == current_version, "one of the variables "
          "needed for gradient computation has been modified by an "
          "inplace operation");
    }
    Py_INCREF(tensor);
    PyTuple_SET_ITEM(saved_tensors.get(), i, tensor);
  }
  return saved_tensors.release();
}

PyObject *THPFunction_previous_functions(THPFunction *self, void *_unused)
{
  THPObjectPtr previous_functions = PyTuple_New(self->num_inputs);
  if (!previous_functions)
    return NULL;
  for (int i = 0; i < self->num_inputs; i++) {
    THPObjectPtr fn_tuple = PyTuple_New(2);
    if (!fn_tuple)
      return NULL;
    Py_INCREF(self->previous_functions[i].get());
    PyTuple_SET_ITEM(fn_tuple.get(), 0, self->previous_functions[i].get());
    PyTuple_SET_ITEM(fn_tuple.get(), 1, PyInt_FromLong(self->previous_functions[i].output_nr));
    PyTuple_SET_ITEM(previous_functions.get(), i, fn_tuple.release());
  }
  return previous_functions.release();
}


typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

static struct PyGetSetDef THPFunction_properties[] = {
  {"saved_tensors", (getter)THPFunction_saved_tensors, NULL, NULL, NULL},
  {"previous_functions", (getter)THPFunction_previous_functions, NULL, NULL, NULL},
  {NULL}
};

static struct PyMemberDef THPFunction_members[] = {
  {(char*)"_backward_hooks", T_OBJECT, offsetof(THPFunction, backward_hooks), 0, NULL},
  {(char*)"to_save", T_OBJECT, offsetof(THPFunction, to_save), 0, NULL},
  {(char*)"shared_pairs", T_OBJECT, offsetof(THPFunction, shared_pairs), 0, NULL},
  {(char*)"non_differentiable", T_OBJECT, offsetof(THPFunction, non_differentiable), 0, NULL},
  {(char*)"dirty_tensors", T_OBJECT, offsetof(THPFunction, dirty_tensors), 0, NULL},
  {(char*)"needs_input_grad", T_OBJECT, offsetof(THPFunction, needs_input_grad), 0, NULL},
  {(char*)"requires_grad", T_BOOL, offsetof(THPFunction, requires_grad), 0, NULL},
  {(char*)"num_inputs", T_INT, offsetof(THPFunction, num_inputs), 0, NULL},
  {(char*)"num_outputs", T_INT, offsetof(THPFunction, num_outputs), 0, NULL},
  {NULL}
};

static struct PyMethodDef THPFunction_methods[] = {
  {(char*)"_do_forward", (PyCFunction)THPFunction_do_forward, METH_VARARGS, NULL},
  {(char*)"_do_backward", (PyCFunction)THPFunction_do_backward, METH_VARARGS, NULL},
  {(char*)"_register_hook_dict", (PyCFunction)THPFunction__register_hook_dict, METH_O, NULL},
  {NULL}
};

PyTypeObject THPFunctionType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._FunctionBase",              /* tp_name */
  sizeof(THPFunction),                   /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPFunction_dealloc,       /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  NULL,                                  /* tp_doc */
  (traverseproc)THPFunction_traverse,    /* tp_traverse */
  (inquiry)THPFunction_clear,            /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPFunction_methods,                   /* tp_methods */
  THPFunction_members,                   /* tp_members */
  THPFunction_properties,                /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPFunction_new                        /* tp_new */
};

bool THPFunction_initModule(PyObject *module)
{
  if (PyType_Ready(&THPFunctionType) < 0)
    return false;
  Py_INCREF(&THPFunctionType);
  PyModule_AddObject(module, "_FunctionBase", (PyObject *)&THPFunctionType);
  return true;
}
