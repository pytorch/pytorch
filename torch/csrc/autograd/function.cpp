#include <Python.h>
#include <structmember.h>

#include <unordered_map>

#include "THP.h"

#ifdef WITH_CUDA
#include "cuda/AutoGPU.h"
#endif

PyObject *THPFunctionClass = NULL;

static void THPFunction_dealloc(THPFunction* self)
{
  PyObject_GC_UnTrack(self);
  self->num_inputs = 0;
  self->num_outputs = 0;

  Py_XDECREF(self->needs_input_grad);
  Py_XDECREF(self->saved_variables);
  Py_XDECREF(self->backward_hooks);

  Py_XDECREF(self->to_save);
  Py_XDECREF(self->shared_pairs);
  Py_XDECREF(self->non_differentiable);
  Py_XDECREF(self->dirty_tensors);

  THPFunctionPtr *previous_functions = self->previous_functions;
  self->previous_functions = NULL;
  delete[] previous_functions;
  delete self->output_info;

  Py_TYPE(self)->tp_free((PyObject*)self);
}

// Traverse and clear are required for supporting Python's GC cycle handling.
static int THPFunction_traverse(THPFunction *self, visitproc visit, void *arg)
{
  Py_VISIT(self->needs_input_grad);
  Py_VISIT(self->saved_variables);
  Py_VISIT(self->backward_hooks);
  for (int i = 0; i < self->num_inputs; i++)
      Py_VISIT(self->previous_functions[i].get());

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
  Py_CLEAR(self->saved_variables);
  Py_CLEAR(self->backward_hooks);

  Py_CLEAR(self->to_save);
  Py_CLEAR(self->shared_pairs);
  Py_CLEAR(self->non_differentiable);
  Py_CLEAR(self->dirty_tensors);

  THPFunctionPtr *previous_functions = self->previous_functions;
  self->previous_functions = NULL;
  delete[] previous_functions;

  return 0;
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

using t2var_type = std::unordered_map<PyObject *, THPVariable *>;

static bool _mark_dirty(THPFunction *self, t2var_type &t2var)
{
  // Increase versions of modified tensors
  if (self->dirty_tensors) {
    THPUtils_assertRet(false, PyTuple_Check(self->dirty_tensors), "autograd "
        "internal error: dirty_tensors attribute is expected to be a tuple "
        "but is %s", THPUtils_typename(self->dirty_tensors));
    Py_ssize_t num_dirty = PyTuple_GET_SIZE(self->dirty_tensors);
    for (int i = 0; i < num_dirty; i++) {
      PyObject *tensor = PyTuple_GET_ITEM(self->dirty_tensors, i);
      THPVariable *variable;
      try {
        variable = t2var.at(tensor);
      } catch (std::out_of_range &e) {
        THPUtils_assertRet(false, THPModule_isTensor(tensor), "mark_dirty can "
            "only accept tensors, but argument %d is of type %s", i,
            THPUtils_typename(tensor));
        THPUtils_setError("mark_dirty only accepts input tensors, but "
            "argument %d isn't one", i);
        return false;
      }
      auto &v_counter = *variable->version_counter;
      THPUtils_assert(v_counter.refcnt() == 1, "in-place operations can be "
          "only used on variables that don't share storage with any other "
          "variables, but detected that there are %d objects sharing it",
          v_counter.refcnt());
      v_counter++;
    }
    // We're not going to ever need this so let's remove references now
    Py_DECREF(self->dirty_tensors);
    self->dirty_tensors = NULL;
  }
  return true;
}

static bool _wrap_outputs(THPFunction *self, t2var_type &t2var,
    PyObject *raw_output, PyObject *outputs)
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
        // If it's a leaf it's not as simple. Leaves will raise an error in
        // backward if they've been changed, or they're no longer leaves. In
        // some cases (e.g. broadcast) it's perfectly valid to return the same
        // tensor untouched, so instead of moving it we're going to create a
        // copy and join their version counters. This works for broadcast,
        // and if the use wasn't valid we'll still detect an error, because
        // the leaf will have a version != 0.
        output_var = (THPVariable*)THPVariable_New(output, (PyObject*)self, self->requires_grad);
        if (!output_var) return false;
        output_var->version_counter->join_with(*input_var->version_counter);
      }
    }
    if (!output_var)
      return false;

    torch::THPVoidTensor *output_obj = (torch::THPVoidTensor*)output_var->data;
    torch::THVoidTensor *output_tensor = output_obj->cdata;
    long ndim = output_tensor->nDimension;
    int device_id = -1;
    THPObjectPtr is_cuda = PyObject_GetAttrString(output_var->data, "is_cuda");
    if (is_cuda.get() == Py_True) {
      THPObjectPtr device_id_obj = PyObject_CallMethod(output_var->data,
          "get_device", "");
      THPUtils_assertRet(false, THPUtils_checkLong(device_id_obj), "get_device "
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
  return true;
}

static bool _save_variables(THPFunction*self, t2var_type &t2var)
{
  // TODO: this can be stored without using python types
  if (self->to_save) {
    THPUtils_assertRet(false, PyTuple_Check(self->to_save), "autograd internal "
        "error: to_save attribute is expected to be a tuple but is %s",
        THPUtils_typename(self->to_save));
    Py_ssize_t num_saved = PyTuple_GET_SIZE(self->to_save);
    self->saved_variables = PyTuple_New(num_saved);
    if (!self->saved_variables) return false;
    for (int i = 0; i < num_saved; i++) {
      PyObject *tensor = PyTuple_GET_ITEM(self->to_save, i);
      THPVariable *variable;
      try {
        variable = t2var.at(tensor);
      } catch(std::out_of_range &e) {
        THPUtils_assertRet(false, THPModule_isTensor(tensor),
            "save_for_backward can only save tensors, but argument %d is of "
            "type %s", i, THPUtils_typename(tensor));
        THPUtils_setError("save_for_backward can only save input or output "
            "tensors, but argument %d doesn't satisfy this condition", i);
        return false;
      }

      PyObject *tuple = PyTuple_New(2);
      if (!tuple)
        return false;
      Py_INCREF(variable);
      PyTuple_SET_ITEM(tuple, 0, (PyObject*)variable);
      PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(**variable->version_counter));
      PyTuple_SET_ITEM(self->saved_variables, i, tuple);
    }
    // Free .to_save
    Py_DECREF(self->to_save);
    self->to_save = NULL;
  }
  return true;
}

static bool _join_version_counters(THPFunction *self, t2var_type &t2var)
{
  if (self->shared_pairs) {
    THPUtils_assertRet(false, PyTuple_Check(self->shared_pairs), "autograd internal "
        "error: shared_pairs attribute is expected to be a tuple but is %s",
        THPUtils_typename(self->shared_pairs));
    Py_ssize_t num_shared = PyTuple_GET_SIZE(self->shared_pairs);
    for (int i = 0; i < num_shared; i++) {
      PyObject *shared_tuple = PyTuple_GET_ITEM(self->shared_pairs, i);
      THPUtils_assertRet(false, PyTuple_Check(shared_tuple), "mark_shared_storages "
          "accepts a number of pairs, but one of the arguments is of type %s",
          THPUtils_typename(shared_tuple));
      THPUtils_assertRet(false, PyTuple_GET_SIZE(shared_tuple) == 2,
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
        THPUtils_assertRet(false, THPModule_isTensor(t1) && THPModule_isTensor(t2),
          "mark_shared_storages accepts pairs of tensors, but one of them "
          "contains %s and %s", THPUtils_typename(t1), THPUtils_typename(t2));
        THPUtils_setError("mark_shared_storages only accepts pairs of input "
            "and output tensors, but argument %d doesn't satify this "
            "condition", i);
        return false;
      }
      v2->version_counter->join_with(*v1->version_counter);
    }
    // Free .shared_pairs
    Py_DECREF(self->shared_pairs);
    self->shared_pairs = NULL;
  }
  return true;
}

static bool _mark_non_differentiable(THPFunction *self, t2var_type &t2var)
{
  if (self->non_differentiable) {
    THPUtils_assertRet(false, PyTuple_Check(self->non_differentiable), "autograd "
        "internal error: non_differentiable attribute is expected to be a "
        "tuple but is %s", THPUtils_typename(self->non_differentiable));
    Py_ssize_t num_nondiff = PyTuple_GET_SIZE(self->non_differentiable);
    for (int i = 0; i < num_nondiff; i++) {
      PyObject *t = PyTuple_GET_ITEM(self->non_differentiable, i);
      THPVariable *var;
      try {
        var = t2var.at(t);
        THPUtils_assertRet(false, var->creator == (PyObject*)self,
            "mark_non_differentiable only accepts output tensors, but "
            "argument %d isn't an output", i);
      } catch (std::out_of_range &e) {
        THPUtils_assertRet(false, THPModule_isTensor(t), "mark_non_differentiable "
            "only accepts tensor arguments, but got %s", THPUtils_typename(t));
        THPUtils_setError("mark_non_differentiable only accepts function "
            "outputs");
        return false;
      }
      var->requires_grad = 0;
    }
    Py_DECREF(self->non_differentiable);
    self->non_differentiable = NULL;
  }
  return true;
}


PyObject *THPFunction_do_forward(THPFunction *self, PyObject *inputs)
{
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
  if (!raw_output)
    return NULL;
  // Wrap output in a tuple, if it's not one already
  if (!PyTuple_Check(raw_output.get())) {
    PyObject *tuple = PyTuple_New(1);
    if (!tuple)
      return NULL;
    PyTuple_SET_ITEM(tuple, 0, raw_output.release());
    raw_output = tuple;
  }
  int num_outputs = PyTuple_GET_SIZE(raw_output.get());


  THPObjectPtr outputs = PyTuple_New(num_outputs);
  if (!outputs)
    return NULL;
  if (is_volatile) {
    // If one of the inputs is volatile let's take a fast path - we want
    // minimize the overhead of inference
    for (int i = 0; i < num_outputs; i++) {
      PyObject *output = PyTuple_GET_ITEM(raw_output.get(), i);
      THPVariable *output_var = (THPVariable*)THPVariable_NewVolatile(output);
      if (!output_var)
        return NULL;
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
      // the object and remembers which output did we use.
      PyObject *prev_fn = input_var->creator ? input_var->creator : (PyObject*)input_var;
      Py_INCREF(prev_fn);
      self->previous_functions[i] = THPFunctionPtr(prev_fn, input_var->output_nr);
    }

    if (!_mark_dirty(self, t2var))
      return NULL;
    if (!_wrap_outputs(self, t2var, raw_output, outputs))
      return NULL;
    if (!_join_version_counters(self, t2var))
      return NULL;
    if (!_save_variables(self, t2var))
      return NULL;
    if (!_mark_non_differentiable(self, t2var))
      return NULL;
  }

  if (num_outputs == 1) {
    PyObject *output = PyTuple_GET_ITEM(outputs.get(), 0);
    Py_INCREF(output);
    return output;
  }

  return outputs.release();
}

PyObject * THPFunction_do_backward(THPFunction *self, PyObject *args)
{
  Py_ssize_t num_args = args ? PyTuple_GET_SIZE(args) : 0;
  THPUtils_assert(num_args == 2, "_do_backward expects exactly two arguments");
  PyObject *raw_grad_output = PyTuple_GET_ITEM(args, 0);
  PyObject *retain_variables = PyTuple_GET_ITEM(args, 1);
  if (!PyTuple_Check(raw_grad_output) || !PyBool_Check(retain_variables)) {
    THPUtils_invalidArguments(args, "_do_backward", 1, "(tuple, bool)");
    return NULL;
  }

  int num_grad_output = PyTuple_GET_SIZE(raw_grad_output);
  THPObjectPtr grad_output = PyTuple_New(num_grad_output);
  if (!grad_output) return NULL;
#ifdef WITH_CUDA
  THCPAutoGPU gpu_guard(-1);
#endif
  for (int i = 0; i < num_grad_output; i++) {
    PyObject *grad = PyTuple_GET_ITEM(raw_grad_output, i);
    // If there's no gradient we have to allocate a buffer ourselves
    if (grad == Py_None) {
      auto &info = (*self->output_info)[i];
      PyObject *tensor_cls = std::get<0>(info);
#ifdef WITH_CUDA
      gpu_guard.setDevice(std::get<1>(info));
#endif
      std::vector<long> &sizes = std::get<2>(info);
      THPObjectPtr grad_size = THPSize_New(sizes.size(), sizes.data());
      THPObjectPtr new_grad = PyObject_CallFunctionObjArgs(tensor_cls, grad_size.get(), NULL);
      if (!new_grad) return NULL;
      THPObjectPtr result = PyObject_CallMethod(new_grad.get(), "zero_", "");
      if (!result) return NULL;
      grad = new_grad.release();
    } else {
      Py_INCREF(grad);
    }
    PyTuple_SET_ITEM(grad_output.get(), i, grad);
  }

  THPObjectPtr backward_fn = PyObject_GetAttrString((PyObject*)self, "backward");
  THPUtils_assert(backward_fn.get(), "function %s doesn't implement a required "
      "'backward' method", THPUtils_typename((PyObject*)self));
  THPObjectPtr grad_input = PyObject_CallObject(backward_fn, grad_output.get());
  if (!grad_input)
    return NULL;

  if (!PyTuple_Check(grad_input)) {
    PyObject *grad_tuple = PyTuple_New(1);
    if (!grad_tuple)
      return NULL;
    PyTuple_SET_ITEM(grad_tuple, 0, grad_input.release());
    grad_input = grad_tuple;
  }
  int num_grads = PyTuple_GET_SIZE(grad_input.get());
  int num_prev_fns = self->num_inputs;

  THPUtils_assert(num_grads == num_prev_fns, "%s returned an invalid number of "
      "gradient tensors (expected %d, but got %d)", THPUtils_typename(self),
      num_prev_fns, num_grads);

  if (self->backward_hooks) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    THPUtils_assert(PyDict_Check(self->backward_hooks), "backward_hooks "
        "attribute has to be a dictionary");
    while (PyDict_Next(self->backward_hooks, &pos, &key, &value)) {
      THPObjectPtr result = PyObject_CallFunctionObjArgs(value,
          grad_input.get(), grad_output.get(), NULL);
      if (!result)
        return NULL;
    }
  }

  if (retain_variables == Py_False) {
    Py_XDECREF(self->saved_variables);
    self->saved_variables = NULL;
    self->has_freed_buffers = 1;
  }

  return grad_input.release();
}

PyObject *THPFunction_saved_tensors(THPFunction *self, void *_unused)
{
  THPUtils_assert(!self->has_freed_buffers, "Trying to backward through the "
      "graph second time, but the buffers have already been freed. Please "
      "specify retain_variables=True when calling backward for the first time.");
  if (!self->saved_variables)
    return PyTuple_New(0);

  Py_ssize_t num_saved = PyTuple_GET_SIZE(self->saved_variables);
  THPObjectPtr saved_tensors = PyTuple_New(num_saved);
  if (!saved_tensors)
    return NULL;
  for (int i = 0; i < num_saved; i++) {
    PyObject *tuple = PyTuple_GET_ITEM(self->saved_variables, i);
    long expected_version = THPUtils_unpackLong(PyTuple_GET_ITEM(tuple, 1));
    THPVariable *variable = (THPVariable*)PyTuple_GET_ITEM(tuple, 0);
    int current_version = **variable->version_counter;
    THPUtils_assert(expected_version == current_version, "one of the variables "
        "needed for gradient computation has been modified by an "
        "inplace operation");
    Py_INCREF(variable->data);
    PyTuple_SET_ITEM(saved_tensors.get(), i, variable->data);
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
  {(char*)"saved_variables", T_OBJECT, offsetof(THPFunction, saved_variables), 0, NULL},
  {(char*)"backward_hooks", T_OBJECT, offsetof(THPFunction, backward_hooks), 0, NULL},
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
