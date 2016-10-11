#include <Python.h>
#include <structmember.h>

#include <vector>
#include <unordered_map>
#include <deque>
#include <set>

#include "THP.h"

PyObject *THPEngineClass = NULL;

// used for topological sort
using dependencies_type = std::unordered_map<THPFunction *, int>;
// stores gradient buffers
using grad_list_type = std::vector<THPObjectPtr>;
// used for need_copy set (to ensure correct gradient buffering)
using buffer_set_type = std::set<std::pair<size_t, int>>;
// gradient buffer - a list of gradient tensors + id
struct grad_buffer_type: public grad_list_type {
  template<typename... Args>
  grad_buffer_type(size_t buffer_id, Args&&... args):
      grad_list_type(std::forward<Args>(args)...),
      buffer_id(buffer_id) {};
  grad_buffer_type(grad_buffer_type &&other):
      grad_list_type(std::move(other)),
      buffer_id(other.buffer_id) {};
  grad_buffer_type& operator=(grad_buffer_type &&other) {
      grad_list_type::operator=(std::move(other));
      buffer_id = other.buffer_id;
      return *this;
  };

  size_t buffer_id;
};

// Computes graph dependencies (using a super simple topological sort)
dependencies_type THPEngine_compute_dependencies(THPFunction *function)
{
  dependencies_type dependencies;
  std::set<THPFunction *> seen;
  std::vector<THPFunction *> queue = {function};
  while (queue.size() > 0) {
    THPFunction *fn = queue.back(); queue.pop_back();
    for (int i = 0; i < fn->num_inputs; i++) {
      THPFunction *prev_fn = (THPFunction*)fn->previous_functions[i].get();
      // We can ignore variables (their backprop is called every time we have
      // gradient ready) and functions that don't require gradient.
      if (THPVariable_Check((PyObject*)prev_fn) || !prev_fn->requires_grad)
        continue;
      dependencies[prev_fn] += 1;
      if (seen.count(prev_fn) == 0) {
        seen.insert(prev_fn);
        queue.push_back(prev_fn);
      }
    }
  }
  return dependencies;
}

// Frees backward dependency and returns true if prev_fn is ready for backward
bool THPEngine_free_backward_dependency(dependencies_type &dependencies,
    THPFunction *prev_fn)
{
  if (--dependencies[prev_fn] == 0) {
    dependencies.erase(prev_fn);
    return true;
  }
  return false;
}

// Accumulates d_prev_fn gradient tensor into output_idx position of prev_grad buffer
bool THPEngine_add_grad(buffer_set_type &need_copy, grad_buffer_type &prev_grad,
    int output_nr, PyObject *d_prev_fn)
{
  // TODO: we should probably clean up need_copy, because most tensors will
  // probably never hit the else clause
  auto set_key = std::make_pair(prev_grad.buffer_id, output_nr);
  if (!prev_grad[output_nr]) {
    Py_INCREF(d_prev_fn);
    prev_grad[output_nr] = d_prev_fn;
    need_copy.insert(set_key);
  } else {
    PyObject *grad_tensor = prev_grad[output_nr];
    if (need_copy.count(set_key) != 0) {
      grad_tensor = PyObject_CallMethod(grad_tensor, "clone", "");
      if (!grad_tensor)
          return false;
      need_copy.erase(set_key);
      prev_grad[output_nr] = grad_tensor;
    }
    THPObjectPtr result = PyObject_CallMethod(grad_tensor, "add_", "O", d_prev_fn);
    if (!result)
        return false;
  }
  return true;
}

// Main backward function
PyObject *THPEngine_run_backward(THPEngine *self, PyObject *args, PyObject *kwargs)
{
  THPVariable *variable = NULL;
  PyObject *grad_variable = NULL;
  unsigned char retain_variables = 0;
  size_t next_buf_id = 0;
  const char *accepted_kwargs[] = {"variable", "grad_variable",
      "retain_variables", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOb", (char**)accepted_kwargs,
        &variable, &grad_variable, &retain_variables))
    return NULL;
  PyObject *retain_variables_obj = retain_variables ? Py_True : Py_False;

  // If someone calls .backward() on a leaf, it's simple...
  if (variable->creator == NULL) {
    THPObjectPtr result = PyObject_CallMethod((PyObject*)variable,
            "_do_backward", "(O)O", grad_variable, retain_variables_obj);
    Py_RETURN_NONE;
  }

  std::deque<std::pair<THPFunction *, grad_buffer_type>> ready;
  std::unordered_map<THPFunction *, grad_buffer_type> not_ready;
  buffer_set_type need_copy;

  // Initialize the queue
  grad_buffer_type buf(next_buf_id++, 1);
  Py_INCREF(grad_variable);
  buf[0] = grad_variable;
  ready.emplace_front((THPFunction*)variable->creator, std::move(buf));

  dependencies_type dependencies = THPEngine_compute_dependencies((THPFunction*)variable->creator);

  while (ready.size() > 0) {
    std::pair<THPFunction *, grad_buffer_type> ready_pair =
        std::move(ready.back()); ready.pop_back();
    THPFunction *fn = ready_pair.first;
    grad_buffer_type &fn_grad_buffer = ready_pair.second;

    // Prepare a tuple for a call to _do_backward
    THPObjectPtr grad_tuple = PyTuple_New(fn_grad_buffer.size());
    if (!grad_tuple) return NULL;
    for (unsigned int i = 0; i < fn_grad_buffer.size(); i++) {
      // TODO: allocate correctly sized zero buffer
      THPUtils_assert(fn_grad_buffer[i], "error no grad buffer - this will be "
              "fixed in upcoming releases!");
      PyTuple_SET_ITEM(grad_tuple.get(), i, fn_grad_buffer[i].release());
    }

    // Call _do_backward and make sure grad_input is sound
    THPObjectPtr grad_input = PyObject_CallMethod((PyObject*)fn, "_do_backward",
        "OO", grad_tuple.get(), retain_variables_obj);
    if (!grad_input)
      return NULL;
    THPUtils_assert(PyTuple_Check(grad_input), "error, _do_backward should "
            "return a tuple, but got %s", THPUtils_typename(grad_input));
    int num_grads = PyTuple_GET_SIZE(grad_input.get());

    // Process tensors inside grad_input
    for (int i = 0; i < num_grads; i++) {
      PyObject *prev_obj = fn->previous_functions[i].get();
      PyObject *grad_prev = PyTuple_GET_ITEM(grad_input.get(), i);

      // A shortcut for variables - there's no need to buffer gradients for them
      // as their _do_backward is super fast (and we can save memory).
      // TODO: this might call leaf variable hooks multiple times
      if (THPVariable_Check(prev_obj)) {
        THPVariable *prev_var = (THPVariable*)prev_obj;
        if (prev_var->requires_grad) {
          THPObjectPtr ret = PyObject_CallMethod(prev_obj, "_do_backward",
              "(O)O", grad_prev, retain_variables_obj);
          if (!ret) return NULL;
        }
        continue;
      }

      // No need to do any work for functions that don't require gradients
      THPFunction *prev_fn = (THPFunction*)prev_obj;
      if (!prev_fn->requires_grad)
        continue;

      // Check if the function is ready for backward and see if it has any
      // buffers allocated
      int output_idx = fn->previous_functions[i].output_nr;
      bool is_ready = THPEngine_free_backward_dependency(dependencies, prev_fn);
      auto not_ready_it = not_ready.find(prev_fn);
      if (is_ready) {
        // this is only a temporary, so no need for a correct id
        grad_buffer_type prev_buffer(-1);
        if (not_ready_it == not_ready.end()) {
          // The function is ready and no buffers have been allocated for it.
          prev_buffer = grad_buffer_type(next_buf_id++, 1);
          Py_INCREF(grad_prev);
          prev_buffer[output_idx] = grad_prev;
        } else {
          // The function is ready and it already has a buffer allocated.
          prev_buffer = std::move(not_ready_it->second);
          not_ready.erase(not_ready_it);
          if (!THPEngine_add_grad(need_copy, prev_buffer, output_idx, grad_prev))
              return NULL;
        }
        // Put the function into the ready queue.
        ready.emplace_front(prev_fn, std::move(prev_grad));
      } else {
        // Allocate a buffer if necessary
        if (not_ready_it == not_ready.end()) {
          int num_prev_fn_outputs = prev_fn->num_outputs;
          std::tie(not_ready_it, std::ignore) =
              not_ready.emplace(prev_fn, grad_buffer_type(next_buf_id++, num_prev_fn_outputs));
        }
        // Accumulate the gradient into the buffer
        grad_buffer_type &grad_buffer = not_ready_it->second;
        if (!THPEngine_add_grad(need_copy, grad_buffer, output_idx, grad_prev))
            return NULL;
      }
    }
  }

  Py_RETURN_NONE;
}

PyObject *THPEngine_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  return type->tp_alloc(type, 0);
}

static struct PyMethodDef THPEngine_methods[] = {
  {(char*)"run_backward", (PyCFunction)THPEngine_run_backward, METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL}
};


PyTypeObject THPEngineType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._EngineBase",                /* tp_name */
  sizeof(THPEngine),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
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
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPEngine_methods,                     /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPEngine_new                          /* tp_new */
};


bool THPEngine_initModule(PyObject *module)
{
  if (PyType_Ready(&THPEngineType) < 0)
    return false;
  Py_INCREF(&THPEngineType);
  PyModule_AddObject(module, "_ImperativeEngine", (PyObject *)&THPEngineType);
  return true;
}

