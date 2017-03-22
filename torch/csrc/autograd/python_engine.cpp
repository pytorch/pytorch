#include "torch/csrc/autograd/python_engine.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/utils/auto_gil.h"

using namespace torch::autograd;

struct THPEngine {
    PyObject_HEAD
};

struct PythonEngine : public Engine {
  virtual void thread_main(ReadyQueue& queue) override {
    // Create a PyThreadState, but release the GIL. This lets AutoGIL calls
    // inside thread_main acquire the GIL without having to create a new
    // PyThreadState each time.
    AutoGIL gil;
    AutoNoGIL no_gil;
    Engine::thread_main(queue);
  }

  virtual void thread_on_exception(FunctionTask& task, std::exception& e) override {
    auto python_err = dynamic_cast<python_error*>(&e);
    if (python_err) {
      python_err->persist();
    }
    Engine::thread_on_exception(task, e);
  }
};

static PythonEngine engine;

PyObject *THPEngineClass = NULL;

// Main backward function
PyObject *THPEngine_run_backward(THPEngine *self, PyObject *args, PyObject *kwargs)
{
  PyObject *variables = NULL;
  PyObject *grad_variables = NULL;
  unsigned char retain_variables = 0;
  const char *accepted_kwargs[] = {"variables", "grad_variables",
      "retain_variables", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOb", (char**)accepted_kwargs,
        &variables, &grad_variables, &retain_variables))
    return NULL;
  PyObject *retain_variables_obj = retain_variables ? Py_True : Py_False;

  THPUtils_assert(retain_variables_obj == Py_True || retain_variables_obj == Py_False,
      "retain_variables argument is expected to be a bool, but got %s",
      THPUtils_typename(retain_variables_obj));
  THPUtils_assert(PyTuple_Check(variables), "variables argument is expected to "
      "be a tuple, but got %s", THPUtils_typename(variables));
  THPUtils_assert(PyTuple_Check(grad_variables), "variables argument is "
      "expected to be a tuple, but got %s", THPUtils_typename(grad_variables));

  Py_ssize_t num_variables = PyTuple_GET_SIZE(variables);
  Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_variables);
  THPUtils_assert(num_variables == num_gradients, "got %ld variables and %ld "
      "gradients", num_variables, num_gradients);

  variable_list vars(num_variables);
  tensor_list grads(num_variables);
  for (int i = 0; i < num_variables; i++) {
    PyObject *variable = PyTuple_GET_ITEM(variables, i);
    THPUtils_assert(THPVariable_Check(variable), "element %d of variables "
        "tuple is not a Variable", i);
    vars[i] = ((THPVariable*)variable)->cdata;

    PyObject *grad = PyTuple_GET_ITEM(grad_variables, i);
    if (THPModule_isTensor(grad)) {
      grads[i] = torch::createTensor(grad);
    } else {
      THPUtils_assert(grad == Py_None,
          "element %d of gradients tuple is not a Tensor or None", i);
      THPUtils_assert(!vars[i]->requires_grad,
          "element %d of gradients tuple is None, but the corresponding Variable requires grad");
    }
  }

  try {
    AutoNoGIL no_gil;
    engine.backward(vars, grads, retain_variables);
  } catch (python_error &e) {
    e.restore();
    return nullptr;
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
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
